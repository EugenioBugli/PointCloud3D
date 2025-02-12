import torch
import numpy as np
import open3d as o3d
import plotly.graph_objects as go
import trimesh
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import DatasetFolder
from sklearn.model_selection import train_test_split

class FAUST_Dataset(Dataset):
    """
        This class is used to load a partition of the FAUST dataset. Before using this you must access the file via DatasetFolder.
        The Encoder takes in input 3000 points sampled from the surface of a mesh + some noise
        The Decoder takes in input 2048 points sampled from the bounding box of the mesh
        The augmentation is applied only to the encoder input
    """

    def __init__(self, scan_files, reg_files=None, encoder_input_size=3000, decoder_input_size=2048, partition_type="TRAIN",
                 occupancy_threshold=0.05, encoder_noise_std=0.05, encoder_aug=False, transform=None):
        super(FAUST_Dataset, self).__init__()
        self.scan_files = scan_files # list of .ply files
        self.reg_files = reg_files if partition_type in ["TRAIN", "VAL"] else None
        self.partition_type = partition_type
        self.encoder_input_size = encoder_input_size # 3000
        self.decoder_input_size = decoder_input_size # 2048
        self.occupancy_threshold = occupancy_threshold
        self.encoder_noise_std = encoder_noise_std
        self.encoder_aug = encoder_aug
        self.transform = transform

        self.scans, self.regs, self.query = self.extractClouds()
        self.labels = self.extractLabels()

    def __getitem__(self, index):
        if self.partition_type in ["TRAIN", "VAL"]: # needed since only train and val set have the registration set

            scan = self.scans[index]
            reg = self.regs[index]
            query = self.query[index]
            scan_path = self.scan_files[index]
            reg_path = self.reg_files[index]
            label = self.labels[index]

            if self.encoder_aug:
                scan = self.applyAugmentation(scan)

            if self.transform:
                return self.transform(scan).to(torch.float32).squeeze(0), self.transform(reg).to(torch.float32).squeeze(0), self.transform(query).to(torch.float32).squeeze(0), scan_path, reg_path, self.transform(label).to(torch.float32).squeeze(0)
            else:
                return scan, reg, query, scan_path, reg_path, label
        else:
            scan = self.scans[index]
            query = self.query[index]
            scan_path = self.scan_files[index]
            label = self.labels[index]

            if self.transform:
                return self.transform(scan).to(torch.float32).squeeze(0), self.transform(query).to(torch.float32).squeeze(0), scan_path, self.transform(label).to(torch.float32).squeeze(0)
            else:
                return scan, query, scan_path, label

    def __len__(self):
        return len(self.scan_files)

    def extractClouds(self):
        # use opend3d to open .ply files
        scans = []
        regs = []
        query = []

        for i in range(len(self.scan_files)):
            f = self.scan_files[i]
            cloud = o3d.io.read_point_cloud(f)
            scans.append(self.surfaceSampling(cloud, sampling_type="RANDOM"))

            if self.partition_type in ["TRAIN", "VAL"]:
                f = self.reg_files[i] # I need a mesh to perform sampling -> better to use the registration if i have it

            query.append(self.boxSampling(f))

        if self.partition_type in ["TRAIN", "VAL"]:
            for f in self.reg_files:
                regs.append(np.asarray(o3d.io.read_point_cloud(f).points))

            return scans, regs, query
        else:
            return scans, None, query

    def extractLabels(self):
        # need meshes to obtain the ground truth occupancy
        labels = []

        for i in range(len(self.scan_files)):

            mesh_file = self.reg_files[i] if self.partition_type in ["TRAIN", "VAL"] else self.scan_files[i] # use the registration if you can (has less points)
            mesh = trimesh.load_mesh(mesh_file)

            l = np.zeros(self.decoder_input_size, dtype=np.int32)
            dist = mesh.nearest.signed_distance(self.query[i])

            l[dist >= 0] = 1 # inside or on the surface
            l[dist > -self.occupancy_threshold] = 1 # outside in the threshold range
            labels.append(l.reshape(-1,1))

        return np.asarray(labels)

    def surfaceSampling(self, cloud, sampling_type="RANDOM"):
        # cloud is assumed to be a open3d object
        points = np.asarray(cloud.points)
        indices = np.random.choice(len(points), size=self.encoder_input_size)
        sampled_cloud = points[indices]

        noise = np.random.normal(0, self.encoder_noise_std, sampled_cloud.shape)
        sampled_cloud = sampled_cloud + noise
        return sampled_cloud


    def boxSampling(self, file_path, factor=5):
        # this path is either from the registration (TRAIN or VAL) or from the entire scan
        mesh = trimesh.load_mesh(file_path)
        min_bound, max_bound = mesh.bounds

        padding = 0.05 * (max_bound - min_bound)
        min_bound = min_bound - padding
        max_bound = max_bound + padding

        possible_candidates = np.random.uniform(low=min_bound, high=max_bound, size=(self.decoder_input_size*factor, 3))
        distances = mesh.nearest.signed_distance(possible_candidates)
        occupancy = np.zeros(self.decoder_input_size*factor, dtype=np.int32)
        occupancy[distances >= 0] = 1
        occupancy[distances > -self.occupancy_threshold] = 1

        # I would like to have balanced classes
        half_size = self.decoder_input_size // 2

        inside_indices = np.where(occupancy == 1)[0]
        outside_indices = np.where(occupancy == 0)[0]

        if len(inside_indices) < half_size: # need replacement
            inside_choice = np.random.choice(inside_indices, half_size, replace=True)
        else:
            inside_choice = np.random.choice(inside_indices, half_size, replace=False)

        if len(outside_indices) < half_size: # need replacement
            outside_choice = np.random.choice(outside_indices, half_size, replace=True)
        else:
            outside_choice = np.random.choice(outside_indices, half_size, replace=False)

        selected_indices = np.concatenate([inside_choice, outside_choice])
        np.random.shuffle(selected_indices)

        return possible_candidates[selected_indices]

    def applyAugmentation(self, cloud):
        angle = np.random.uniform(0, 2*np.pi, 3) # random RPY angles

        roll = angle[0]
        pitch = angle[1]
        yaw = angle[2]

        R_r = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])

        R_p = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])

        R_y = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])

        R = R_r @ R_p @ R_y
        scale = np.random.uniform(0.9, 1.1)
        translation = np.random.uniform(-0.1, 0.1, (1, 3))

        return (cloud @ R.T) * scale + translation

# Save your preprocessed Dataset:
def SaveDataset(dataset, path):
    torch.save(dataset, "/content/drive/MyDrive/CV/NewData/"+path)

# Load your preprocessed Dataset:
def LoadDataset(path):
    return torch.load("/content/drive/MyDrive/CV/NewData/"+path, weights_only=False)

def openDataFiles(training_path, test_path, val_size):
    """
    Imports all the .ply files from the specified folders and partitions
    the training data into training and validation sets.
    """
    training_dataset = DatasetFolder(
        root=training_path,
        loader=o3d.io.read_point_cloud,
        extensions=('ply',),
        allow_empty=True,
    )

    test_dataset = DatasetFolder(
        root=test_path,
        loader=o3d.io.read_point_cloud,
        extensions=('ply',),
        allow_empty=True,
    )

    unsorted_training_scan_files = [sample for sample, t in training_dataset.samples if t == 2]
    training_scan_files = sorted(unsorted_training_scan_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))

    unsorted_training_reg_files = [sample for sample, t in training_dataset.samples if t == 1]
    training_reg_files = sorted(unsorted_training_reg_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))

    # train-validation split.
    train_scan_files, val_scan_files, train_reg_files, val_reg_files = train_test_split(
        training_scan_files, training_reg_files, test_size=val_size, random_state=15)

    unsorted_test_scan_files = [sample for sample, t in test_dataset.samples if t == 1]
    test_scan_files = sorted(unsorted_test_scan_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    test_reg_files = test_scan_files

    return train_scan_files, train_reg_files, val_scan_files, val_reg_files, test_scan_files, test_reg_files