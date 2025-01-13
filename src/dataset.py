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
    """

    def __init__(self, scan_files, reg_files=None, sampling_size=2048, partition="TRAIN", transform=None):
        super(FAUST_Dataset, self).__init__()
        self.scan_files = scan_files # list of files .ply
        self.partition = partition
        self.reg_files = reg_files if partition in ["TRAIN", "VAL"] else None # list of files .ply
        self.sampling_size = sampling_size
        self.transform = transform

        self.scans, self.regs, self.query = self.extractClouds()
        self.labels = self.extractLabels()

    def __getitem__(self, index):
        if self.partition in ["TRAIN", "VAL"]:
            if self.transform:
                return self.transform(self.scans[index]).to(torch.float32).squeeze(0), self.transform(self.regs[index]).to(torch.float32).squeeze(0), self.transform(self.query[index]).to(torch.float32).squeeze(0), self.scan_files[index], self.reg_files[index], self.transform(self.labels[index]).to(torch.float32).squeeze(0)
            return self.scans[index], self.regs[index], self.query[index], self.scan_files[index], self.reg_files[index], self.labels[index]
        else: # test case
            if self.transform:
                return self.transform(self.scans[index]).to(torch.float32).squeeze(0), self.transform(self.query[index]).to(torch.float32).squeeze(0), self.scan_files[index], self.transform(self.labels[index]).to(torch.float32).squeeze(0)
            return self.scans[index], self.query[index], self.scan_files[index], self.labels[index]

    def __len__(self):
        return len(self.scan_files)

    def extractClouds(self):
        # we have to open the files .ply and transform them into point clouds:

        scans = []
        regs = []
        query = []

        for i in range(len(self.scan_files)):
            s = o3d.io.read_point_cloud(self.scan_files[i])
            scans.append(self.SamplingFunction(s, operation="RANDOM"))

            if self.partition in ["TRAIN", "VAL"]:
                r = np.asarray(o3d.io.read_point_cloud(self.reg_files[i]).points)
                regs.append(r)

            q = o3d.io.read_point_cloud(self.scan_files[i])
            query.append(self.SamplingFunction(q, operation="QUERY"))

        return scans, regs if regs else None, query

    def extractLabels(self, threshold=0.05):

        # registration which will gives us the watertight mesh
        # sampled_cloud is the point cloud obtained by uniform sampling in the space of the cloud
        label_container = []

        for cloud in range(len(self.scan_files)):
            if self.partition in ["TRAIN", "VAL"]:
                reg_name = self.reg_files[cloud]
            else:
                reg_name = self.scan_files[cloud]
            mesh = trimesh.load_mesh(reg_name)
            labels = np.zeros(self.query[cloud].shape[0], dtype=np.int32)
            distances = mesh.nearest.signed_distance(self.query[cloud])
            # outside = negative distance
            # inside = positive distance
            labels[distances >= 0] = 1
            labels[distances > -threshold] = 1
            label_container.append(labels.reshape(-1,1))

        return np.asarray(label_container)


    def plotCloud(self, cloud):
        """
        This function is used to plot the Point Cloud

        Args:
            cloud (np.array): Point Cloud

        Returns:
            None
        """
        if len(cloud.shape) == 3:
            # when using DataLoader you have a shape (1, SAMPLING_SIZE, 3)
            cloud = cloud[0]

        fig = go.Figure(
            data=[
            go.Scatter3d(
                x =cloud[:,0], y=cloud[:,1], z=cloud[:,2],
                mode = 'markers',
                marker = dict(size=0.5, color=[])
            )
        ],
        layout=dict(
            scene=dict(
                xaxis=dict(visible=True),
                yaxis=dict(visible=True),
                zaxis=dict(visible=True),
                )
            )
        )
        fig.show()

    def SamplingFunction(self, cloud, operation=None):
        """
            This function is used to sample a small Subset of points from the Point Clouds inside our Dataset.

            @INPUT :
                > cloud : Point Cloud extracted from .ply file
                > operation : specifies which type of operation must be performed (sampling from the surface or over all the space)

            @OUTPUT :
                > sampled_cloud : Sampled Point Cloud
        """
        if operation == "RANDOM": # sample from the surface of the cloud (sampling_size=3000)
            points = np.asarray(cloud.points)
            indices = np.random.choice(len(points), size=3000)
            sampled_cloud = points[indices]

        if operation == "QUERY": # uniform sample from all the space (points will be both outside and inside the cloud) (sampling_size=2048)
            axis_bounding_box = cloud.get_axis_aligned_bounding_box()
            min_bound = axis_bounding_box.min_bound
            max_bound = axis_bounding_box.max_bound

            padding = 0.1*(max_bound - min_bound)

            sampled_cloud = np.random.uniform(np.asarray(min_bound-padding), np.asarray(max_bound-padding), size=(self.sampling_size, 3))

        return sampled_cloud

# Save your preprocessed Dataset:
def SaveDataset(dataset, path):
    torch.save(dataset, "/content/drive/MyDrive/CV/SavedData/"+path)

# Load your preprocessed Dataset:
def LoadDataset(path):
    return torch.load("/content/drive/MyDrive/CV/SavedData/"+path, weights_only=False)

def openDataFiles(training_path, test_path, val_size):
    """
        This function is used to import all the .ply files from the folders. Training is partitioned into train and validation set directly here.
    """

    training_dataset = DatasetFolder(
        root = training_path,
        loader = o3d.io.read_point_cloud,
        extensions = ('ply',),
        allow_empty = True,
        )

    test_dataset = DatasetFolder(
        root = test_path,
        loader = o3d.io.read_point_cloud,
        extensions = ('ply',),
        allow_empty = True,
    )

    unsorted_training_scan_files = [sample for sample, t in training_dataset.samples if t == 2]
    training_scan_files = sorted(unsorted_training_scan_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))

    unsorted_training_reg_files = [sample for sample, t in training_dataset.samples if t == 1]
    training_reg_files = sorted(unsorted_training_reg_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))

    # perform train-validation split :

    train_scan_files, val_scan_files, train_reg_files, val_reg_files = train_test_split(training_scan_files, training_reg_files, test_size=val_size, random_state=15)


    unsorted_test_scan_files = [sample for sample, t in test_dataset.samples if t == 1]
    test_scan_files = sorted(unsorted_test_scan_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    test_reg_files = test_scan_files # check here
    # we don't have any registration for the test set --> use instead the complete point cloud


    return train_scan_files, train_reg_files, val_scan_files, val_reg_files, test_scan_files, test_reg_files