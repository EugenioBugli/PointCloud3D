import torch
import numpy as np
import open3d as o3d
import plotly.graph_objects as go
from torch.utils.data import DataLoader, Dataset

class FAUST_Dataset(Dataset):
    """
        This class is used to load a partition of the FAUST dataset. Before using this you must access the file via DatasetFolder.
    """

    def __init__(self, scan_files, reg_files=None, sampling_type="RANDOM", sampling_size=1024, partition="TRAIN", transform=None):
        super(FAUST_Dataset, self).__init__()
        self.scan_files = scan_files # list of files .ply
        self.partition = partition
        self.reg_files = reg_files if partition in ["TRAIN", "VAL"] else None # list of files .ply
        self.sampling_type = sampling_type
        self.sampling_size = sampling_size
        self.transform = transform

        self.scans, self.regs = self.extractClouds()

    def __getitem__(self, index):
        if self.partition in ["TRAIN", "VAL"]:
            if self.transform:
                return self.transform(self.scans[index]).to(torch.float32).squeeze(0), self.transform(self.regs[index]).to(torch.float32).squeeze(0)
            return self.scans[index], self.regs[index]
        else: # test case
            if self.transform:
                return self.transform(self.scans[index]).to(torch.float32).squeeze(0)
            return self.scans[index]

    def __len__(self):
        return len(self.scan_files)

    def extractClouds(self):
        # we have to open the files .ply and transform them into point clouds:

        scans = []
        regs = []

        for i in range(len(self.scan_files)):
            s = o3d.io.read_point_cloud(self.scan_files[i])
            scans.append(self.SamplingFunction(s))

            if self.partition in ["TRAIN", "VAL"]:
                r = np.asarray(o3d.io.read_point_cloud(self.reg_files[i]).points)
                regs.append(r)

        return scans, regs if regs else None

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

    def SamplingFunction(self, cloud):
        """
            This function is used to sample a small Subset of points from the Point Clouds inside our Dataset.

            @INPUT :
                > cloud : Point Cloud extracted from .ply file

            @OUTPUT :
                > sampled_cloud : Sampled Point Cloud
        """

        if self.sampling_type == 'RANDOM':
            points = np.asarray(cloud.points)
            indices = np.random.choice(len(points), size=self.sampling_size)
            sampled_cloud = points[indices]
        if self.sampling_type == 'IMPORTANCE':
            numOfNeighbors = 20
            # estimate normal vectors to the surface at each point of the cloud
            cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=numOfNeighbors))
            tree = o3d.geometry.KDTreeFlann(cloud) # faster
            # loop over the points and compute curvature
            curvature = np.zeros(len(cloud.points))
            for i in range(len(cloud.points)):
                # find indices of the neighbors
                [_ , idx, _] = tree.search_knn_vector_3d(cloud.points[i], numOfNeighbors)
                neighbors = np.asarray(cloud.points)[idx, :]
                # compute covariance matrix for each point
                covarianceMat = np.cov(neighbors.T)
                # extract eigenvalues
                eigen, _ = np.linalg.eigh(covarianceMat)
                # compute curvature
                curvature[i] = min(eigen) / sum(eigen)
            # extract the best SamplingPoints points
            maxCurvaturePoints = curvature.argsort()[-self.SamplingSize:]
            sampled_cloud = np.asarray(cloud.points)[maxCurvaturePoints]

        return sampled_cloud

# Save your preprocessed Dataset:
def SaveDataset(dataset, path):
    torch.save(dataset, "/content/drive/MyDrive/CV/PreProcessed/"+path)

# Load your preprocessed Dataset:
def LoadDataset(path):
    return torch.load("/content/drive/MyDrive/CV/PreProcessed/"+path, weights_only=False)
