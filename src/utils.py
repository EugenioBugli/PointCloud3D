import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
import os


class FAUST_Dataset(Dataset):
    """
        This class is used to load the FAUST dataset
    """

    def __init__(self, data_path, transform=None):

        self.data_path = data_path
        self.clouds = self.importData(self.data_path)

    def __getitem__(self, index):

      return torch.Tensor.numpy(self.clouds[index])

    def __len__(self):
        # num samples, num points, coordinates
        return self.clouds.size()[0]

    def importData(self, data_path, SAMPLING=True, SAMPLING_SIZE=1024):
        """
        This function is used to import the Dataset from Google Drive

        Args:
            data_path (str): path to the dataset
            SAMPLING (bool): whether to sample the Point Cloud or not
            SAMPLING_SIZE (int): number of points to sample from the complete Point Cloud

        Returns:
            torch.tensor: matrix of Point Clouds
        """

        input_mat = []
        s = 0
        directory_path = data_path + "/scans/"
        for file in os.listdir(directory_path):
            if file.endswith(".ply"):
                file_mesh = o3d.io.read_triangle_mesh(directory_path+file)
                vert = np.asarray(file_mesh.vertices)
                if SAMPLING:
                    indices = np.random.choice(len(vert), size=SAMPLING_SIZE)
                    vert = vert[indices]
                s += len(vert)
                input_mat.append(vert)

        return torch.tensor(np.array(input_mat)).to(torch.float32)

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


def CoordinateNormalization(input_cloud):
    """
        This Function is used to Normalize coordinates from the input point cloud.
        After this operation the cloud will be centered at the origin and all coordinates will be inside a standard range [0,1]

        Args:
            input_cloud : (batch_size, n_points, 3) tensor

        Returns:
            norm_cloud : (batch_size, n_points, 3) tensor
    """

    return 0