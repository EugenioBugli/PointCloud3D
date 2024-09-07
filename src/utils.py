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


def CoordinateNormalization(input_cloud, planeType='xz'):
    """
        This Function is used to Normalize coordinates from the input point cloud.
        After this operation the cloud will be centered at the origin and all coordinates will be inside a standard range [0,1].
        The results will be the normalized point cloud focused only on the plane coordinate. This means that we will have only 2 coordinates for
        our points instead of 3.

        Args:
            input_cloud : (batch_size, n_points, 3) tensor
            planeType: (str) represent the kind of plane that we are using (xz, xy, yz)

        Returns:
            norm_cloud : (batch_size, n_points, 2) tensor
    """
    # Extract coordinates according to the planeType you are using
    if planeType == 'xz':
        plane = input_cloud[:, :, [0,2]] # (b,p,3) -> (b,p,2)
    elif planeType == 'xy':
        plane = input_cloud[:, :, [0,1]] # (b,p,3) -> (b,p,2)
    elif planeType == 'yz':
        plane = input_cloud[:, :, [1,2]] # (b,p,3) -> (b,p,2)


    norm_cloud = (plane - plane.min())/(plane.max() - plane.min() + 10e-6)

    return norm_cloud


def PlaneCoordinate2Index(input_cloud, voxel_size=0.2):
    """
        This function is used to Obtain an Index from the Plane Coordinates of the original input cloud.
        We need to transform Coordinates into Indices because we would like to discretize space into Voxels.
        This index is extremely important since is used to perform Local Pooling, one of the operation of the ResNetPointNet Architecture.

        Args:
            input_cloud : (batch_size, n_points, 2) tensor
            voxel_size : (float) represent how large the Voxels (Local Regions) are (default = 0.2)

        Returns:
            index : (batch_size, n_points, 2) tensor
    """
    return torch.floor(input_cloud/voxel_size).long()

    
def LocalPooling(norm_cloud, indices, input_cloud):
    """
        Unlike the Vanilla PointNet, we perform Local Max Pooling on the output of each ResBlock and then we concatenate the result
        with the features before the operation.
        In order to perform this operation we have to normalize the planes coordinates and transform into indices.
        These indices represent which voxel each point belongs to.

        ResBlock ________________
            |                    |
            |                    |
        LocalPool                |
            |                    |
            v                    |
            +   <----------------'

        Args:
            norm_cloud: (batch_size, num_points, 2) tensor that represent the normalized input cloud related to a specific plane (xz)
            indices: (batch_size, num_points, 2) tensor that represent the indices of the input cloud related to the Voxel
            input_cloud: (batch_size, num_points, n_features) tensor that represent the input cloud

        Returns:
            pool_cloud: tensor of shape (batch_size, num_points, n_features)
    """

    pool_cloud = torch.zeros_like(input_cloud)

    for b in range (norm_cloud.shape[0]):
        buckets = dict()

        # each point in the batch will have its own index along with the ones given by the PlaneCoordinate2Index transformation
        for i, voxel_idx in enumerate(indices[b]):
            # each point is identified by its own key (index)

            voxel_key = tuple(voxel_idx.tolist())
            if voxel_key not in buckets:
                buckets[voxel_key] = []

            # insert inside the correct bucket the point
            buckets[voxel_key].append(input_cloud[b][i])


        for voxel_key, points_list in buckets.items():
            # inside each buckets you have a list of points (tensors)

            if len(points_list) > 0:
                # extract the local maximum
                bucket_points = torch.stack(points_list)
                max_p = torch.max(bucket_points, dim=0)[0]

                # assign the maximum to all the points inside the buckets

                for i, voxel_idx in enumerate(indices[b]):
                    if torch.equal(voxel_idx, torch.Tensor(voxel_key)):
                        pool_cloud[b][i] = max_p
    return pool_cloud
