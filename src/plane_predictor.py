import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

import os
import numpy as np


class SimplePointNet(nn.Module):
    """
        This class is used to define a simple variant of the PointNet Model, which is one of the main components of the Plane Predictor Network.
        This Network will provide us a global context of the Input Point Clouds
        Architecture Design:

            @ INPUT: Tensor of shape (batch_size, num_points, 3) which represent the Input Point Clouds

            > Fully Connected Layer (3, 64)

              |> Fully Connected Layer (64, 32)
            2*|> Global Max Pooling
              |> Concatenation btw Pooled and unpooled features

            > Fully Connected Layer (64, 32)
            > Global Max Pooling

            @ OUTPUT: Tensor of shape (batch_size, num_points, 32) which will be used by the rest of the Plane Predictor
    """

    def __init__(self, batch_size, in_dim=64, n_points=1024, hid_dim=32, out_dim=64):
        super(SimplePointNet, self).__init__()

        self.batch_size = batch_size
        self.n_points = n_points

        self.initial_fc = nn.Linear(in_features=3, out_features=in_dim)

        self.fc1 = nn.Linear(in_features=in_dim, out_features=hid_dim)

        self.fc2 = nn.Linear(in_features=in_dim, out_features=hid_dim)

        self.final_fc = nn.Linear(in_features=in_dim, out_features=hid_dim)

        self.pool = nn.AdaptiveMaxPool1d(output_size=1)

    def forward(self, x):

        x = self.initial_fc(x) # (b,p,3) --> (b,p,64)

        x1 = self.fc1(x) # (b,p,64) --> (b,p,32)
        x1_t = x1.transpose(1,2) # (b,p,32) -> (b,32,p)
        pool_x1 = self.pool(x1_t) # (b,32,p) -> (b,32,1)
        exp_pool_x1 = pool_x1.transpose(1,2).expand(self.batch_size, self.n_points, 32) # (b,32,1) -> (b,1,32) -> (b,p,32)
        concat_x1 = torch.cat([x1, exp_pool_x1], dim=2) # (b,p,32) | (b,p,32) -> (b,p,64)

        x2 = self.fc2(concat_x1) # (b,p,64) -> (b,p,32)
        x2_t = x2.transpose(1,2) # (b,p,32) -> (b,32,p)
        pool_x2 = self.pool(x2_t) # (b,32,p) -> (b,32,1)
        exp_pool_x2 = pool_x2.transpose(1,2).expand(self.batch_size, self.n_points, 32) # (b,32,1) -> (b,1,32) -> (b,p,32)
        concat_x2 = torch.cat([x2, exp_pool_x2], dim=2) # (b,p,32) | (b,p,32) -> (b,p,64)

        pre_pool_out = self.final_fc(concat_x2) # (b,p,64) -> (b,p,32)
        pre_pool_out_t = pre_pool_out.transpose(1,2) # (b,p,32) -> (b,32,p)
        out = self.pool(pre_pool_out_t) # (b,32,p) -> (b,32,1)

        return out.transpose(1,2) # (b,32,1) -> (b,1,32)

class PlanePredictor(nn.Module):
    """
        This class is used to define the Plane Predictor of our Architecture, which will predict the plane parameters of L dynamic planes
        Architecture design:

            @ INPUT: Tensor of shape (batch_size, num_points, 3) which represent Point Clouds

            > Simple PointNet which learns the global context of the input point clouds
            > This information is encoded into one global feature by using Max Pooling
            > 4 Fully Connected Layers with hidden dimension = 32
            > L Shallow Networks with hidden dimension = 3 which will give us the Predicted Plane Parameters
            > L Fully Connected Layers with 1 layer and hidden dimension = D (same as point cloud encoder hidden dimension)
            > Each plane-specific feature is expanded to N x D to match the output of the point cloud encoder, which will be summed together


            @ OUTPUT: Tensor of shape (batch_size, num_points, 32) which will be processed into U-Net

    """
    def __init__(self, in_dim=32, n_points=1024, n_fc=4, L=3):
        super(PlanePredictor, self).__init__()

        self.pointNet = SimplePointNet(batch_size=1)
        self.n_points = n_points

        # 4 FC layers with hidden dim = 32

        self.four_fc = nn.ModuleList(
            [nn.Linear(in_dim, in_dim) for i in range(n_fc)]
        )

        # Plane parameters (L shallow networks with hidden dim = 3)

        self.first_shallow = nn.Linear(in_dim, 3)
        self.shallows = nn.ModuleList(
            [nn.Linear(3, 3) for i in range(L-1)]
        )

        # L FC layers with hidden dim = 32

        self.first_fc = nn.Linear(3, in_dim)
        self.L_fc = nn.ModuleList(
            [nn.Linear(in_dim, in_dim) for i in range(L-1)]
        )

    def forward(self, x):

        print("input:",x.shape)

        flc = self.pointNet(x) # (b,p,3) -> (b,1,32)

        print("ptn", flc.shape)

        # 4 FC layers with hidden dim = 32

        for fc in self.four_fc:
            flc = F.relu(fc(flc))  # (b,1,32) -> (b,1,32)
            print("flc",flc.shape)

        # Plane parameters ( L Shallow Networks )

        first_sh = F.relu(self.first_shallow(flc)) # (b,1,32) -> (b,1,3)
        print("first shallow", first_sh.shape)

        shal = first_sh

        for s in self.shallows:
            shal = F.relu(s(shal)) # (b,1,3) -> (b,1,3)

        print("plane params", shal.shape)

        # L FC networks dim = 32

        first_fc_L = F.relu(self.first_fc(shal)) # (b,1,3) -> (b,1,32)
        print("first fully", first_fc_L.shape)

        L_fully = first_fc_L

        for fc in self.L_fc:
            L_fully = F.relu(fc(L_fully))  # (b,1,32) -> (b,1,32)

        print("L fully", first_fc_L.shape)

        # Expansion

        out = L_fully.expand(x.shape[0], self.n_points, first_fc_L.shape[-1]) # (b,1,32) -> (b,p,32)

        return out