import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

import os
import numpy as np


class ResBlock(nn.Module):
    """
        This class is used to define a Residual Block, which is one of the main component of the ResNetPointNet architecture
    """

    def __init__(self, in_dim=64, h_dim=32, out_dim=64):
        super(ResBlock, self).__init__()

        #> First part of the Block

        self.fc1 = nn.Linear(
            in_dim,
            h_dim
        )
        self.bn1 = nn.BatchNorm1d(h_dim)

        #> Second part of the Block

        self.fc2 = nn.Linear(
            h_dim,
            out_dim
        )
        self.bn2 = nn.BatchNorm1d(out_dim)

        #> Skip connection

        if in_dim != out_dim:
            # size mismatch
            self.residual = nn.Linear(in_dim, out_dim)
        else:
            # same size
            self.residual = None


    def forward(self, x):

        # first part of the block
        first_part = F.relu(self.bn1(self.conv1(x)))

        # second part of the block
        second_part = self.bn2(self.conv2(first_part))

        if self.residual is None:
            # no size mismatch
            self.residual = x
        else:
            # transformation if there is a size mismatch
            self.residual = self.residual(x)

        # add residual connection
        third_part = second_part + self.residual(x)

        return F.relu(third_part)