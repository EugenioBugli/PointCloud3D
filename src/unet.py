import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class UNet(nn.Module):
    """
        This class is used to define the UNet of our Architecture, which is the final part of our Encoder.
        Architecture design:

        We use a U-Net to process the plane features and adapt a modified implementation from[5].
        We set the input and output feature dimensions to 32 and choose the depth of the U-Net such that the receptive field is equal to the size of the feature plane.
        In doing so, we set a depth of 4 for our experiments with ShapeNet dataset (64^2 grids) and a depth of 5 for our scene experiments (128^2 grids).

        @ INPUT: Tensor of shape (batch_size, num_points, 32)
            # Encoder:
                > Conv2D
                > MaxPool2D

                > Conv1D
                > MaxPool1D

                > Conv1D
                > MaxPool1D

            # BottleNeck:
                >
                >
                >
                >
                >
            # Decoder:
                > UpConv2D
                > MaxPool2D

                > UpConv2D
                > MaxPool2D

                > UpConv2D
                > MaxPool2D

        @ OUTPUT: Tensor of shape (batch_size, num_points, 32)
    """
    def __init__(self, in_dim=32, out_dim=32, n_points=1024):
        super(UNet, self).__init__()

        self.pool = nn.MaxPool1d(
            kernel_size=2,
            stride=2
        )

        # Encoder

        self.e_conv1 = nn.Conv1d(
            in_channels=in_dim,
            out_channels=in_dim,
            kernel_size=3,
            stride=1,
            padding=1)

        self.e_conv2 = nn.Conv1d(
            in_channels=in_dim,
            out_channels=in_dim,
            kernel_size=3,
            stride=1,
            padding=1)

        self.e_conv3 = nn.Conv1d(
            in_channels=in_dim,
            out_channels=in_dim,
            kernel_size=3,
            stride=1,
            padding=1)

        # Bottleneck

        self.b_conv = nn.Conv1d(
            in_channels=in_dim,
            out_channels=in_dim,
            kernel_size=3,
            stride=1,
            padding=1)

        # Decoder

        self.d_upconv1 = nn.ConvTranspose1d(
            in_channels=in_dim,
            out_channels=in_dim,
            kernel_size=2,
            stride=2)

        self.d_conv1 = nn.Conv1d(
            in_channels=in_dim*2,
            out_channels=in_dim,
            kernel_size=3,
            stride=1,
            padding=1)

        self.d_upconv2 = nn.ConvTranspose1d(
            in_channels=in_dim,
            out_channels=in_dim,
            kernel_size=2,
            stride=2)

        self.d_conv2 = nn.Conv1d(
            in_channels=in_dim*2,
            out_channels=in_dim,
            kernel_size=3,
            stride=1,
            padding=1)

        self.d_upconv3 = nn.ConvTranspose1d(
            in_channels=in_dim,
            out_channels=in_dim,
            kernel_size=2,
            stride=2)

        self.d_conv3 = nn.Conv1d(
            in_channels=in_dim*2,
            out_channels=out_dim,
            kernel_size=3,
            stride=1,
            padding=1)


    def forward(self, x):

        x = x.transpose(1,2) # (b,p,32) -> (b,32,p)

        e_1 = self.e_conv1(x)
        print("conv1",e_1.shape)
        e_1_pool = self.pool(F.relu(e_1))
        print("pool1",e_1_pool.shape)

        e_2 = self.e_conv1(e_1_pool)
        print("conv2",e_2.shape)
        e_2_pool = self.pool(F.relu(e_2))
        print("pool2",e_2_pool.shape)

        e_3 = self.e_conv1(e_2_pool)
        print("conv3",e_2.shape)
        e_3_pool = self.pool(F.relu(e_3))
        print("pool3",e_3_pool.shape)

        print("\n\n")

        b = F.relu(self.b_conv(e_3_pool))
        print("bottleneck",b.shape)

        print("\n\n")

        d_1 = self.d_upconv1(b)
        print("upconv1",d_1.shape)
        d_1 = torch.cat([d_1, e_3], dim=1)
        print("concat1", d_1.shape)
        d_1 = F.relu(self.d_conv1(d_1))
        print("conv1",d_1.shape)

        d_2 = self.d_upconv1(d_1)
        print("upconv2",d_2.shape)
        d_2 = torch.cat([d_2, e_2], dim=1)
        print("concat2", d_2.shape)
        d_2 = F.relu(self.d_conv2(d_2))
        print("conv2",d_2.shape)

        d_3 = self.d_upconv1(d_2)
        print("upconv3",d_3.shape)
        d_3 = torch.cat([d_3, e_1], dim=1)
        print("concat3", d_3.shape)
        d_3 = F.relu(self.d_conv3(d_3))
        print("conv3",d_3.shape)

        return d_3.transpose(1,2)