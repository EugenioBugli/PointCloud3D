import torch
import torch.nn as nn

class UNet(nn.Module):
    """
        This class is used to define the UNet of our Architecture, which is the final part of our Encoder.
        Architecture design:

        @ INPUT: Tensor of shape (batch_size*L, 32, H, W)
            # Encoder
            # BottleNeck
            # Decoder

        @ OUTPUT: Tensor of shape (batch_size*L, 32, H, W)
    """
    def __init__(self, in_dim=32, out_dim=32, features_dim=64, n_points=1024):
        super(UNet, self).__init__()

        self.pool1 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # Encoder

        ## Block 1

        self.e_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_dim,
                out_channels=features_dim,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(num_features=features_dim),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=features_dim,
                out_channels=features_dim,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(num_features=features_dim),
            nn.ReLU(),
        )

        self.p_1 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        ## Block 2

        self.e_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=features_dim,
                out_channels=features_dim*2,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(num_features=features_dim*2),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=features_dim*2,
                out_channels=features_dim*2,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(num_features=features_dim*2),
            nn.ReLU(),
        )

        self.p_2 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        ## Block 3

        self.e_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=features_dim*2,
                out_channels=features_dim*4,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(num_features=features_dim*4),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=features_dim*4,
                out_channels=features_dim*4,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(num_features=features_dim*4),
            nn.ReLU(),
        )

        self.p_3 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        ## Block 4

        self.e_4 = nn.Sequential(
            nn.Conv2d(
                in_channels=features_dim*4,
                out_channels=features_dim*8,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(num_features=features_dim*8),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=features_dim*8,
                out_channels=features_dim*8,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(num_features=features_dim*8),
            nn.ReLU(),
        )

        self.p_4 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # Bottleneck

        self.b = nn.Sequential(
            nn.Conv2d(
                in_channels=features_dim*8,
                out_channels=features_dim*16,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(num_features=features_dim*16),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=features_dim*16,
                out_channels=features_dim*16,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(num_features=features_dim*16),
            nn.ReLU(),
        )

        # Decoder

        ## Block 1

        self.d_upconv1 = nn.ConvTranspose2d(
            in_channels=features_dim*16,
            out_channels=features_dim*8,
            kernel_size=2,
            stride=2)

        self.d_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=features_dim*16,
                out_channels=features_dim*8,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(num_features=features_dim*8),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=features_dim*8,
                out_channels=features_dim*8,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(num_features=features_dim*8),
            nn.ReLU(),
        )

        ## Block 2

        self.d_upconv2 = nn.ConvTranspose2d(
            in_channels=features_dim*8,
            out_channels=features_dim*4,
            kernel_size=2,
            stride=2)

        self.d_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=features_dim*8,
                out_channels=features_dim*4,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(num_features=features_dim*4),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=features_dim*4,
                out_channels=features_dim*4,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(num_features=features_dim*4),
            nn.ReLU(),
        )

        ## Block 3

        self.d_upconv3 = nn.ConvTranspose2d(
            in_channels=features_dim*4,
            out_channels=features_dim*2,
            kernel_size=2,
            stride=2)

        self.d_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=features_dim*4,
                out_channels=features_dim*2,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(num_features=features_dim*2),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=features_dim*2,
                out_channels=features_dim*2,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(num_features=features_dim*2),
            nn.ReLU(),
        )

        ## Block 4

        self.d_upconv4 = nn.ConvTranspose2d(
            in_channels=features_dim*2,
            out_channels=features_dim,
            kernel_size=2,
            stride=2)

        self.d_4 = nn.Sequential(
            nn.Conv2d(
                in_channels=features_dim*2,
                out_channels=features_dim,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(num_features=features_dim),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=features_dim,
                out_channels=features_dim,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(num_features=features_dim),
            nn.ReLU(),
        )

        self.final = nn.Conv2d(
                in_channels=features_dim,
                out_channels=out_dim,
                kernel_size=1
            )



    def forward(self, x):

        enc1 = self.e_1(x)
        #print(f"Encoder block 1 : input ({x.shape}) ---> output ({enc1.shape})")
        enc2 = self.e_2(self.p_1(enc1))
        #print(f"Encoder block 2 : input ({self.p_1(enc1).shape}) ---> output ({enc2.shape})")
        enc3 = self.e_3(self.p_2(enc2))
        #print(f"Encoder block 3 : input ({self.p_2(enc2).shape}) ---> output ({enc3.shape})")
        enc4 = self.e_4(self.p_3(enc3))
        #print(f"Encoder block 4 : input ({self.p_3(enc3).shape}) ---> output ({enc4.shape})")

        bottle = self.b(self.p_4(enc4))
        #print(f"BottleNeck : input ({self.p_4(enc4).shape}) ---> output ({bottle.shape})")

        dec1 = self.d_upconv1(bottle)
        dec1 = torch.cat((dec1, enc4), dim=1)
        dec1 = self.d_1(dec1)
        #print(f"Decoder block 1 : input ({bottle.shape}) ---> output ({dec1.shape})")
        dec2 = self.d_upconv2(dec1)
        dec2 = torch.cat((dec2, enc3), dim=1)
        dec2 = self.d_2(dec2)
        #print(f"Decoder block 2 : input ({dec1.shape}) ---> output ({dec2.shape})")
        dec3 = self.d_upconv3(dec2)
        dec3 = torch.cat((dec3, enc2), dim=1)
        dec3 = self.d_3(dec3)
        #print(f"Decoder block 3 : input ({dec2.shape}) ---> output ({dec3.shape})")
        dec4 = self.d_upconv4(dec3)
        dec4 = torch.cat((dec4, enc1), dim=1)
        dec4 = self.d_4(dec4)
        #print(f"Decoder block 4 : input ({dec3.shape}) ---> output ({dec4.shape})")
        out = self.final(dec4)
        #print(f"Final : input ({dec4.shape}) ---> output ({out.shape})")


        # print(f"UNet : input ({x.shape}) ---> output ({out.shape}) \n")
        return out