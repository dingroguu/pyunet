import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import sys
import os

# from .double_conv import DoubleConv
from .ghost_conv import GhostConv
from .depthwise_seperable_conv import DepthwiseSeperableConv


class UNet(nn.Module):
    def __init__(
        self, in_channels=3, out_channels=1
    ):
        super(UNet, self).__init__()

        self.in_channels    = in_channels
        self.out_channels   = out_channels

        self.ups    = nn.ModuleList()
        self.downs  = nn.ModuleList()
        self.pool   = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = nn.Sequential(
            GhostConv(512, 512 * 2),
            DepthwiseSeperableConv(512 * 2, 512 *2)
        )

        self.downs.append(GhostConv(in_channels, 64)) # Ghost Convolution Block for downsampling
        self.downs.append(GhostConv(64, 128))
        self.downs.append(GhostConv(128, 256))
        self.downs.append(GhostConv(256, 512))

        self.ups.append(
            nn.ConvTranspose2d(
                512 * 2,
                512,
                kernel_size=2,
                stride=2
            )
        )

        self.ups.append(GhostConv(512 * 2, 512)) # Ghost Convolution Block for upsampling
        
        self.ups.append(
            nn.ConvTranspose2d(
                256 * 2,
                256,
                kernel_size=2,
                stride=2
            )
        )

        self.ups.append(DepthwiseSeperableConv(256 * 2, 256)) # Depthwise Seperable Convolution Block for upsampling
        
        self.ups.append(
            nn.ConvTranspose2d(
                128 * 2,
                128,
                kernel_size=2,
                stride=2
            )
        )

        self.ups.append(GhostConv(128 * 2, 128)) 
        
        self.ups.append(
            nn.ConvTranspose2d(
                64 * 2,
                64,
                kernel_size=2,
                stride=2
            )
        )

        self.ups.append(DepthwiseSeperableConv(64 * 2, 64)) 

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        # Reverse skip_connections
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            #print(x.shape[-1])
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)