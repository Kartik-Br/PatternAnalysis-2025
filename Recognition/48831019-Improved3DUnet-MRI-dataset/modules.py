import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """
    Standard 3D convolutional block with two convolutions,
    each followed by instance normalization and ReLU activation.
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_block(x)

class Encoder(nn.Module):
    """
    Encoder part of the U-Net.
    Downsamples the input using max pooling.
    """
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.conv_block = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        skip = self.conv_block(x)
        down = self.pool(skip)
        return down, skip

class Decoder(nn.Module):
    """
    Decoder part of the U-Net.
    Upsamples the feature map and concatenates with the skip connection.
    """
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_block = ConvBlock(in_channels, out_channels) # in_channels = skip_channels + upsampled_channels

    def forward(self, x, skip_connection):
        up = self.up(x)
        # Ensure spatial dimensions match for concatenation
        diffZ = skip_connection.size()[2] - up.size()[2]
        diffY = skip_connection.size()[3] - up.size()[3]
        diffX = skip_connection.size()[4] - up.size()[4]

        up = F.pad(up, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])
        
        x = torch.cat([skip_connection, up], dim=1)
        return self.conv_block(x)

class Improved3DUNet(nn.Module):
    """
    The main Improved 3D U-Net model.
    """
    # Updated to handle 6 classes by default
    def __init__(self, in_channels=1, out_channels=6, features=[32, 64, 128, 256]):
        super(Improved3DUNet, self).__init__()
        
        self.encoders = nn.ModuleList()
        for feature in features:
            self.encoders.append(Encoder(in_channels, feature))
            in_channels = feature

        self.bottleneck = ConvBlock(features[-1], features[-1] * 2)

        self.decoders = nn.ModuleList()
        for feature in reversed(features):
            self.decoders.append(Decoder(feature * 2, feature))

        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        
        # Encoder path
        for encoder in self.encoders:
            x, skip = encoder(x)
            skip_connections.append(skip)
            
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder path
        skip_connections = skip_connections[::-1] # Reverse for decoding
        for i in range(len(self.decoders)):
            x = self.decoders[i](x, skip_connections[i])
            
        # Final convolution
        return self.final_conv(x)