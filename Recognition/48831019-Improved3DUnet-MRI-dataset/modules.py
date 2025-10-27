import torch
import torch.nn as nn
import torch.nn.functional as F


class PreActResBlock(nn.Module):
    """
    A pre-activation residual block with two 3D convolutions.
    Includes instance normalization, ReLU activation, and dropout.
    """

    def __init__(self, in_channels, out_channels, dropout_prob=0.2):
        super(PreActResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # extract features and channel expansion
        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.norm1 = nn.InstanceNorm3d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)

        # convolution to refine features
        self.conv2 = nn.Conv3d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.norm2 = nn.InstanceNorm3d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        # dropout prob for regularization
        self.dropout = nn.Dropout3d(p=dropout_prob)

        # Shortcut connection to match dimensions if in_channels != out_channels
        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(
                in_channels, out_channels, kernel_size=1, bias=False
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.relu1(self.norm1(x))
        out = self.conv1(out)
        out = self.dropout(out)

        out = self.relu2(self.norm2(out))
        out = self.conv2(out)

        # adding residual connections from previous layers
        return out + residual


class Encoder(nn.Module):
    """Encoder block using PreActResBlock and max pooling."""

    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.block = PreActResBlock(in_channels, out_channels)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        skip = self.block(x)
        down = self.pool(skip)
        return down, skip


class Decoder(nn.Module):
    """Decoder block with upsampling and PreActResBlock."""

    def __init__(self, in_channels, skip_channels, out_channels):
        super(Decoder, self).__init__()
        # Upsamples the tensor from the previous decoder layer
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        # Correctly defines the block to handle the concatenated tensor
        self.block = PreActResBlock(skip_channels + out_channels, out_channels)

    def forward(self, x, skip_connection):
        up = self.up(x)
        # Pad upsampled tensor to match skip connection's spatial dimensions
        diffZ = skip_connection.size()[2] - up.size()[2]
        diffY = skip_connection.size()[3] - up.size()[3]
        diffX = skip_connection.size()[4] - up.size()[4]
        up = F.pad(
            up,
            [
                diffX // 2,
                diffX - diffX // 2,
                diffY // 2,
                diffY - diffY // 2,
                diffZ // 2,
                diffZ - diffZ // 2,
            ],
        )

        x = torch.cat([skip_connection, up], dim=1)
        return self.block(x)


class Improved3DUNet(nn.Module):
    """
    A 3D U-Net with pre-activation residual blocks, dropout, and deep supervision.
    """

    def __init__(self, in_channels=1, out_channels=6, features=[32, 64, 128, 256]):
        super(Improved3DUNet, self).__init__()

        self.encoders = nn.ModuleList()
        # Initial block to get to the first feature dimension
        self.initial_block = PreActResBlock(in_channels, features[0])

        # Encoder path
        in_ch = features[0]
        for feature in features[1:]:
            self.encoders.append(Encoder(in_ch, feature))
            in_ch = feature

        self.bottleneck = PreActResBlock(features[-1], features[-1] * 2)

        self.decoders = nn.ModuleList()
        self.deep_supervision_layers = nn.ModuleList()

        # Decoder path
        reversed_features = features[::-1]  # [256, 128, 64, 32]
        in_ch = features[-1] * 2  # From bottleneck

        for i in range(len(reversed_features) - 1):
            skip_ch = reversed_features[i]
            out_ch = reversed_features[i + 1]
            self.decoders.append(Decoder(in_ch, skip_ch, out_ch))
            self.deep_supervision_layers.append(
                nn.Conv3d(out_ch, out_channels, kernel_size=1)
            )
            in_ch = out_ch

        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        x = self.initial_block(x)
        skip_connections.append(x)

        # Encoder path
        for i, encoder in enumerate(self.encoders):
            x, skip = encoder(x)
            skip_connections.append(skip)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path with deep supervision
        deep_outputs = []
        skip_connections = skip_connections[::-1]

        for i, decoder in enumerate(self.decoders):
            skip = skip_connections[i]
            x = decoder(x, skip)
            # Generate deep supervision output
            deep_outputs.append(self.deep_supervision_layers[i](x))

        # Final output layer
        final_output = self.final_conv(x)

        # Upsample all deep supervision outputs to the size of the final output
        outputs = [final_output]
        for deep_out in deep_outputs:
            outputs.append(
                F.interpolate(
                    deep_out,
                    size=final_output.shape[2:],
                    mode="trilinear",
                    align_corners=False,
                )
            )

        # Return a list of outputs, with the highest-resolution one first
        return outputs
