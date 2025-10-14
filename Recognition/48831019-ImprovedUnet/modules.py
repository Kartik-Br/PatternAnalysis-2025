# modules.py
# 3D Improved UNet-ish architecture (PyTorch)
# Reasonable, modular, intended for 3D prostate segmentation (multi-label)
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_bn=True):
        super().__init__()
        layers = [
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=not use_bn),
            nn.InstanceNorm3d(out_ch) if use_bn else nn.Identity(),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=not use_bn),
            nn.InstanceNorm3d(out_ch) if use_bn else nn.Identity(),
            nn.LeakyReLU(0.01, inplace=True),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool3d(2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, tr_mode='trilinear'):
        super().__init__()
        # in_ch = channels from skip + features (so typically 2*ch)
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # pad if needed
        if x.shape != skip.shape:
            # simple center crop / pad
            diffZ = skip.size(2) - x.size(2)
            diffY = skip.size(3) - x.size(3)
            diffX = skip.size(4) - x.size(4)
            x = F.pad(x, [diffX//2, diffX - diffX//2,
                          diffY//2, diffY - diffY//2,
                          diffZ//2, diffZ - diffZ//2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class OutputConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class ImprovedUNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=4, base_filters=16):
        """
        out_channels: number of segmentation labels (including background)
        base_filters: starting filter count. Increase if GPU permits.
        """
        super().__init__()
        f = base_filters
        self.inc = ConvBlock(in_channels, f)
        self.down1 = Down(f, f*2)
        self.down2 = Down(f*2, f*4)
        self.down3 = Down(f*4, f*8)

        # bottleneck with dropout
        self.bottleneck = ConvBlock(f*8, f*16)
        self.drop = nn.Dropout3d(0.3)

        self.up3 = Up(f*16 + f*8, f*8)
        self.up2 = Up(f*8 + f*4, f*4)
        self.up1 = Up(f*4 + f*2, f*2)
        self.up0 = Up(f*2 + f, f)

        self.outc = OutputConv(f, out_channels)

    def forward(self, x):
        x1 = self.inc(x)   # f
        x2 = self.down1(x1) # f*2
        x3 = self.down2(x2) # f*4
        x4 = self.down3(x3) # f*8
        xb = self.bottleneck(x4)
        xb = self.drop(xb)

        xu = self.up3(xb, x4)
        xu = self.up2(xu, x3)
        xu = self.up1(xu, x2)
        xu = self.up0(xu, x1)
        out = self.outc(xu)
        # logits returned
        return out

