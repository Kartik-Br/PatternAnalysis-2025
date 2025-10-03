import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ConvBlock with GroupNorm
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, p=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
    def forward(self, x):
        return self.block(x)

# UNet with 3-class output
class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=3):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.dec4 = ConvBlock(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec3 = ConvBlock(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec2 = ConvBlock(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec1 = ConvBlock(128, 64)

        self.final = nn.Sequential(
            nn.Conv2d(64, out_ch, kernel_size=1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.final(d1)

# Loss functions
def dice_loss(pred, target, smooth=1.0):
    pred_flat = pred.permute(0, 2, 3, 1).contiguous().view(-1, pred.size(1))
    target_flat = target.permute(0, 2, 3, 1).contiguous().view(-1, target.size(1))
    intersection = (pred_flat * target_flat).sum(dim=0)
    union = pred_flat.sum(dim=0) + target_flat.sum(dim=0)
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

def combined_loss(pred, target, alpha=0.5):
    dice = dice_loss(pred, target)
    ce = F.cross_entropy(pred, target.argmax(dim=1))
    return alpha * dice + (1 - alpha) * ce

# Metric
def calculate_dice(pred, target):
    pred_bin = pred.argmax(dim=1)
    target_bin = target.argmax(dim=1)
    dice_scores = []
    for class_idx in range(pred.size(1)):
        pred_mask = (pred_bin == class_idx).float()
        target_mask = (target_bin == class_idx).float()
        intersection = (pred_mask * target_mask).sum()
        union = pred_mask.sum() + target_mask.sum()
        dice_scores.append((2. * intersection / union).item() if union > 0 else 1.0)
    return np.mean(dice_scores)

