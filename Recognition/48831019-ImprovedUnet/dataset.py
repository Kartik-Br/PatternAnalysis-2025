# dataset.py
import os
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
import torch
import random

# optional: use pyimgaug3d for augmentation if installed
try:
    import pyimgaug3d as pia3d
    HAS_PIA = True
except Exception:
    HAS_PIA = False


def load_nifti_as_array(path):
    img = nib.load(path)
    data = img.get_fdata()
    # remove singleton 4th dim if present
    if data.ndim == 4:
        data = data[..., 0]
    return data.astype(np.float32), img.affine, img.header


class Prostate3DDataset(Dataset):
    """
    Expects a list of (image_path, label_path) pairs.

    Returns patches (C=1, D, H, W) and labels (D,H,W) as ints
    """

    def __init__(self, pairs, patch_size=(64,128,128), normalize=True, augment=True):
        self.pairs = pairs
        self.patch_size = patch_size
        self.normalize = normalize
        self.augment = augment and HAS_PIA
        if self.augment:
            # example transform chain - adjust as desired
            self.aug = pia3d.Sequential([
                pia3d.RandomFlip(axis=(0,1,2), p=0.25),
                pia3d.RandomRotate3D(angle_range=(-10,10), p=0.3),
                pia3d.RandomZoom3D(zoom_range=(0.9,1.1), p=0.3),
                pia3d.RandomElasticDeformation3D(alpha=5, sigma=4, p=0.2),
                # intensity transforms:
                pia3d.RandomGamma(p=0.2),
            ])
        else:
            self.aug = None

    def __len__(self):
        return len(self.pairs)

    def random_patch(self, vol, label):
        d, h, w = vol.shape
        pd, ph, pw = self.patch_size
        if d <= pd:
            sd = 0
        else:
            sd = random.randint(0, d - pd)
        if h <= ph:
            sh = 0
        else:
            sh = random.randint(0, h - ph)
        if w <= pw:
            sw = 0
        else:
            sw = random.randint(0, w - pw)
        patch = vol[sd:sd+pd, sh:sh+ph, sw:sw+pw]
        lpatch = label[sd:sd+pd, sh:sh+ph, sw:sw+pw]
        return patch, lpatch

    def center_patch(self, vol, label):
        d, h, w = vol.shape
        pd, ph, pw = self.patch_size
        sd = max(0, (d - pd)//2)
        sh = max(0, (h - ph)//2)
        sw = max(0, (w - pw)//2)
        patch = vol[sd:sd+pd, sh:sh+ph, sw:sw+pw]
        lpatch = label[sd:sd+pd, sh:sh+ph, sw:sw+pw]
        return patch, lpatch

    def __getitem__(self, idx):
        img_path, lbl_path = self.pairs[idx]
        vol, _aff, _ = load_nifti_as_array(img_path)
        label, _, _ = load_nifti_as_array(lbl_path)

        if random.random() < 0.8:
            x, y = self.random_patch(vol, label)
        else:
            x, y = self.center_patch(vol, label)

        # normalize intensity (z-score)
        if self.normalize:
            x = (x - x.mean()) / (x.std() + 1e-8)

        # add channel dim once
        x = np.expand_dims(x, 0).astype(np.float32)  # (1, D, H, W)
        y = np.rint(y).astype(np.uint8)

        if self.augment and self.aug is not None:
            d = {'image': x, 'mask': y}
            out = self.aug(d)
            x = out['image']
            y = out['mask']

        # one-hot encode AFTER augmentation
        num_classes = 6
        onehot = np.eye(num_classes)[y]          # (D,H,W,C)
        onehot = np.transpose(onehot, (3,0,1,2)) # (C,D,H,W)

        # --- to torch ---
        x_t = torch.from_numpy(x).float()        # (1, D, H, W)
        y_t = torch.from_numpy(onehot).float()   # (C, D, H, W)

        return x_t, y_t


