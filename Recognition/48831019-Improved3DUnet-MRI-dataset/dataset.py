import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
from scipy.ndimage import zoom


class ProstateDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        """
        Args:
            image_dir (string): Directory with all the training images.
            label_dir (string): Directory with all the training labels (masks).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform

        # Assumes that image and label files are sorted and correspond to each other
        self.image_files = sorted(
            [
                os.path.join(image_dir, f)
                for f in os.listdir(image_dir)
                if f.endswith(".nii.gz")
            ]
        )
        self.label_files = sorted(
            [
                os.path.join(label_dir, f)
                for f in os.listdir(label_dir)
                if f.endswith(".nii.gz")
            ]
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        label_path = self.label_files[idx]

        # Load image and label using nibabel
        image_nii = nib.load(image_path)
        label_nii = nib.load(label_path)

        image_array = image_nii.get_fdata()
        label_array = label_nii.get_fdata()

        # Resample to a common size, e.g., 128x128x64
        # Note: nibabel loads images in (H, W, D), changing to (D, H, W) for consistency with PyTorch
        image_array = np.transpose(image_array, (2, 0, 1))
        label_array = np.transpose(label_array, (2, 0, 1))

        target_shape = (64, 128, 128)
        # Resize factors for each dimension if the size is not correct
        zoom_factors = (
            target_shape[0] / image_array.shape[0],
            target_shape[1] / image_array.shape[1],
            target_shape[2] / image_array.shape[2],
        )

        image_array = zoom(
            image_array, zoom_factors, order=1, mode="constant", cval=0.0
        )
        label_array = zoom(
            label_array, zoom_factors, order=0, mode="constant", cval=0.0
        )  # Nearest-neighbor for mask

        # Normalize image
        if np.max(image_array) > np.min(image_array):
            image_array = (image_array - np.min(image_array)) / (
                np.max(image_array) - np.min(image_array)
            )

        # Add channel dimension
        image_array = np.expand_dims(image_array, axis=0)
        label_array = np.expand_dims(label_array, axis=0)

        sample = {
            "image": torch.from_numpy(image_array).float(),
            "mask": torch.from_numpy(label_array).long(),
        }

        if self.transform:
            sample = self.transform(sample)

        return sample
