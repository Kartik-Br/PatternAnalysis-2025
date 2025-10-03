import os
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms

class OASISSegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_size=128):
        self.img_paths = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".png")])
        self.mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith(".png")])
        self.img_size = img_size
        self.num_classes = 3  # Background, CSF, GM (WM/255 → background)

        self.img_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
        print(f"Loaded {len(self.img_paths)} images and {len(self.mask_paths)} masks from {img_dir}")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("L")
        mask = Image.open(self.mask_paths[idx]).convert("L")

        img = self.img_transform(img)

        # Resize and convert mask to tensor
        mask = transforms.functional.resize(mask, (self.img_size, self.img_size), interpolation=Image.NEAREST)
        mask = np.array(mask, dtype=np.uint8)
        mask_tensor = torch.tensor(mask, dtype=torch.long)

        # Map OASIS mask values
        segmap = torch.zeros_like(mask_tensor)
        segmap[mask_tensor == 0] = 0    # Background
        segmap[mask_tensor == 85] = 1   # CSF
        segmap[mask_tensor == 170] = 2  # GM
        segmap[mask_tensor == 255] = 0  # Ignore → background

        # One-hot encode
        seg_onehot = F.one_hot(segmap, num_classes=self.num_classes).permute(2, 0, 1).float()

        return img, seg_onehot



