# train.py
import os
import argparse
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from modules import ImprovedUNet3D
from dataset import Prostate3DDataset

# Dice / loss helpers
def dice_coef(pred, target, smooth=1e-6, ignore_background=False):
    """
    pred: B x C x D x H x W (logits)
    target: B x C x D x H x W (one-hot)
    """
    if pred.dim() != 5 or target.dim() != 5:
        raise ValueError("Expected tensors of shape [B, C, D, H, W]")

    # Convert logits → probabilities
    probs = torch.softmax(pred, dim=1)

    num_classes = pred.shape[1]
    dices = []

    # Optionally ignore background
    start = 1 if ignore_background else 0

    for c in range(start, num_classes):
        p = probs[:, c].contiguous().view(probs.size(0), -1)
        t = target[:, c].contiguous().view(target.size(0), -1)

        inter = (p * t).sum(1)
        denom = p.sum(1) + t.sum(1)
        dice = (2. * inter + smooth) / (denom + smooth)
        dices.append(dice.mean().item())

    return dices  # list of per-class Dice scores

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target_oh):
        probs = torch.softmax(logits, dim=1)
        inter = (probs * target_oh).sum(dim=(2,3,4))
        denom = probs.sum(dim=(2,3,4)) + target_oh.sum(dim=(2,3,4))
        dice = (2 * inter + self.smooth) / (denom + self.smooth)
        return 1 - dice.mean()

def make_pairs_from_directory(img_dir, lbl_dir):
    imgs = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.nii') or f.endswith('.nii.gz')])
    lbls = sorted([os.path.join(lbl_dir, f) for f in os.listdir(lbl_dir) if f.endswith('.nii') or f.endswith('.nii.gz')])
    # naive matching by filename; adjust logic if names differ
    pairs = []
    for im in imgs:
        base = os.path.basename(im)
        # attempt to find matching label file
        for l in lbls:
            if os.path.basename(l).startswith(os.path.splitext(base)[0]):
                pairs.append((im, l))
                break
    # fallback: pair by index
    if len(pairs) == 0:
        pairs = list(zip(imgs, lbls))
    return pairs

def train_loop(args):
    os.makedirs(os.path.dirname(args.checkpoint), exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pairs = make_pairs_from_directory(args.img_dir, args.lbl_dir)
    # shuffle and split
    random_seed = 42
    np.random.seed(random_seed)
    np.random.shuffle(pairs)
    n_total = len(pairs)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)
    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:n_train+n_val]
    test_pairs = pairs[n_train+n_val:]

    train_ds = Prostate3DDataset(train_pairs, patch_size=args.patch, augment=True)
    val_ds = Prostate3DDataset(val_pairs, patch_size=args.patch, augment=False)
    test_ds = Prostate3DDataset(test_pairs, patch_size=args.patch, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=1, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=1)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=1)

    model = ImprovedUNet3D(in_channels=1, out_channels=args.num_classes, base_filters=args.base_filters).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = DiceLoss()
    model = ImprovedUNet3D(
        in_channels=1,
        out_channels=args.num_classes,
        base_filters=args.base_filters
    ).to(device)


    train_losses = []
    val_losses = []
    val_dice_scores = []  # list of lists (per-class)

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        train_losses.append(epoch_loss / len(train_loader))

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        dices_per_class = []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item()

                # per-class DSC
                dices = dice_coef(logits, yb)  # list of floats
                dices_per_class.append(dices)

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # average DSC per class across validation set
        dices_per_class = np.array(dices_per_class)
        mean_dices = dices_per_class.mean(axis=0).tolist()
        val_dice_scores.append(mean_dices)

        print(f"[Epoch {epoch+1}/{args.epochs}] "
              f"Train Loss: {train_losses[-1]:.4f}, "
              f"Val Loss: {val_loss:.4f}")
        for i, dsc in enumerate(mean_dices):
            print(f"  Class {i}: DSC={dsc:.4f}")

        # Optional: checkpoint
        torch.save(model.state_dict(), f"checkpoints/epoch_{epoch+1:03d}.pt")

    # --- Plot curves after training ---
    plot_training_curves(train_losses, val_losses, val_dice_scores)

def plot_training_curves(train_losses, val_losses, val_dice_scores):
    epochs = np.arange(1, len(train_losses) + 1)

    # Plot training vs validation loss
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train Loss (Dice)")
    plt.plot(epochs, val_losses, label="Val Loss (Dice)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Dice Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results_prostate3d/loss_curve.png", dpi=150)
    plt.close()

    # Plot per-class DSC
    val_dice_scores = np.array(val_dice_scores)
    num_classes = val_dice_scores.shape[1]
    plt.figure(figsize=(8, 5))
    for c in range(num_classes):
        plt.plot(epochs, val_dice_scores[:, c], label=f"Class {c}")
    plt.xlabel("Epoch")
    plt.ylabel("Mean DSC")
    plt.title("Per-Class Validation Dice Coefficient")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results_prostate3d/dsc_per_class.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, required=True, help='path to nifti images folder')
    parser.add_argument('--lbl_dir', type=str, required=True, help='path to label nifti folder')
    parser.add_argument('--checkpoint', type=str, default='best_model.pt')
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patch', nargs=3, type=int, default=(64,128,128))
    parser.add_argument('--num_classes', type=int, default=5)  # adjust to your dataset labels count
    parser.add_argument('--base_filters', type=int, default=16)
    parser.add_argument('--save_every', type=int, default=10)
    args = parser.parse_args()
    train_loop(args)

