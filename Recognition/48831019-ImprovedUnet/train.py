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
    # pred: B x C x D x H x W logits or probs; target: B x D x H x W ints
    if pred.dim() == 5:
        probs = torch.softmax(pred, dim=1)
        # convert target to one-hot
        num_classes = pred.shape[1]
        target_oh = torch.nn.functional.one_hot(target, num_classes).permute(0,4,1,2,3).float()
    else:
        raise ValueError("Expected logits of shape BxCxDHW")
    dices = []
    start = 1 if ignore_background else 0
    for c in range(start, num_classes):
        p = probs[:, c].contiguous().view(probs.size(0), -1)
        t = target_oh[:, c].contiguous().view(target_oh.size(0), -1)
        inter = (p * t).sum(1)
        denom = p.sum(1) + t.sum(1)
        dice = (2. * inter + smooth) / (denom + smooth)
        dices.append(dice.mean().item())
    return dices  # list per-class dice

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, weight=None, ignore_background=False):
        super().__init__()
        self.smooth = smooth
        self.weight = weight
        self.ignore_background = ignore_background

    def forward(self, logits, target):
        probs = torch.softmax(logits, dim=1)
        num_classes = logits.shape[1]
        target_oh = torch.nn.functional.one_hot(target, num_classes).permute(0,4,1,2,3).float()
        start = 1 if self.ignore_background else 0
        loss = 0.0
        for c in range(start, num_classes):
            p = probs[:, c].contiguous().view(probs.size(0), -1)
            t = target_oh[:, c].contiguous().view(target_oh.size(0), -1)
            inter = (p * t).sum(1)
            denom = p.sum(1) + t.sum(1)
            loss_c = 1 - ((2. * inter + self.smooth) / (denom + self.smooth))
            loss += loss_c.mean()
        if self.weight is not None:
            loss = loss * self.weight
        return loss

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
    criterion = DiceLoss(ignore_background=False)

    best_val_dice = 0.0
    history = {'train_loss': [], 'val_loss': [], 'val_mean_dice': []}

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            if torch.any(yb >= args.num_classes):
                bad = yb[yb >= args.num_classes]
                print("🔥 Found invalid label(s):", bad.unique(), "max label:", yb.max().item())
                raise ValueError("Label index out of range for num_classes")
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= len(train_loader.dataset)
        history['train_loss'].append(epoch_loss)

        # validation
        model.eval()
        val_loss = 0.0
        dice_list = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * xb.size(0)
                dices = dice_coef(logits, yb)
                dice_list.append(np.mean(dices))
        val_loss /= len(val_loader.dataset)
        mean_dice = np.mean(dice_list) if len(dice_list) > 0 else 0.0
        history['val_loss'].append(val_loss)
        history['val_mean_dice'].append(mean_dice)

        print(f"Epoch {epoch}/{args.epochs} train_loss={epoch_loss:.4f} val_loss={val_loss:.4f} val_mean_dice={mean_dice:.4f} time={(time.time()-t0):.1f}s")

        # checkpoint best
        if mean_dice > best_val_dice:
            best_val_dice = mean_dice
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'best_val_dice': best_val_dice,
                'history': history
            }, args.checkpoint)
            print(f"Saved best model (dice={best_val_dice:.4f}) -> {args.checkpoint}")

        # simple scheduler
        if epoch % args.save_every == 0:
            # also save a regular checkpoint
            ck = args.checkpoint.replace('.pt', f'.epoch{epoch}.pt')
            torch.save({'epoch': epoch, 'model_state': model.state_dict()}, ck)

    # write training plots
    plt.figure()
    plt.plot(history['train_loss'], label='train_loss')
    plt.plot(history['val_loss'], label='val_loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig('loss_plot.png')

    plt.figure()
    plt.plot(history['val_mean_dice'], label='val_mean_dice')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig('dice_plot.png')
    print("Training finished. Best val dice:", best_val_dice)

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

