import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np
import argparse

from modules import Improved3DUNet as Model
from dataset import ProstateDataset

CLASS_NAMES = ["0", "1", "2", "3", "4", "5"]


class DiceCELoss(nn.Module):
    """
    A combined Dice and Cross-Entropy loss function.
    This is very effective for imbalanced segmentation tasks.
    """

    def __init__(self, num_classes, ce_weights=None, smooth=1e-6):
        super(DiceCELoss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ce = nn.CrossEntropyLoss(weight=ce_weights)

    def forward(self, logits, targets):
        # --- Dice Loss Calculation ---
        probas = F.softmax(logits, dim=1)
        targets_one_hot = (
            F.one_hot(targets.squeeze(1), num_classes=self.num_classes)
            .permute(0, 4, 1, 2, 3)
            .float()
        )

        dice_loss = 0
        for i in range(1, self.num_classes):
            # Ignore background for dice (always around 1)
            probas_i = probas[:, i, :, :, :]
            targets_i = targets_one_hot[:, i, :, :, :]
            intersection = torch.sum(probas_i * targets_i)
            union = torch.sum(probas_i) + torch.sum(targets_i)
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_loss += 1 - dice

        dice_loss = dice_loss / (self.num_classes - 1)

        # --- Cross-Entropy Loss Calculation ---
        ce_targets = targets.squeeze(1)  # Shape: [B, D, H, W]
        ce_loss = self.ce(logits, ce_targets)

        # Combine the two losses (often weighted equally)
        return dice_loss + ce_loss


class DeepSupervisionLoss(nn.Module):
    """Calculates a weighted sum of losses for deep supervision outputs."""

    def __init__(self, base_loss, num_outputs, weights=None):
        super(DeepSupervisionLoss, self).__init__()
        self.base_loss = base_loss
        self.num_outputs = num_outputs
        if weights is None:
            self.weights = [1.0 / (2**i) for i in range(num_outputs)]
        else:
            self.weights = weights

    def forward(self, predictions, target):
        total_loss = 0
        for i, pred in enumerate(predictions):
            # Downsample target to match prediction size if necessary
            if pred.shape[2:] != target.shape[2:]:
                scaled_target = F.interpolate(
                    target.float(), size=pred.shape[2:], mode="nearest"
                ).long()
            else:
                scaled_target = target

            loss = self.base_loss(pred, scaled_target)
            total_loss += loss * self.weights[i]
        return total_loss


# --- Validation and Training Loop ---
def calculate_dice_scores(logits, targets, num_classes, smooth=1e-6):
    probas = F.softmax(logits, dim=1)
    preds = torch.argmax(probas, dim=1)
    dice_scores = []
    for i in range(num_classes):
        pred_i = (preds == i).float()
        target_i = (targets.squeeze(1) == i).float()
        # Variables in the dice formula
        # A n B
        intersection = torch.sum(pred_i * target_i)
        # A U B
        union = torch.sum(pred_i) + torch.sum(target_i)
        if union == 0:
            dice = torch.tensor(1.0, device=preds.device)
        else:
            dice = (2.0 * intersection + smooth) / (union + smooth)
        dice_scores.append(dice.item())
    return dice_scores


def train_model(model, train_loader, val_loader, optimizer, loss_fn, device, args):
    best_val_loss = float("inf")
    class_scores = [[] for _ in range(args.num_classes)]
    vt_loss = [[], []]
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        # progress bar construction, has train images
        progress_bar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Training]", leave=False
        )
        for batch in progress_bar:
            images, masks = batch["image"].to(device), batch["mask"].to(device)
            optimizer.zero_grad()
            outputs = model(images)

            # loss and backwards propagation
            loss = loss_fn(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            # update progress bar for epoch
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        total_dice_scores = np.zeros(args.num_classes)
        # same loop for validation progress bar
        val_progress_bar = tqdm(
            val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Validation]", leave=False
        )
        # validation loop
        with torch.no_grad():
            for batch in val_progress_bar:
                images, masks = batch["image"].to(device), batch["mask"].to(device)
                outputs = model(images)
                loss = loss_fn(outputs, masks)
                val_loss += loss.item()

                # calculate validation dice scores
                dice_scores = calculate_dice_scores(outputs[0], masks, args.num_classes)
                total_dice_scores += np.array(dice_scores)
                val_progress_bar.set_postfix(val_loss=f"{loss.item():.4f}")

        avg_val_loss = val_loss / len(val_loader)
        avg_dice_scores = total_dice_scores / len(val_loader)

        # end of epoch
        print(f"\n--- Epoch {epoch+1}/{args.epochs} Summary ---")
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        # add the vt losses
        vt_loss[0].append(avg_train_loss)
        vt_loss[1].append(avg_val_loss)
        print("Validation DSC Scores (Primary Output):")
        for i, class_name in enumerate(CLASS_NAMES):
            print(f"- {class_name}: {avg_dice_scores[i]:.4f}")
            # add the DSC scores
            class_scores[i].append(avg_dice_scores[i])
        print("-------------------------\n")

        # save the best model so far
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), args.model_save_path)
            print(f"Model improved and saved to {args.model_save_path}\n")

    np.save("DSCSCORES.npy", class_scores)
    np.save("VTLOSSES.npy", vt_loss)


def three_way_split_indices(n, train_ratio=0.8, val_ratio=0.1, seed=42):
    """
    Return train_idx, val_idx, test_idx that partition range(n) in 80/10/10.
    for use to save JSON file
    """
    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)
    n_test = n - n_train - n_val
    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]
    return train_idx, val_idx, test_idx


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Deep Supervision 3D U-Net.")
    parser.add_argument(
        "--img_dir", type=str, required=True, help="Directory for training images."
    )
    parser.add_argument(
        "--lbl_dir", type=str, required=True, help="Directory for training labels."
    )
    parser.add_argument(
        "--epochs", type=int, default=25, help="Number of training epochs."
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument(
        "--model_save_path",
        type=str,
        default="deep_supervision_unet.pth",
        help="Path to save the best model.",
    )
    parser.add_argument(
        "--num_classes", type=int, default=6, help="Number of segmentation classes."
    )
    parser.add_argument(
        "--split_seed", type=int, default=42, help="Random seed for dataset split."
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Full dataset
    full_dataset = ProstateDataset(image_dir=args.img_dir, label_dir=args.lbl_dir)
    n_items = len(full_dataset)
    if n_items < 10:
        print(f"WARNING: Very small dataset size: {n_items}")

    # 80/10/10 split indices
    train_idx, val_idx, test_idx = three_way_split_indices(
        n_items, train_ratio=0.8, val_ratio=0.1, seed=args.split_seed
    )

    # Create Subsets
    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)
    # not used in training, only use for saving JSON file
    test_dataset = Subset(full_dataset, test_idx)

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )

    # Build model
    model = Model(in_channels=1, out_channels=args.num_classes).to(device)

    # Determine number of outputs for deep supervision
    dummy_input = torch.randn(1, 1, 64, 128, 128).to(device)
    num_outputs = len(model(dummy_input))
    print(f"Model has {num_outputs} outputs for deep supervision.")

    # ce weights to prioritise areas that do not dominate CE (1st is background)
    ce_weights = torch.tensor([0.1, 0.5, 1.0, 2.0, 5.0, 10.0]).to(device)
    base_loss = DiceCELoss(num_classes=args.num_classes, ce_weights=ce_weights).to(
        device
    )
    loss_fn = DeepSupervisionLoss(base_loss=base_loss, num_outputs=num_outputs)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Save split information so predict.py can use the test split
    splits = {
        "img_dir": args.img_dir,
        "lbl_dir": args.lbl_dir,
        "split_seed": args.split_seed,
        "train_indices": train_idx,
        "val_indices": val_idx,
        "test_indices": test_idx,
        # store full paths for test images to make predict.py simple and robust
        "test_image_files": [full_dataset.image_files[i] for i in test_idx],
    }
    with open("splits.json", "w") as f:
        json.dump(splits, f, indent=2)
    print(
        f"Saved dataset splits to splits.json (80/10/10). Test images: {len(splits['test_image_files'])}"
    )

    # Train
    train_model(model, train_loader, val_loader, optimizer, loss_fn, device, args)

    print("Training finished.")
