import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np
import argparse

from modules import Improved3DUNet as Model
from dataset import ProstateDataset

# --- Configuration ---
CLASS_NAMES = ["0", "1", "2", "3", "4", "5"]


# --- NEW Combined Loss Function ---
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
        for i in range(1, self.num_classes):  # Ignore background for dice
            probas_i = probas[:, i, :, :, :]
            targets_i = targets_one_hot[:, i, :, :, :]
            intersection = torch.sum(probas_i * targets_i)
            union = torch.sum(probas_i) + torch.sum(targets_i)
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_loss += 1 - dice

        dice_loss = dice_loss / (self.num_classes - 1)

        # --- Cross-Entropy Loss Calculation ---
        # Reshape for CrossEntropyLoss
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


# --- Validation and Training Loop (No significant changes) ---
def calculate_dice_scores(logits, targets, num_classes, smooth=1e-6):
    probas = F.softmax(logits, dim=1)
    preds = torch.argmax(probas, dim=1)
    dice_scores = []
    for i in range(num_classes):
        pred_i = (preds == i).float()
        target_i = (targets.squeeze(1) == i).float()
        intersection = torch.sum(pred_i * target_i)
        union = torch.sum(pred_i) + torch.sum(target_i)
        if union == 0:
            dice = torch.tensor(1.0, device=preds.device)
        else:
            dice = (2.0 * intersection + smooth) / (union + smooth)
        dice_scores.append(dice.item())
    return dice_scores


def train_model(model, train_loader, val_loader, optimizer, loss_fn, device, args):
    best_val_loss = float("inf")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        progress_bar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Training]", leave=False
        )
        for batch in progress_bar:
            images, masks = batch["image"].to(device), batch["mask"].to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        total_dice_scores = np.zeros(args.num_classes)
        val_progress_bar = tqdm(
            val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Validation]", leave=False
        )
        with torch.no_grad():
            for batch in val_progress_bar:
                images, masks = batch["image"].to(device), batch["mask"].to(device)
                outputs = model(images)
                loss = loss_fn(outputs, masks)
                val_loss += loss.item()
                dice_scores = calculate_dice_scores(outputs[0], masks, args.num_classes)
                total_dice_scores += np.array(dice_scores)
                val_progress_bar.set_postfix(val_loss=f"{loss.item():.4f}")

        avg_val_loss = val_loss / len(val_loader)
        avg_dice_scores = total_dice_scores / len(val_loader)

        print(f"\n--- Epoch {epoch+1}/{args.epochs} Summary ---")
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        print("Validation DSC Scores (Primary Output):")
        for i, class_name in enumerate(CLASS_NAMES):
            print(f"- {class_name}: {avg_dice_scores[i]:.4f}")
        print("-------------------------\n")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), args.model_save_path)
            print(f"Model improved and saved to {args.model_save_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Deep Supervision 3D U-Net.")
    # ... (parser arguments are the same) ...
    parser.add_argument(
        "--img_dir", type=str, required=True, help="Directory for training images."
    )
    parser.add_argument(
        "--lbl_dir", type=str, required=True, help="Directory for training labels."
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs."
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument(
        "--val_split", type=float, default=0.2, help="Validation split proportion."
    )
    parser.add_argument(
        "--model_save_path",
        type=str,
        default="deep_supervision_unet.pth",
        help="Path to save the best model.",
    )
    parser.add_argument(
        "--num_classes", type=int, default=6, help="Number of segmentation classes."
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    full_dataset = ProstateDataset(image_dir=args.img_dir, label_dir=args.lbl_dir)
    val_size = int(args.val_split * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

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

    model = Model(in_channels=1, out_channels=args.num_classes).to(device)

    dummy_input = torch.randn(1, 1, 64, 128, 128).to(device)
    num_outputs = len(model(dummy_input))
    print(f"Model has {num_outputs} outputs for deep supervision.")

    # --- DEFINE WEIGHTS FOR CROSS-ENTROPY ---
    # Give small classes a much higher weight.
    # These are example weights, you may need to tune them.
    # Order: Background, Body, Bone, Bladder, Rectum, Prostate
    ce_weights = torch.tensor([0.1, 0.5, 1.0, 2.0, 5.0, 10.0]).to(device)
    print(f"Using Cross-Entropy weights: {ce_weights.cpu().numpy()}")

    base_loss = DiceCELoss(num_classes=args.num_classes, ce_weights=ce_weights).to(
        device
    )
    loss_fn = DeepSupervisionLoss(base_loss=base_loss, num_outputs=num_outputs)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_model(model, train_loader, val_loader, optimizer, loss_fn, device, args)

    print("Training finished.")
