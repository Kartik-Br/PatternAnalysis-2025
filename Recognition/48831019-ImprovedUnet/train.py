import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np
import argparse
import os

from dataset import ProstateDataset
from modules import Improved3DUNet

# --- Class Definitions (No Changes) ---
CLASS_NAMES = ["Background", "Body", "Bone", "Bladder", "Rectum", "Prostate"]


class MultiClassDiceLoss(nn.Module):
    def __init__(self, num_classes, smooth=1e-6):
        super(MultiClassDiceLoss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits, targets):
        probas = F.softmax(logits, dim=1)
        targets_one_hot = (
            F.one_hot(targets.squeeze(1), num_classes=self.num_classes)
            .permute(0, 4, 1, 2, 3)
            .float()
        )
        total_dice_loss = 0
        for i in range(1, self.num_classes):
            probas_i = probas[:, i, :, :, :]
            targets_i = targets_one_hot[:, i, :, :, :]
            intersection = torch.sum(probas_i * targets_i)
            union = torch.sum(probas_i) + torch.sum(targets_i)
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            total_dice_loss += 1 - dice
        return total_dice_loss / (self.num_classes - 1)


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


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_fn,
    device,
    epochs,
    model_save_path,
    num_classes,
):
    best_val_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        progress_bar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{epochs} [Training]", leave=False
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
        total_dice_scores = np.zeros(num_classes)
        val_progress_bar = tqdm(
            val_loader, desc=f"Epoch {epoch+1}/{epochs} [Validation]", leave=False
        )
        with torch.no_grad():
            for batch in val_progress_bar:
                images, masks = batch["image"].to(device), batch["mask"].to(device)
                outputs = model(images)
                loss = loss_fn(outputs, masks)
                val_loss += loss.item()
                dice_scores = calculate_dice_scores(outputs, masks, num_classes)
                total_dice_scores += np.array(dice_scores)
                val_progress_bar.set_postfix(val_loss=f"{loss.item():.4f}")

        avg_val_loss = val_loss / len(val_loader)
        avg_dice_scores = total_dice_scores / len(val_loader)

        print(f"\n--- Epoch {epoch+1}/{epochs} Summary ---")
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        print("Validation DSC Scores:")
        for i, class_name in enumerate(CLASS_NAMES):
            print(f"- {class_name}: {avg_dice_scores[i]:.4f}")
        print("-------------------------\n")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"Model improved and saved to {model_save_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train an Improved 3D U-Net for multiclass segmentation."
    )
    parser.add_argument(
        "--img_dir",
        type=str,
        required=True,
        help="Directory containing the training images.",
    )
    parser.add_argument(
        "--lbl_dir",
        type=str,
        required=True,
        help="Directory containing the training labels.",
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for training. Default is 1.",
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.2,
        help="Proportion of the dataset to use for validation.",
    )
    parser.add_argument(
        "--model_save_path",
        type=str,
        default="prostate_segmentation_model_multiclass.pth",
        help="Path to save the best model.",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=6,
        help="Number of classes for segmentation (including background).",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Image Directory: {args.img_dir}")
    print(f"Label Directory: {args.lbl_dir}")

    full_dataset = ProstateDataset(image_dir=args.img_dir, label_dir=args.lbl_dir)

    val_size = int(args.val_split * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    model = Improved3DUNet(in_channels=1, out_channels=args.num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = MultiClassDiceLoss(num_classes=args.num_classes).to(device)

    train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        loss_fn,
        device,
        args.epochs,
        args.model_save_path,
        args.num_classes,
    )

    print("Training finished.")
