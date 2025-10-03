import os, time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataset import OASISSegDataset
from modules import UNet, combined_loss, calculate_dice

def train_unet(model, train_loader, val_loader, epochs=50, lr=1e-3, device="cuda", out_dir="./results"):
    os.makedirs(out_dir, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler('cuda')

    train_losses, val_losses, val_dices = [], [], []
    for epoch in range(epochs):
        start_time = time.time()
        model.train(); running_loss = 0.0

        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                preds = model(imgs)
                loss = combined_loss(preds, masks)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        scheduler.step()

        # Validation
        model.eval(); val_loss, val_dice = 0.0, 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True)
                with torch.amp.autocast('cuda'):
                    preds = model(imgs)
                    loss = combined_loss(preds, masks)
                val_loss += loss.item()
                val_dice += calculate_dice(preds, masks)

        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)

        print(f"Epoch {epoch+1}/{epochs} | Train {avg_train_loss:.4f} | Val {avg_val_loss:.4f} | Dice {avg_val_dice:.4f} | Time {time.time()-start_time:.2f}s")

        train_losses.append(avg_train_loss); val_losses.append(avg_val_loss); val_dices.append(avg_val_dice)

        if avg_val_dice > 0.9:
            print(f"🎯 Target DSC achieved: {avg_val_dice:.4f}")
        if (epoch + 1) % 10 == 0 or avg_val_dice > 0.85:
            torch.save(model.state_dict(), os.path.join(out_dir, f"unet_epoch_{epoch+1}.pth"))

    return train_losses, val_losses, val_dices

if __name__ == "__main__":
    root = "/home/groups/comp3710/OASIS"
    train_ds = OASISSegDataset(root+"keras_png_slices_train", root+"keras_png_slices_seg_train")
    val_ds = OASISSegDataset(root+"keras_png_slices_validate", root+"keras_png_slices_seg_validate")

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet(in_ch=1, out_ch=3).to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    train_unet(model, train_loader, val_loader, epochs=10, lr=1e-4, device=device)

