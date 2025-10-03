import torch
import matplotlib.pyplot as plt
from dataset import OASISSegDataset
from modules import UNet

if __name__ == "__main__":
    ckpt = "./results/unet_epoch_10.pth"
    root = "/home/groups/comp3710/OASIS"

    val_ds = OASISSegDataset(
        root + "keras_png_slices_validate", root + "keras_png_slices_seg_validate"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = UNet(in_ch=1, out_ch=3)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device).eval()

    img, mask = val_ds[0]
    with torch.no_grad():
        pred = model(img.unsqueeze(0).to(device)).cpu().squeeze(0)

    plt.subplot(1, 3, 1)
    plt.imshow(img[0], cmap="gray")
    plt.title("Image")
    plt.subplot(1, 3, 2)
    plt.imshow(mask.argmax(0), cmap="jet")
    plt.title("GT Mask")
    plt.subplot(1, 3, 3)
    plt.imshow(pred.argmax(0), cmap="jet")
    plt.title("Pred Mask")
    plt.show()
