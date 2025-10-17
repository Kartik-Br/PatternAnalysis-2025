# predict.py
import argparse
import torch
import nibabel as nib
import numpy as np
from modules import ImprovedUNet3D
from dataset import load_nifti_as_array

def predict_volume(model_path, image_path, out_path, device='cuda'):
    dev = torch.device(device if torch.cuda.is_available() else 'cpu')
    # adjust num_classes/base_filters if different in your checkpoint
    ck = torch.load(model_path, map_location=dev, weights_only=False)
    ck = torch.load(model_path, map_location=dev, weights_only=False)

    num_classes = 6   # adjust to match training
    model = ImprovedUNet3D(in_channels=1, out_channels=num_classes, base_filters=16).to(dev)
    model.load_state_dict(ck['model_state'] if 'model_state' in ck else ck)
    model.eval()


    vol, aff, _ = load_nifti_as_array(image_path)
    # simple center crop or pad to model expected patch_size
    # naive: assume model accepts full volume (works if you trained full volumes)
    x = (vol - vol.mean()) / (vol.std() + 1e-8)
    x = np.expand_dims(x, 0)  # channel
    x = np.expand_dims(x, 0)  # batch
    x_t = torch.from_numpy(x).float().to(dev)
    with torch.no_grad():
        logits = model(x_t)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).cpu().numpy()[0]
    # save as nifti

    out_img = nib.Nifti1Image(pred.astype(np.uint8), affine=aff)
    nib.save(out_img, out_path)
    print("Saved prediction to", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    predict_volume(args.model, args.image, args.out)

