import os
import json
import torch
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom

from modules import Improved3DUNet

NUM_CLASSES = 6
MODEL_PATH = "deep_supervision_unet.pth"
# If splits.json exists, we will use its test split; make sure to change this if it doesn't
IMAGE_PATH = (
    "/home/groups/comp3710/HipMRI_Study_open/semantic_MRs/H017_Week6_LFOV.nii.gz"
)
OUTPUT_DIR = "predictions_test"  # predictions for test split will be saved here
SPLITS_PATH = "splits.json"


def predict_one(model, image_path, device):
    model.eval()

    # Load and preprocess the image using nibabel
    image_nii = nib.load(image_path)
    original_affine = image_nii.affine
    image_array = image_nii.get_fdata()

    # Transpose to (D, H, W)
    image_array = np.transpose(image_array, (2, 0, 1))
    original_shape = image_array.shape

    # Resample to the size the model expects
    target_shape = (64, 128, 128)
    zoom_factors = (
        target_shape[0] / original_shape[0],
        target_shape[1] / original_shape[1],
        target_shape[2] / original_shape[2],
    )

    image_array = zoom(image_array, zoom_factors, order=1, mode="constant", cval=0.0)

    # Normalize
    if np.max(image_array) > np.min(image_array):
        image_array = (image_array - np.min(image_array)) / (
            np.max(image_array) - np.min(image_array)
        )

    # Add batch and channel dimensions
    image_tensor = (
        torch.from_numpy(image_array).unsqueeze(0).unsqueeze(0).float().to(device)
    )

    with torch.no_grad():
        output = model(image_tensor)

        # If model returns multiple outputs (deep supervision), pick the primary one.
        if isinstance(output, (list, tuple)):
            primary = output[0]
        else:
            primary = output

        # primary should be a tensor of shape [B, C, D, H, W]
        probs = torch.softmax(primary, dim=1)
        prediction = torch.argmax(probs, dim=1).squeeze(0).cpu().numpy()

    # Resample prediction back to original size
    zoom_factors_inverse = (
        original_shape[0] / prediction.shape[0],
        original_shape[1] / prediction.shape[1],
        original_shape[2] / prediction.shape[2],
    )
    prediction_resampled = zoom(
        prediction, zoom_factors_inverse, order=0, mode="constant", cval=0.0
    )

    # Transpose back to (H, W, D) for saving
    prediction_resampled = np.transpose(prediction_resampled, (1, 2, 0))

    return prediction_resampled.astype(np.uint8), original_affine


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = Improved3DUNet(in_channels=1, out_channels=NUM_CLASSES).to(device)
    ckpt = torch.load(MODEL_PATH, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    model.load_state_dict(ckpt)

    # Determine images to predict: prefer test split from splits.json
    test_images = None
    if os.path.exists(SPLITS_PATH):
        try:
            with open(SPLITS_PATH, "r") as f:
                splits = json.load(f)
            test_images = splits.get("test_image_files", None)
            if test_images:
                print(f"Found test split in {SPLITS_PATH}: {len(test_images)} images.")
        except Exception as e:
            print(f"Failed to read {SPLITS_PATH}: {e}")

    if test_images:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        for img_path in test_images:
            try:
                pred_mask, affine = predict_one(model, img_path, device)
                base = os.path.basename(img_path)
                out_name = (
                    os.path.splitext(os.path.splitext(base)[0])[0] + "_pred.nii.gz"
                    if base.endswith(".nii.gz")
                    else base + "_pred.nii.gz"
                )
                out_path = os.path.join(OUTPUT_DIR, out_name)
                nib.save(nib.Nifti1Image(pred_mask, affine), out_path)
                print(f"Saved: {out_path}")
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        print(f"All test predictions saved in: {OUTPUT_DIR}")
    else:
        # Fallback: single image mode using IMAGE_PATH and single OUTPUT_PATH
        OUTPUT_PATH = "predicted_segmentation_multiclass2.nii.gz"
        print(f"No test split found; falling back to single image: {IMAGE_PATH}")
        pred_mask, affine = predict_one(model, IMAGE_PATH, device)
        nib.save(nib.Nifti1Image(pred_mask, affine), OUTPUT_PATH)
        print(f"Segmentation saved to {OUTPUT_PATH}")
