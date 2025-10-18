import torch
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom

from modules import Improved3DUNet


NUM_CLASSES = 6
MODEL_PATH = 'prostate_segmentation_model_multiclass.pth'
IMAGE_PATH = '/home/groups/comp3710/HipMRI_Study_open/semantic_MRs/W012_Week3_LFOV.nii.gz'  # IMPORTANT: Change to your test image
OUTPUT_PATH = 'predicted_segmentation_multiclass.nii.gz'

def predict(model, image_path, device):
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
        target_shape[2] / original_shape[2]
    )
    
    image_array = zoom(image_array, zoom_factors, order=1, mode='constant', cval=0.0)

    # Normalize
    if np.max(image_array) > np.min(image_array):
        image_array = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))
    
    # Add batch and channel dimensions
    image_tensor = torch.from_numpy(image_array).unsqueeze(0).unsqueeze(0).float().to(device)

    with torch.no_grad():
        output = model(image_tensor)
        # Get the predicted class by taking argmax across the channel dimension
        prediction = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    # Resample prediction back to original size
    zoom_factors_inverse = (
        original_shape[0] / prediction.shape[0],
        original_shape[1] / prediction.shape[1],
        original_shape[2] / prediction.shape[2]
    )
    prediction_resampled = zoom(prediction, zoom_factors_inverse, order=0, mode='constant', cval=0.0)

    # Transpose back to (H, W, D) for saving
    prediction_resampled = np.transpose(prediction_resampled, (1, 2, 0))

    return prediction_resampled.astype(np.uint8), original_affine

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model = Improved3DUNet(in_channels=1, out_channels=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print("Model loaded.")

    # Perform prediction
    print(f"Predicting segmentation for {IMAGE_PATH}...")
    predicted_mask, affine = predict(model, IMAGE_PATH, device)
    
    # Save the prediction as a .nii.gz file
    output_image = nib.Nifti1Image(predicted_mask, affine)
    nib.save(output_image, OUTPUT_PATH)
    
    print(f"Segmentation saved to {OUTPUT_PATH}")
