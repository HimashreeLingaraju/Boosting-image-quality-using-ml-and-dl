import os
import glob
import cv2
import numpy as np
import torch
import RRDBNet_arch as arch

def process_image(model, img):
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)

    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round().astype(np.uint8)
    return output

# Set up the model and device
model_path = 'models/RRDB_PSNR_x4.pth'  # Make sure this path is correct
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.to(device)

# Set up input and output directories
input_folder = 'input_images'  # Place your input images in this folder
output_folder = 'output_images'  # Upscaled images will be saved here

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Process images
for img_path in glob.glob(os.path.join(input_folder, '*')):
    # Read image
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    
    if img is None:
        print(f"Error reading image: {img_path}")
        continue
    
    # Get the base name of the file
    base_name = os.path.basename(img_path)
    
    print(f"Processing: {base_name}")
    
    # Upscale image
    output = process_image(model, img)
    
    # Save the result
    output_path = os.path.join(output_folder, f"upscaled_{base_name}")
    cv2.imwrite(output_path, output)
    
    print(f"Saved: {output_path}")

print("All images processed.")