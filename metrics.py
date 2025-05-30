import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import os

def calculate_metrics(original, upscaled):
    # Ensure images are in the same size
    original = cv2.resize(original, (upscaled.shape[1], upscaled.shape[0]))
    
    # Calculate PSNR
    psnr_value = psnr(original, upscaled)
    
    # Calculate SSIM
    min_dim = min(original.shape[0], original.shape[1], upscaled.shape[0], upscaled.shape[1])
    win_size = min(7, min_dim)  # Use 7 or the smallest dimension, whichever is smaller
    if win_size % 2 == 0:
        win_size -= 1  # Ensure win_size is odd
    
    ssim_value = ssim(original, upscaled, win_size=win_size, channel_axis=-1)
    
    return psnr_value, ssim_value

def plot_comparison(original, upscaled, metrics, output_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    ax1.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    ax2.imshow(cv2.cvtColor(upscaled, cv2.COLOR_BGR2RGB))
    ax2.set_title('Upscaled Image')
    ax2.axis('off')
    
    plt.suptitle(f'Image Comparison\nPSNR: {metrics[0]:.2f} dB, SSIM: {metrics[1]:.4f}')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def process_images(original_path, upscaled_path, output_dir):
    original = cv2.imread(original_path)
    upscaled = cv2.imread(upscaled_path)
    
    if original is None:
        raise FileNotFoundError(f"Could not read the original image: {original_path}")
    if upscaled is None:
        raise FileNotFoundError(f"Could not read the upscaled image: {upscaled_path}")
    
    metrics = calculate_metrics(original, upscaled)
    
    output_path = os.path.join(output_dir, 'comparison.png')
    plot_comparison(original, upscaled, metrics, output_path)
    
    return metrics

if __name__ == '__main__':
    # Use relative paths
    original_path = os.path.join('input_images', 'image.jpg')
    upscaled_path = os.path.join('output_images', 'image.jpg')
    output_dir = 'metrics_output'
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        metrics = process_images(original_path, upscaled_path, output_dir)
        print(f'PSNR: {metrics[0]:.2f} dB')
        print(f'SSIM: {metrics[1]:.4f}')
        print(f'Comparison image saved in: {os.path.abspath(output_dir)}')
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")