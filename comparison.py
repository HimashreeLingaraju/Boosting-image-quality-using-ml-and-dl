import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import os
import glob
import torch
import RRDBNet_arch as arch

def setup_ai_model():
    model_path = 'models/RRDB_PSNR_x4.pth'  # Make sure this path is correct
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = arch.RRDBNet(3, 3, 64, 23, gc=32)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.to(device)
    return model, device

def process_image_ai(model, device, img):
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)

    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round().astype(np.uint8)
    return output

def upscale_image(image, method, scale_factor, ai_model=None, device=None):
    if method == 'bicubic':
        return cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    elif method == 'lanczos':
        return cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LANCZOS4)
    elif method == 'ai':
        if ai_model is None or device is None:
            raise ValueError("AI model and device must be provided for AI upscaling")
        return process_image_ai(ai_model, device, image)
    else:
        raise ValueError(f"Unknown upscaling method: {method}")

def calculate_metrics(original, upscaled):
    original = cv2.resize(original, (upscaled.shape[1], upscaled.shape[0]))
    return psnr(original, upscaled), ssim(original, upscaled, channel_axis=-1)

def compare_methods(image_path, methods, scale_factor, ai_model=None, device=None):
    original = cv2.imread(image_path)
    if original is None:
        raise FileNotFoundError(f"Could not read the image file: {image_path}")
    results = {}
    
    for method in methods:
        upscaled = upscale_image(original, method, scale_factor, ai_model, device)
        psnr_value, ssim_value = calculate_metrics(original, upscaled)
        results[method] = {'psnr': psnr_value, 'ssim': ssim_value}
    
    return results

def plot_comparison(results, output_path):
    methods = list(results.keys())
    psnr_values = [results[m]['psnr'] for m in methods]
    ssim_values = [results[m]['ssim'] for m in methods]
    
    x = np.arange(len(methods))
    width = 0.35
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    ax1.bar(x - width/2, psnr_values, width, label='PSNR')
    ax1.set_ylabel('PSNR (dB)')
    ax1.set_title('PSNR Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods)
    ax1.legend()
    
    ax2.bar(x - width/2, ssim_values, width, label='SSIM')
    ax2.set_ylabel('SSIM')
    ax2.set_title('SSIM Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def find_image_files(directory):
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(directory, ext)))
    return image_files

if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    methods = ['bicubic', 'lanczos', 'ai']
    scale_factor = 4
    output_dir = 'output'
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Find image files in the current directory
        image_files = find_image_files(current_dir)
        
        if not image_files:
            raise FileNotFoundError(f"No image files found in {current_dir}")
        
        print("Found the following image files:")
        for i, file in enumerate(image_files, 1):
            print(f"{i}. {os.path.basename(file)}")
        
        selection = int(input("Enter the number of the image you want to process: ")) - 1
        image_path = image_files[selection]
        
        # Set up AI model
        try:
            ai_model, device = setup_ai_model()
            print("AI model loaded successfully.")
        except Exception as e:
            print(f"Error loading AI model: {e}")
            print("Proceeding with bicubic and lanczos methods only.")
            methods = ['bicubic', 'lanczos']
            ai_model, device = None, None
        
        results = compare_methods(image_path, methods, scale_factor, ai_model, device)
        
        output_path = os.path.join(output_dir, 'methods_comparison.png')
        plot_comparison(results, output_path)
        
        print(f"\nComparison graph saved to: {os.path.abspath(output_path)}")
        print("\nUpscaling method comparison results:")
        for method, metrics in results.items():
            print(f'{method.capitalize()}:')
            print(f'  PSNR: {metrics["psnr"]:.2f} dB')
            print(f'  SSIM: {metrics["ssim"]:.4f}')
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease ensure that image files are present in the same directory as this script.")
        print("Supported formats: .jpg, .jpeg, .png, .bmp, .tiff")
    except IndexError:
        print("Invalid selection. Please run the script again and choose a valid number.")
    except ValueError:
        print("Invalid input. Please run the script again and enter a valid number.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")