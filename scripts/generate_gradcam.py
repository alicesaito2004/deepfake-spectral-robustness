"""
Generate Grad-CAM comparison figures for presentation.

Creates side-by-side visualizations showing what each classifier attends to.
"""

import os
import random
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms

from src.classifier import PixelClassifier, SpectrumClassifier, DualBranchClassifier
from src.fft import compute_spectrum
from src.gradcam import GradCAM, DualBranchGradCAM, overlay_cam_on_image


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

IMG_SIZE = 256
OUTPUT_DIR = "outputs/gradcam"

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])


def load_models():
    pixel_model = PixelClassifier()
    pixel_model.load_state_dict(torch.load("checkpoints/pixel_classifier.pt", map_location=DEVICE, weights_only=True))
    pixel_model.to(DEVICE).eval()
    
    spectrum_model = SpectrumClassifier()
    spectrum_model.load_state_dict(torch.load("checkpoints/spectrum_classifier.pt", map_location=DEVICE, weights_only=True))
    spectrum_model.to(DEVICE).eval()
    
    dual_model = DualBranchClassifier()
    dual_model.load_state_dict(torch.load("checkpoints/dual_classifier.pt", map_location=DEVICE, weights_only=True))
    dual_model.to(DEVICE).eval()
    
    return pixel_model, spectrum_model, dual_model


def get_sample_images(n_real=3, n_fake=3, seed=42):
    random.seed(seed)
    
    real_dir = "data/processed/real"
    fake_dir = "data/processed/fake"
    
    real_files = sorted(os.listdir(real_dir))
    fake_files = sorted(os.listdir(fake_dir))
    
    real_samples = random.sample(real_files, n_real)
    fake_samples = random.sample(fake_files, n_fake)
    
    samples = []
    for f in real_samples:
        img = Image.open(os.path.join(real_dir, f)).convert("RGB")
        samples.append((transform(img), 0, f))
    for f in fake_samples:
        img = Image.open(os.path.join(fake_dir, f)).convert("RGB")
        samples.append((transform(img), 1, f))
    
    return samples


def generate_comparison_figure(samples, pixel_model, spectrum_model, dual_model):
    pixel_gradcam = GradCAM(pixel_model, pixel_model.backbone.features[-2])
    spectrum_gradcam = GradCAM(spectrum_model, spectrum_model.backbone.features[-2])
    dual_gradcam = DualBranchGradCAM(
        dual_model,
        dual_model.pixel_backbone.features[-2],
        dual_model.spectrum_backbone.features[-2]
    )
    
    n = len(samples)
    fig, axes = plt.subplots(n, 5, figsize=(15, 3*n))
    
    for i, (img_tensor, label, filename) in enumerate(samples):
        img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
        spectrum = compute_spectrum(img_tensor)
        img_np = img_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
        spec_np = spectrum.squeeze().cpu().numpy()
        
        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title(f"{'Real' if label == 0 else 'Fake'}", fontsize=10)
        axes[i, 0].axis("off")
        
        pixel_cam = pixel_gradcam.generate(img_tensor)
        pixel_overlay = overlay_cam_on_image(img_np, pixel_cam)
        axes[i, 1].imshow(pixel_overlay)
        axes[i, 1].set_title("Pixel CAM" if i == 0 else "", fontsize=10)
        axes[i, 1].axis("off")
        
        spectrum_cam = spectrum_gradcam.generate(spectrum)
        axes[i, 2].imshow(spectrum_cam, cmap="jet")
        axes[i, 2].set_title("Spectrum CAM" if i == 0 else "", fontsize=10)
        axes[i, 2].axis("off")
        
        pixel_cam_dual, spectrum_cam_dual = dual_gradcam.generate(img_tensor, spectrum)
        dual_overlay = overlay_cam_on_image(img_np, pixel_cam_dual)
        axes[i, 3].imshow(dual_overlay)
        axes[i, 3].set_title("Dual (Pixel)" if i == 0 else "", fontsize=10)
        axes[i, 3].axis("off")
        
        axes[i, 4].imshow(spectrum_cam_dual, cmap="jet")
        axes[i, 4].set_title("Dual (Spectrum)" if i == 0 else "", fontsize=10)
        axes[i, 4].axis("off")
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "gradcam_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def generate_single_example(img_path, pixel_model, spectrum_model, dual_model, output_name):
    pixel_gradcam = GradCAM(pixel_model, pixel_model.backbone.features[-2])
    spectrum_gradcam = GradCAM(spectrum_model, spectrum_model.backbone.features[-2])
    dual_gradcam = DualBranchGradCAM(
        dual_model,
        dual_model.pixel_backbone.features[-2],
        dual_model.spectrum_backbone.features[-2]
    )
    
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)
    spectrum = compute_spectrum(img_tensor)
    img_np = img_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    spec_np = spectrum.squeeze().cpu().numpy()
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    axes[0, 0].imshow(img_np)
    axes[0, 0].set_title("Input Image", fontsize=12)
    axes[0, 0].axis("off")
    
    spec_vis = (spec_np - spec_np.min()) / (spec_np.max() - spec_np.min())
    axes[0, 1].imshow(spec_vis, cmap="inferno")
    axes[0, 1].set_title("Frequency Spectrum", fontsize=12)
    axes[0, 1].axis("off")
    
    pixel_cam = pixel_gradcam.generate(img_tensor)
    pixel_overlay = overlay_cam_on_image(img_np, pixel_cam)
    axes[0, 2].imshow(pixel_overlay)
    axes[0, 2].set_title("Pixel Classifier Attention", fontsize=12)
    axes[0, 2].axis("off")
    
    spectrum_cam = spectrum_gradcam.generate(spectrum)
    axes[0, 3].imshow(spectrum_cam, cmap="jet")
    axes[0, 3].set_title("Spectrum Classifier Attention", fontsize=12)
    axes[0, 3].axis("off")
    
    pixel_cam_dual, spectrum_cam_dual = dual_gradcam.generate(img_tensor, spectrum)
    
    axes[1, 0].imshow(pixel_cam, cmap="jet")
    axes[1, 0].set_title("Pixel CAM (raw)", fontsize=12)
    axes[1, 0].axis("off")
    
    dual_pixel_overlay = overlay_cam_on_image(img_np, pixel_cam_dual)
    axes[1, 1].imshow(dual_pixel_overlay)
    axes[1, 1].set_title("Dual - Pixel Branch", fontsize=12)
    axes[1, 1].axis("off")
    
    axes[1, 2].imshow(spectrum_cam_dual, cmap="jet")
    axes[1, 2].set_title("Dual - Spectrum Branch", fontsize=12)
    axes[1, 2].axis("off")
    
    with torch.no_grad():
        pixel_pred = torch.softmax(pixel_model(img_tensor), dim=1)
        spec_pred = torch.softmax(spectrum_model(spectrum), dim=1)
        dual_pred = torch.softmax(dual_model(img_tensor, spectrum), dim=1)
    
    pred_text = f"Pixel: {'Fake' if pixel_pred[0,1] > 0.5 else 'Real'} ({pixel_pred[0,1]:.2%})\n"
    pred_text += f"Spectrum: {'Fake' if spec_pred[0,1] > 0.5 else 'Real'} ({spec_pred[0,1]:.2%})\n"
    pred_text += f"Dual: {'Fake' if dual_pred[0,1] > 0.5 else 'Real'} ({dual_pred[0,1]:.2%})"
    
    axes[1, 3].text(0.5, 0.5, pred_text, ha="center", va="center", fontsize=14,
                    transform=axes[1, 3].transAxes, family="monospace")
    axes[1, 3].set_title("Predictions", fontsize=12)
    axes[1, 3].axis("off")
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, output_name)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Device: {DEVICE}")
    print("Loading models...")
    pixel_model, spectrum_model, dual_model = load_models()
    
    print("\nGenerating comparison figure (3 real + 3 fake)...")
    samples = get_sample_images(n_real=3, n_fake=3)
    generate_comparison_figure(samples, pixel_model, spectrum_model, dual_model)
    
    print("\nGenerating detailed examples...")
    real_img = "data/processed/real/00001.png"
    fake_img = "data/processed/fake/00001.png"
    
    if os.path.exists(real_img):
        generate_single_example(real_img, pixel_model, spectrum_model, dual_model, "example_real.png")
    if os.path.exists(fake_img):
        generate_single_example(fake_img, pixel_model, spectrum_model, dual_model, "example_fake.png")
    
    print(f"\nAll visualizations saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
