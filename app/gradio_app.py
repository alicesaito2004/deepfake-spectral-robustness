"""
Gradio web interface for deepfake detection with adversarial attack demo.
Single page layout with side-by-side comparison.

Usage:
    PYTHONPATH=. python app/gradio_app.py
"""

import gradio as gr
import torch
import torch.nn as nn
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

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# Global models
pixel_model = None
spectrum_model = None
dual_model = None


def load_models():
    global pixel_model, spectrum_model, dual_model
    
    pixel_model = PixelClassifier()
    pixel_model.load_state_dict(torch.load("checkpoints/pixel_classifier.pt", map_location=DEVICE, weights_only=True))
    pixel_model.to(DEVICE).eval()
    
    spectrum_model = SpectrumClassifier()
    spectrum_model.load_state_dict(torch.load("checkpoints/spectrum_classifier.pt", map_location=DEVICE, weights_only=True))
    spectrum_model.to(DEVICE).eval()
    
    dual_model = DualBranchClassifier()
    dual_model.load_state_dict(torch.load("checkpoints/dual_classifier.pt", map_location=DEVICE, weights_only=True))
    dual_model.to(DEVICE).eval()


def get_model(model_choice):
    if pixel_model is None:
        load_models()
    
    if model_choice == "Pixel":
        return pixel_model, "pixel"
    elif model_choice == "Spectrum":
        return spectrum_model, "spectrum"
    else:
        return dual_model, "dual"


def fgsm_attack(model, images, labels, epsilon, model_type):
    criterion = nn.CrossEntropyLoss()
    images = images.clone().detach().requires_grad_(True)
    
    if model_type == "pixel":
        outputs = model(images)
    elif model_type == "spectrum":
        spectra = compute_spectrum(images)
        outputs = model(spectra)
    else:
        spectra = compute_spectrum(images)
        outputs = model(images, spectra)
    
    loss = criterion(outputs, labels)
    loss.backward()
    
    grad_sign = images.grad.sign()
    perturbed = images + epsilon * grad_sign
    perturbed = torch.clamp(perturbed, 0.0, 1.0)
    
    return perturbed.detach()


def pgd_attack(model, images, labels, epsilon, model_type, num_steps=20):
    criterion = nn.CrossEntropyLoss()
    alpha = epsilon / 4
    
    perturbed = images.clone().detach()
    perturbed = perturbed + torch.empty_like(perturbed).uniform_(-epsilon, epsilon)
    perturbed = torch.clamp(perturbed, 0.0, 1.0)
    
    for _ in range(num_steps):
        perturbed.requires_grad_(True)
        
        if model_type == "pixel":
            outputs = model(perturbed)
        elif model_type == "spectrum":
            spectra = compute_spectrum(perturbed)
            outputs = model(spectra)
        else:
            spectra = compute_spectrum(perturbed)
            outputs = model(perturbed, spectra)
        
        loss = criterion(outputs, labels)
        loss.backward()
        
        grad_sign = perturbed.grad.sign()
        perturbed = perturbed.detach() + alpha * grad_sign
        delta = torch.clamp(perturbed - images, -epsilon, epsilon)
        perturbed = torch.clamp(images + delta, 0.0, 1.0)
    
    return perturbed.detach()


def get_prediction(model, img_tensor, model_type):
    with torch.no_grad():
        if model_type == "pixel":
            outputs = model(img_tensor)
        elif model_type == "spectrum":
            spectra = compute_spectrum(img_tensor)
            outputs = model(spectra)
        else:
            spectra = compute_spectrum(img_tensor)
            outputs = model(img_tensor, spectra)
        
        probs = torch.softmax(outputs, dim=1)
        pred = outputs.argmax(dim=1).item()
        conf = probs[0, pred].item()
    
    label = "FAKE" if pred == 1 else "REAL"
    return label, conf


def get_gradcam(model, img_tensor, model_type):
    """Generate Grad-CAM heatmap."""
    img_np = img_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    
    if model_type == "pixel":
        gradcam = GradCAM(model, model.backbone.features[-2])
        cam = gradcam.generate(img_tensor)
        overlay = overlay_cam_on_image(img_np, cam)
        gradcam.remove_hooks()
        return (overlay * 255).astype(np.uint8)
    
    elif model_type == "spectrum":
        spectra = compute_spectrum(img_tensor)
        gradcam = GradCAM(model, model.backbone.features[-2])
        cam = gradcam.generate(spectra)
        gradcam.remove_hooks()
        # Return heatmap on spectrum (as jet colormap)
        import matplotlib.pyplot as plt
        cmap = plt.cm.jet
        heatmap = cmap(cam)[:, :, :3]
        return (heatmap * 255).astype(np.uint8)
    
    else:  # dual
        spectra = compute_spectrum(img_tensor)
        gradcam = DualBranchGradCAM(
            model,
            model.pixel_backbone.features[-2],
            model.spectrum_backbone.features[-2]
        )
        pixel_cam, spectrum_cam = gradcam.generate(img_tensor, spectra)
        gradcam.remove_hooks()
        overlay = overlay_cam_on_image(img_np, pixel_cam)
        return (overlay * 255).astype(np.uint8)


def run_demo(image, model_choice, attack_type, epsilon, show_gradcam):
    if image is None:
        return None, None, None, None, "Upload an image", "", "", ""
    
    model, model_type = get_model(model_choice)
    
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    img_tensor = transform(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    
    # Original prediction
    orig_label, orig_conf = get_prediction(model, img_tensor, model_type)
    orig_pred = 1 if orig_label == "FAKE" else 0
    target_label = torch.tensor([orig_pred]).to(DEVICE)
    
    # Run attack
    epsilon_val = epsilon / 255.0
    
    if attack_type == "FGSM":
        adv_tensor = fgsm_attack(model, img_tensor, target_label, epsilon_val, model_type)
    else:
        adv_tensor = pgd_attack(model, img_tensor, target_label, epsilon_val, model_type)
    
    # Adversarial prediction
    adv_label, adv_conf = get_prediction(model, adv_tensor, model_type)
    
    # Convert to display images
    orig_img = img_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    orig_img = (orig_img * 255).astype(np.uint8)
    
    adv_img = adv_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    adv_img = (adv_img * 255).astype(np.uint8)
    
    # Perturbation magnified 10x
    perturbation = (adv_tensor - img_tensor).squeeze().permute(1, 2, 0).cpu().numpy()
    perturbation = np.abs(perturbation) * 10
    if perturbation.max() > 0:
        perturbation = (perturbation / perturbation.max() * 255).astype(np.uint8)
    else:
        perturbation = np.zeros_like(orig_img, dtype=np.uint8)
    
    # Grad-CAM
    gradcam_img = None
    gradcam_text = ""
    if show_gradcam:
        try:
            gradcam_img = get_gradcam(model, img_tensor, model_type)
            gradcam_text = f"Grad-CAM\n({model_choice})"
        except Exception as e:
            gradcam_text = f"Grad-CAM error"
    
    # Format text
    orig_text = f"Pred: {orig_label}\nConf: {orig_conf:.1%}"
    
    attack_success = "✓ FLIPPED!" if orig_label != adv_label else "✗ Failed"
    pert_text = f"ε = {epsilon}/255\n{attack_type}\n{attack_success}"
    
    adv_text = f"Pred: {adv_label}\nConf: {adv_conf:.1%}"
    
    return orig_img, perturbation, adv_img, gradcam_img, orig_text, pert_text, adv_text, gradcam_text


def create_interface():
    with gr.Blocks(title="Deepfake Adversarial Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎭 Deepfake Detection — Adversarial Attack Demo")
        gr.Markdown("See how imperceptible perturbations fool deepfake classifiers")
        
        with gr.Row():
            # Left: Controls
            with gr.Column(scale=1):
                input_image = gr.Image(label="Upload Image", type="pil", height=250)
                
                model_choice = gr.Radio(
                    choices=["Pixel", "Spectrum", "Dual"],
                    value="Pixel",
                    label="Classifier"
                )
                
                attack_type = gr.Radio(
                    choices=["FGSM", "PGD-20"],
                    value="FGSM",
                    label="Attack"
                )
                
                epsilon_slider = gr.Slider(
                    minimum=1, maximum=32, step=1, value=8,
                    label="Epsilon (ε)"
                )
                
                show_gradcam = gr.Checkbox(label="Show Grad-CAM", value=True)
                
                attack_btn = gr.Button("⚡ Run Attack", variant="primary", size="lg")
        
        gr.Markdown("---")
        
        # Results: Four columns
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Original")
                orig_display = gr.Image(label=None, show_label=False, height=180)
                orig_pred = gr.Textbox(label=None, show_label=False, lines=2, text_align="center")
            
            with gr.Column():
                gr.Markdown("### Perturbation (10×)")
                pert_display = gr.Image(label=None, show_label=False, height=180)
                pert_info = gr.Textbox(label=None, show_label=False, lines=2, text_align="center")
            
            with gr.Column():
                gr.Markdown("### Adversarial")
                adv_display = gr.Image(label=None, show_label=False, height=180)
                adv_pred = gr.Textbox(label=None, show_label=False, lines=2, text_align="center")
            
            with gr.Column():
                gr.Markdown("### Grad-CAM")
                gradcam_display = gr.Image(label=None, show_label=False, height=180)
                gradcam_text = gr.Textbox(label=None, show_label=False, lines=2, text_align="center")
        
        attack_btn.click(
            fn=run_demo,
            inputs=[input_image, model_choice, attack_type, epsilon_slider, show_gradcam],
            outputs=[orig_display, pert_display, adv_display, gradcam_display, orig_pred, pert_info, adv_pred, gradcam_text]
        )
        
        gr.Markdown("---")
        gr.Markdown("""
        **Tips:** Try ε=8 first (imperceptible). Compare Pixel vs Spectrum vs Dual — which is harder to fool?
        """)
    
    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=False)
