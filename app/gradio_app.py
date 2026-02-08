"""
Gradio web interface for deepfake detection.

Usage:
    PYTHONPATH=. python app/gradio_app.py
"""

import gradio as gr
import torch
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

pixel_model = None
spectrum_model = None
dual_model = None
pixel_gradcam = None
spectrum_gradcam = None
dual_gradcam = None


def load_models():
    global pixel_model, spectrum_model, dual_model
    global pixel_gradcam, spectrum_gradcam, dual_gradcam
    
    pixel_model = PixelClassifier()
    pixel_model.load_state_dict(torch.load("checkpoints/pixel_classifier.pt", map_location=DEVICE, weights_only=True))
    pixel_model.to(DEVICE).eval()
    pixel_gradcam = GradCAM(pixel_model, pixel_model.backbone.features[-2])
    
    spectrum_model = SpectrumClassifier()
    spectrum_model.load_state_dict(torch.load("checkpoints/spectrum_classifier.pt", map_location=DEVICE, weights_only=True))
    spectrum_model.to(DEVICE).eval()
    spectrum_gradcam = GradCAM(spectrum_model, spectrum_model.backbone.features[-2])
    
    dual_model = DualBranchClassifier()
    dual_model.load_state_dict(torch.load("checkpoints/dual_classifier.pt", map_location=DEVICE, weights_only=True))
    dual_model.to(DEVICE).eval()
    dual_gradcam = DualBranchGradCAM(
        dual_model,
        dual_model.pixel_backbone.features[-2],
        dual_model.spectrum_backbone.features[-2]
    )


def predict(image, model_choice, show_gradcam):
    if image is None:
        return None, "Please upload an image", None, None
    
    if pixel_model is None:
        try:
            load_models()
        except Exception as e:
            return None, f"Error loading models: {e}", None, None
    
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    img_tensor = transform(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    spectrum_tensor = compute_spectrum(img_tensor)
    
    with torch.no_grad():
        if model_choice == "Pixel":
            outputs = pixel_model(img_tensor)
        elif model_choice == "Spectrum":
            outputs = spectrum_model(spectrum_tensor)
        else:
            outputs = dual_model(img_tensor, spectrum_tensor)
        
        probs = torch.softmax(outputs, dim=1)
        pred = outputs.argmax(dim=1).item()
        conf = probs[0, pred].item()
    
    label = "FAKE" if pred == 1 else "REAL"
    result_text = f"Prediction: {label}\nConfidence: {conf:.2%}"
    
    spectrum_vis = spectrum_tensor.squeeze().cpu().numpy()
    spectrum_vis = (spectrum_vis - spectrum_vis.min()) / (spectrum_vis.max() - spectrum_vis.min())
    spectrum_vis = (spectrum_vis * 255).astype(np.uint8)
    
    gradcam_vis = None
    if show_gradcam:
        img_np = img_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
        
        if model_choice == "Pixel":
            cam = pixel_gradcam.generate(img_tensor)
            gradcam_vis = overlay_cam_on_image(img_np, cam)
            gradcam_vis = (gradcam_vis * 255).astype(np.uint8)
        
        elif model_choice == "Spectrum":
            cam = spectrum_gradcam.generate(spectrum_tensor)
            import matplotlib.pyplot as plt
            cmap = plt.cm.jet
            gradcam_vis = cmap(cam)[:, :, :3]
            gradcam_vis = (gradcam_vis * 255).astype(np.uint8)
        
        else:
            pixel_cam, spectrum_cam = dual_gradcam.generate(img_tensor, spectrum_tensor)
            gradcam_vis = overlay_cam_on_image(img_np, pixel_cam)
            gradcam_vis = (gradcam_vis * 255).astype(np.uint8)
    
    return result_text, spectrum_vis, gradcam_vis


def create_interface():
    with gr.Blocks(title="Deepfake Detector") as demo:
        gr.Markdown("# Deepfake Detection with Spectral Analysis")
        gr.Markdown("Upload an image to detect if it's real or AI-generated.")
        
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="Upload Image", type="pil")
                model_choice = gr.Radio(
                    choices=["Pixel", "Spectrum", "Dual"],
                    value="Dual",
                    label="Classifier"
                )
                show_gradcam = gr.Checkbox(label="Show Grad-CAM", value=True)
                submit_btn = gr.Button("Analyze", variant="primary")
            
            with gr.Column():
                result_text = gr.Textbox(label="Result", lines=3)
                spectrum_output = gr.Image(label="Frequency Spectrum")
                gradcam_output = gr.Image(label="Grad-CAM Attention")
        
        submit_btn.click(
            fn=predict,
            inputs=[input_image, model_choice, show_gradcam],
            outputs=[result_text, spectrum_output, gradcam_output]
        )
        
        gr.Markdown("### How it works")
        gr.Markdown("""
        - **Pixel Classifier**: Analyzes raw RGB pixels
        - **Spectrum Classifier**: Analyzes frequency domain (FFT magnitude)
        - **Dual Classifier**: Combines both approaches
        
        The Grad-CAM visualization shows which regions/frequencies the model focuses on.
        """)
    
    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=False)
