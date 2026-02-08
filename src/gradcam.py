"""
Grad-CAM implementation for visualizing classifier attention.

Shows which regions (pixel classifier) or frequency bands (spectrum classifier)
the model focuses on when making predictions.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


class GradCAM:
    def __init__(self, model, target_layer):
        """
        Args:
            model: Trained classifier model.
            target_layer: Layer to compute Grad-CAM for (usually last conv layer).
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self._register_hooks()
    
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate(self, input_tensor, target_class=None):
        """Generate Grad-CAM heatmap.
        
        Args:
            input_tensor: Input image/spectrum tensor (1, C, H, W).
            target_class: Class to generate CAM for. If None, uses predicted class.
        
        Returns:
            cam: Numpy array (H, W) with values in [0, 1].
        """
        self.model.eval()
        
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot)
        
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam


class DualBranchGradCAM:
    def __init__(self, model, pixel_target_layer, spectrum_target_layer):
        """Grad-CAM for dual-branch classifier."""
        self.model = model
        self.pixel_layer = pixel_target_layer
        self.spectrum_layer = spectrum_target_layer
        
        self.pixel_gradients = None
        self.pixel_activations = None
        self.spectrum_gradients = None
        self.spectrum_activations = None
        
        self._register_hooks()
    
    def _register_hooks(self):
        def pixel_forward_hook(module, input, output):
            self.pixel_activations = output.detach()
        
        def pixel_backward_hook(module, grad_input, grad_output):
            self.pixel_gradients = grad_output[0].detach()
        
        def spectrum_forward_hook(module, input, output):
            self.spectrum_activations = output.detach()
        
        def spectrum_backward_hook(module, grad_input, grad_output):
            self.spectrum_gradients = grad_output[0].detach()
        
        self.pixel_layer.register_forward_hook(pixel_forward_hook)
        self.pixel_layer.register_full_backward_hook(pixel_backward_hook)
        self.spectrum_layer.register_forward_hook(spectrum_forward_hook)
        self.spectrum_layer.register_full_backward_hook(spectrum_backward_hook)
    
    def generate(self, pixel_input, spectrum_input, target_class=None):
        """Generate Grad-CAM for both branches."""
        self.model.eval()
        
        output = self.model(pixel_input, spectrum_input)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot)
        
        pixel_weights = self.pixel_gradients.mean(dim=(2, 3), keepdim=True)
        pixel_cam = (pixel_weights * self.pixel_activations).sum(dim=1, keepdim=True)
        pixel_cam = F.relu(pixel_cam)
        pixel_cam = F.interpolate(pixel_cam, size=pixel_input.shape[2:], mode='bilinear', align_corners=False)
        pixel_cam = pixel_cam.squeeze().cpu().numpy()
        pixel_cam = (pixel_cam - pixel_cam.min()) / (pixel_cam.max() - pixel_cam.min() + 1e-8)
        
        spectrum_weights = self.spectrum_gradients.mean(dim=(2, 3), keepdim=True)
        spectrum_cam = (spectrum_weights * self.spectrum_activations).sum(dim=1, keepdim=True)
        spectrum_cam = F.relu(spectrum_cam)
        spectrum_cam = F.interpolate(spectrum_cam, size=spectrum_input.shape[2:], mode='bilinear', align_corners=False)
        spectrum_cam = spectrum_cam.squeeze().cpu().numpy()
        spectrum_cam = (spectrum_cam - spectrum_cam.min()) / (spectrum_cam.max() - spectrum_cam.min() + 1e-8)
        
        return pixel_cam, spectrum_cam


def overlay_cam_on_image(image, cam, alpha=0.5):
    """Overlay CAM heatmap on image.
    
    Args:
        image: Numpy array (H, W, 3) in [0, 1].
        cam: Numpy array (H, W) in [0, 1].
        alpha: Blending factor.
    
    Returns:
        Blended image as numpy array.
    """
    cmap = plt.cm.jet
    heatmap = cmap(cam)[:, :, :3]
    blended = (1 - alpha) * image + alpha * heatmap
    return np.clip(blended, 0, 1)


def visualize_gradcam(
    image_tensor,
    spectrum_tensor,
    cam,
    prediction,
    confidence,
    model_type,
    save_path=None,
):
    """Create visualization figure for Grad-CAM output."""
    image_np = image_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    spectrum_np = spectrum_tensor.squeeze().cpu().numpy()
    
    if model_type == "pixel":
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(image_np)
        axes[0].set_title("Input Image")
        axes[0].axis("off")
        
        axes[1].imshow(cam, cmap="jet")
        axes[1].set_title("Grad-CAM")
        axes[1].axis("off")
        
        overlay = overlay_cam_on_image(image_np, cam)
        axes[2].imshow(overlay)
        axes[2].set_title(f"Pred: {'Fake' if prediction else 'Real'} ({confidence:.2f})")
        axes[2].axis("off")
    
    elif model_type == "spectrum":
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(spectrum_np, cmap="inferno")
        axes[0].set_title("Input Spectrum")
        axes[0].axis("off")
        
        axes[1].imshow(cam, cmap="jet")
        axes[1].set_title("Grad-CAM (Frequency Attention)")
        axes[1].axis("off")
        
        axes[2].imshow(image_np)
        axes[2].set_title(f"Original | Pred: {'Fake' if prediction else 'Real'} ({confidence:.2f})")
        axes[2].axis("off")
    
    else:  # dual
        pixel_cam, spectrum_cam = cam
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        
        axes[0, 0].imshow(image_np)
        axes[0, 0].set_title("Input Image")
        axes[0, 0].axis("off")
        
        axes[0, 1].imshow(pixel_cam, cmap="jet")
        axes[0, 1].set_title("Pixel Branch CAM")
        axes[0, 1].axis("off")
        
        overlay = overlay_cam_on_image(image_np, pixel_cam)
        axes[0, 2].imshow(overlay)
        axes[0, 2].set_title("Pixel Overlay")
        axes[0, 2].axis("off")
        
        axes[1, 0].imshow(spectrum_np, cmap="inferno")
        axes[1, 0].set_title("Input Spectrum")
        axes[1, 0].axis("off")
        
        axes[1, 1].imshow(spectrum_cam, cmap="jet")
        axes[1, 1].set_title("Spectrum Branch CAM")
        axes[1, 1].axis("off")
        
        axes[1, 2].text(0.5, 0.5, f"Prediction: {'Fake' if prediction else 'Real'}\nConfidence: {confidence:.2f}",
                        ha="center", va="center", fontsize=14, transform=axes[1, 2].transAxes)
        axes[1, 2].axis("off")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
    
    return fig
