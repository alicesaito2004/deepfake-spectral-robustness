"""
Grad-CAM implementation for visualizing classifier attention.

Shows which regions (pixel classifier) or frequency bands (spectrum classifier)
the model focuses on when making predictions.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.handles = []
        self._register_hooks()
    
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.clone()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].clone()
        
        self.handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.handles.append(self.target_layer.register_backward_hook(backward_hook))
    
    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()
    
    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        self.model.zero_grad()
        
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        if self.gradients is None or self.activations is None:
            return np.zeros((input_tensor.shape[2], input_tensor.shape[3]))
        
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        return cam


class DualBranchGradCAM:
    def __init__(self, model, pixel_target_layer, spectrum_target_layer):
        self.model = model
        self.pixel_layer = pixel_target_layer
        self.spectrum_layer = spectrum_target_layer
        
        self.pixel_gradients = None
        self.pixel_activations = None
        self.spectrum_gradients = None
        self.spectrum_activations = None
        self.handles = []
        self._register_hooks()
    
    def _register_hooks(self):
        def pixel_forward_hook(module, input, output):
            self.pixel_activations = output.clone()
        
        def pixel_backward_hook(module, grad_input, grad_output):
            self.pixel_gradients = grad_output[0].clone()
        
        def spectrum_forward_hook(module, input, output):
            self.spectrum_activations = output.clone()
        
        def spectrum_backward_hook(module, grad_input, grad_output):
            self.spectrum_gradients = grad_output[0].clone()
        
        self.handles.append(self.pixel_layer.register_forward_hook(pixel_forward_hook))
        self.handles.append(self.pixel_layer.register_backward_hook(pixel_backward_hook))
        self.handles.append(self.spectrum_layer.register_forward_hook(spectrum_forward_hook))
        self.handles.append(self.spectrum_layer.register_backward_hook(spectrum_backward_hook))
    
    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()
    
    def generate(self, pixel_input, spectrum_input, target_class=None):
        self.model.eval()
        self.model.zero_grad()
        
        output = self.model(pixel_input, spectrum_input)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        pixel_cam = np.zeros((pixel_input.shape[2], pixel_input.shape[3]))
        if self.pixel_gradients is not None and self.pixel_activations is not None:
            weights = self.pixel_gradients.mean(dim=(2, 3), keepdim=True)
            cam = (weights * self.pixel_activations).sum(dim=1, keepdim=True)
            cam = F.relu(cam)
            cam = F.interpolate(cam, size=pixel_input.shape[2:], mode='bilinear', align_corners=False)
            pixel_cam = cam.squeeze().detach().cpu().numpy()
            if pixel_cam.max() > pixel_cam.min():
                pixel_cam = (pixel_cam - pixel_cam.min()) / (pixel_cam.max() - pixel_cam.min())
        
        spectrum_cam = np.zeros((spectrum_input.shape[2], spectrum_input.shape[3]))
        if self.spectrum_gradients is not None and self.spectrum_activations is not None:
            weights = self.spectrum_gradients.mean(dim=(2, 3), keepdim=True)
            cam = (weights * self.spectrum_activations).sum(dim=1, keepdim=True)
            cam = F.relu(cam)
            cam = F.interpolate(cam, size=spectrum_input.shape[2:], mode='bilinear', align_corners=False)
            spectrum_cam = cam.squeeze().detach().cpu().numpy()
            if spectrum_cam.max() > spectrum_cam.min():
                spectrum_cam = (spectrum_cam - spectrum_cam.min()) / (spectrum_cam.max() - spectrum_cam.min())
        
        return pixel_cam, spectrum_cam


def overlay_cam_on_image(image, cam, alpha=0.5):
    cmap = plt.cm.jet
    heatmap = cmap(cam)[:, :, :3]
    blended = (1 - alpha) * image + alpha * heatmap
    return np.clip(blended, 0, 1)
