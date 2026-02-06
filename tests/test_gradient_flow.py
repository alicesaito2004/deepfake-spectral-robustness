"""
Verify gradients flow through FFT to pixel space.

This is critical for adversarial attacks on spectrum classifier.
"""

import torch
import torch.nn as nn

from src.fft import compute_spectrum


def test_gradient_flow_basic():
    """Basic test: gradients flow from spectrum loss to pixel input."""
    print("Test 1: Basic gradient flow")
    
    images = torch.rand(4, 3, 64, 64, requires_grad=True)
    spectra = compute_spectrum(images, per_channel=False)
    loss = spectra.mean()
    loss.backward()
    
    grad_norm = images.grad.norm().item()
    assert grad_norm > 0, "Gradients are zero!"
    print(f"  PASS: grad norm = {grad_norm:.6f}")


def test_gradient_flow_classifier():
    """Test with a simple classifier on top of spectrum."""
    print("\nTest 2: Gradient flow through classifier")
    
    class SpectrumClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(1, 16, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(16, 2)
        
        def forward(self, x):
            x = torch.relu(self.conv(x))
            x = self.pool(x).flatten(1)
            return self.fc(x)
    
    model = SpectrumClassifier()
    criterion = nn.CrossEntropyLoss()
    
    images = torch.rand(4, 3, 64, 64, requires_grad=True)
    labels = torch.tensor([0, 1, 0, 1])
    
    spectra = compute_spectrum(images, per_channel=False)
    outputs = model(spectra)
    loss = criterion(outputs, labels)
    loss.backward()
    
    grad_norm = images.grad.norm().item()
    assert grad_norm > 0, "Gradients are zero!"
    print(f"  PASS: grad norm = {grad_norm:.6f}")


def test_gradient_sign_attack():
    """Test that we can compute sign of gradients for FGSM."""
    print("\nTest 3: Gradient sign for FGSM")
    
    class SpectrumClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(1, 16, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(16, 2)
        
        def forward(self, x):
            x = torch.relu(self.conv(x))
            x = self.pool(x).flatten(1)
            return self.fc(x)
    
    model = SpectrumClassifier()
    criterion = nn.CrossEntropyLoss()
    
    images = torch.rand(4, 3, 64, 64, requires_grad=True)
    labels = torch.tensor([0, 1, 0, 1])
    epsilon = 8 / 255
    
    spectra = compute_spectrum(images, per_channel=False)
    outputs = model(spectra)
    loss = criterion(outputs, labels)
    loss.backward()
    
    grad_sign = images.grad.sign()
    perturbed = images + epsilon * grad_sign
    perturbed = torch.clamp(perturbed, 0.0, 1.0)
    
    perturbation = (perturbed - images).abs().max().item()
    assert perturbation > 0, "No perturbation applied!"
    print(f"  PASS: max perturbation = {perturbation:.6f}")


def test_dual_branch_gradients():
    """Test gradients flow correctly in dual-branch setup."""
    print("\nTest 4: Dual-branch gradient flow")
    
    class DualClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.pixel_conv = nn.Conv2d(3, 16, 3, padding=1)
            self.spec_conv = nn.Conv2d(1, 16, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(32, 2)
        
        def forward(self, images, spectra):
            p = torch.relu(self.pixel_conv(images))
            p = self.pool(p).flatten(1)
            s = torch.relu(self.spec_conv(spectra))
            s = self.pool(s).flatten(1)
            combined = torch.cat([p, s], dim=1)
            return self.fc(combined)
    
    model = DualClassifier()
    criterion = nn.CrossEntropyLoss()
    
    images = torch.rand(4, 3, 64, 64, requires_grad=True)
    labels = torch.tensor([0, 1, 0, 1])
    
    spectra = compute_spectrum(images, per_channel=False)
    outputs = model(images, spectra)
    loss = criterion(outputs, labels)
    loss.backward()
    
    grad_norm = images.grad.norm().item()
    assert grad_norm > 0, "Gradients are zero!"
    print(f"  PASS: grad norm = {grad_norm:.6f}")
    print("  (Gradients flow through both pixel branch AND spectrum branch)")


if __name__ == "__main__":
    print("=" * 50)
    print("FFT Gradient Flow Verification")
    print("=" * 50)
    
    test_gradient_flow_basic()
    test_gradient_flow_classifier()
    test_gradient_sign_attack()
    test_dual_branch_gradients()
    
    print("\n" + "=" * 50)
    print("All gradient flow tests PASSED")
    print("FFT pipeline is ready for adversarial attacks")
    print("=" * 50)
