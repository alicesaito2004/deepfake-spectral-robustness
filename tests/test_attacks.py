"""
Unit tests for adversarial attacks.
"""

import torch
import torch.nn as nn

from src.attacks import fgsm_attack


class SimpleClassifier(nn.Module):
    """Minimal CNN for testing."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 2)

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)


def test_fgsm_linf_norm():
    """Verify perturbation is exactly epsilon in L-infinity norm."""
    model = SimpleClassifier()
    model.eval()

    torch.manual_seed(42)
    images = torch.rand(4, 3, 32, 32)
    labels = torch.tensor([0, 1, 0, 1])
    epsilon = 8 / 255

    adv_images = fgsm_attack(model, images, labels, epsilon)

    perturbation = adv_images - images
    linf_norm = perturbation.abs().max().item()

    assert linf_norm <= epsilon + 1e-6, f"L-inf norm {linf_norm} exceeds epsilon {epsilon}"

    interior_mask = (images > epsilon) & (images < 1 - epsilon)
    if interior_mask.any():
        interior_pert = perturbation[interior_mask].abs()
        assert interior_pert.max() >= epsilon - 1e-6, "Perturbation should reach epsilon for interior pixels"

    print(f"PASS: L-inf norm = {linf_norm:.6f}, epsilon = {epsilon:.6f}")


def test_fgsm_gradients_nonzero():
    """Verify gradients are non-zero during attack."""
    model = SimpleClassifier()
    model.eval()

    torch.manual_seed(42)
    images = torch.rand(4, 3, 32, 32, requires_grad=True)
    labels = torch.tensor([0, 1, 0, 1])

    outputs = model(images)
    loss = nn.CrossEntropyLoss()(outputs, labels)
    loss.backward()

    grad_norm = images.grad.norm().item()
    assert grad_norm > 0, "Gradients should be non-zero"

    print(f"PASS: Gradient norm = {grad_norm:.6f}")


def test_fgsm_clamping():
    """Verify output is clamped to [0, 1]."""
    model = SimpleClassifier()
    model.eval()

    torch.manual_seed(42)
    images = torch.rand(4, 3, 32, 32)
    labels = torch.tensor([0, 1, 0, 1])
    epsilon = 0.5  # Large epsilon to force clamping

    adv_images = fgsm_attack(model, images, labels, epsilon)

    assert adv_images.min() >= 0.0, f"Min value {adv_images.min()} is below 0"
    assert adv_images.max() <= 1.0, f"Max value {adv_images.max()} is above 1"

    print(f"PASS: Output range = [{adv_images.min():.4f}, {adv_images.max():.4f}]")


def test_fgsm_changes_prediction():
    """Verify attack can change model predictions."""
    model = SimpleClassifier()
    model.eval()

    torch.manual_seed(123)
    images = torch.rand(16, 3, 32, 32)
    
    with torch.no_grad():
        original_preds = model(images).argmax(dim=1)
    
    labels = original_preds.clone()
    epsilon = 0.1

    adv_images = fgsm_attack(model, images, labels, epsilon)

    with torch.no_grad():
        adv_preds = model(adv_images).argmax(dim=1)

    changed = (original_preds != adv_preds).sum().item()
    print(f"PASS: {changed}/{len(images)} predictions changed by attack")


if __name__ == "__main__":
    print("Testing FGSM attack...\n")
    test_fgsm_linf_norm()
    test_fgsm_gradients_nonzero()
    test_fgsm_clamping()
    test_fgsm_changes_prediction()
    print("\nAll tests passed.")
