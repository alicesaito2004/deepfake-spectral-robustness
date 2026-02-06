"""
Adversarial robustness evaluation.

Runs PGD attacks at multiple epsilon values against pixel, spectrum, and dual-branch
classifiers. Outputs accuracy vs epsilon curves.

Usage:
    PYTHONPATH=. python scripts/eval_adversarial.py \
        --pixel-ckpt checkpoints/pixel_classifier.pt \
        --spectrum-ckpt checkpoints/spectrum_classifier.pt \
        --dual-ckpt checkpoints/dual_classifier.pt \
        --n-samples 500
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm

from src.attacks import fgsm_attack, pgd_attack
from src.fft import compute_spectrum


EPSILONS = [1/255, 2/255, 4/255, 8/255, 16/255]
PGD_STEPS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_test_data(data_dir: str, n_samples: int):
    """Load test images and labels."""
    from src.dataset import DeepfakeDataset, load_splits, get_paths_for_split
    from pathlib import Path
    
    real_dir = Path(data_dir) / "real"
    fake_dir = Path(data_dir) / "fake"
    
    real_paths = sorted([str(p) for p in real_dir.glob("*.png")])
    fake_paths = sorted([str(p) for p in fake_dir.glob("*.png")])
    
    splits = load_splits("config/splits.json")
    real_test, fake_test = get_paths_for_split(real_paths, fake_paths, splits, "test")
    
    n_each = n_samples // 2
    real_test = real_test[:n_each]
    fake_test = fake_test[:n_each]
    
    dataset = DeepfakeDataset(real_test, fake_test)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    
    return loader


def evaluate_clean(model, loader, model_type: str) -> float:
    """Evaluate clean accuracy."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, spectra, labels in loader:
            images, spectra, labels = images.to(DEVICE), spectra.to(DEVICE), labels.to(DEVICE)
            
            if model_type == "pixel":
                outputs = model(images)
            elif model_type == "spectrum":
                outputs = model(spectra)
            else:  # dual
                outputs = model(images, spectra)
            
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    return correct / total


def evaluate_adversarial(model, loader, epsilon: float, model_type: str) -> float:
    """Evaluate accuracy under PGD attack."""
    model.eval()
    correct = 0
    total = 0
    alpha = epsilon / 4
    
    for images, spectra, labels in loader:
        images, spectra, labels = images.to(DEVICE), spectra.to(DEVICE), labels.to(DEVICE)
        
        if model_type == "pixel":
            adv_images = pgd_attack(model, images, labels, epsilon, alpha, PGD_STEPS)
            with torch.no_grad():
                outputs = model(adv_images)
        
        elif model_type == "spectrum":
            adv_images = pgd_attack_through_fft(model, images, labels, epsilon, alpha, PGD_STEPS)
            with torch.no_grad():
                adv_spectra = compute_spectrum(adv_images)
                outputs = model(adv_spectra)
        
        else:  # dual
            adv_images = pgd_attack_dual(model, images, labels, epsilon, alpha, PGD_STEPS)
            with torch.no_grad():
                adv_spectra = compute_spectrum(adv_images)
                outputs = model(adv_images, adv_spectra)
        
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    return correct / total


def pgd_attack_through_fft(model, images, labels, epsilon, alpha, num_steps):
    """PGD attack where gradients flow through FFT to pixel space."""
    criterion = nn.CrossEntropyLoss()
    perturbed = images.clone().detach()
    perturbed = perturbed + torch.empty_like(perturbed).uniform_(-epsilon, epsilon)
    perturbed = torch.clamp(perturbed, 0.0, 1.0)
    
    for _ in range(num_steps):
        perturbed.requires_grad_(True)
        spectra = compute_spectrum(perturbed)
        outputs = model(spectra)
        loss = criterion(outputs, labels)
        loss.backward()
        
        grad_sign = perturbed.grad.sign()
        perturbed = perturbed.detach() + alpha * grad_sign
        delta = torch.clamp(perturbed - images, -epsilon, epsilon)
        perturbed = torch.clamp(images + delta, 0.0, 1.0)
    
    return perturbed.detach()


def pgd_attack_dual(model, images, labels, epsilon, alpha, num_steps):
    """PGD attack on dual-branch model, perturbing pixels."""
    criterion = nn.CrossEntropyLoss()
    perturbed = images.clone().detach()
    perturbed = perturbed + torch.empty_like(perturbed).uniform_(-epsilon, epsilon)
    perturbed = torch.clamp(perturbed, 0.0, 1.0)
    
    for _ in range(num_steps):
        perturbed.requires_grad_(True)
        spectra = compute_spectrum(perturbed)
        outputs = model(perturbed, spectra)
        loss = criterion(outputs, labels)
        loss.backward()
        
        grad_sign = perturbed.grad.sign()
        perturbed = perturbed.detach() + alpha * grad_sign
        delta = torch.clamp(perturbed - images, -epsilon, epsilon)
        perturbed = torch.clamp(images + delta, 0.0, 1.0)
    
    return perturbed.detach()


def verify_gradient_flow():
    """Verify gradients flow through FFT."""
    print("Verifying gradient flow through FFT...")
    
    images = torch.rand(2, 3, 64, 64, requires_grad=True)
    spectra = compute_spectrum(images)
    loss = spectra.sum()
    loss.backward()
    
    grad_norm = images.grad.norm().item()
    if grad_norm > 0:
        print(f"  PASS: Gradient norm = {grad_norm:.6f}")
        return True
    else:
        print("  FAIL: Gradients are zero!")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pixel-ckpt", type=str, default=None)
    parser.add_argument("--spectrum-ckpt", type=str, default=None)
    parser.add_argument("--dual-ckpt", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--output-dir", type=str, default="outputs/adversarial")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if not verify_gradient_flow():
        print("Fix gradient flow before proceeding.")
        return
    
    results = {}
    
    # Load models and evaluate
    # TODO: Implement model loading once Team 2 provides architecture
    # For now, this script is ready to run once checkpoints exist
    
    print("\nWaiting for classifier checkpoints from Team 2.")
    print("Expected files:")
    print("  - checkpoints/pixel_classifier.pt")
    print("  - checkpoints/spectrum_classifier.pt")
    print("  - checkpoints/dual_classifier.pt")
    print("\nOnce available, this script will:")
    print(f"  1. Load {args.n_samples} test images")
    print(f"  2. Run PGD-{PGD_STEPS} at epsilon = {[f'{e*255:.0f}/255' for e in EPSILONS]}")
    print("  3. Output accuracy vs epsilon curves")


if __name__ == "__main__":
    main()
