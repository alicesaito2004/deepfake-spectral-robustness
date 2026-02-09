"""
Adversarial robustness evaluation.

Runs PGD attacks at multiple epsilon values against pixel, spectrum, and dual-branch
classifiers. Outputs accuracy vs epsilon curves.

Usage:
    PYTHONPATH=. python scripts/eval_adversarial.py
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm

from src.classifier import PixelClassifier, SpectrumClassifier, DualBranchClassifier
from src.dataset import DeepfakeDataset, load_splits, get_paths_for_split
from src.fft import compute_spectrum

# Adjustable parameters
EPSILONS = [0, 1/255, 2/255, 4/255, 8/255, 16/255, 32/255, 64/255, 128/255]
# EPSILONS = [0, 1/255, 2/255, 4/255, 8/255, 16/255]
PGD_STEPS = 20
N_SAMPLES = 500

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


def get_test_loader(n_samples: int, batch_size: int = 32):
    data_dir = Path("data/processed")
    real_paths = sorted([str(p) for p in (data_dir / "real").glob("*.png")])
    fake_paths = sorted([str(p) for p in (data_dir / "fake").glob("*.png")])
    
    splits = load_splits("config/splits.json")
    real_test, fake_test = get_paths_for_split(real_paths, fake_paths, splits, "test")
    
    n_each = n_samples // 2
    real_test = real_test[:n_each]
    fake_test = fake_test[:n_each]
    
    dataset = DeepfakeDataset(real_test, fake_test)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader


def pgd_attack_pixel(model, images, labels, epsilon, alpha, num_steps):
    if epsilon == 0:
        return images.clone()
    
    criterion = nn.CrossEntropyLoss()
    perturbed = images.clone().detach()
    perturbed = perturbed + torch.empty_like(perturbed).uniform_(-epsilon, epsilon)
    perturbed = torch.clamp(perturbed, 0.0, 1.0)
    
    for _ in range(num_steps):
        perturbed.requires_grad_(True)
        outputs = model(perturbed)
        loss = criterion(outputs, labels)
        loss.backward()
        
        grad_sign = perturbed.grad.sign()
        perturbed = perturbed.detach() + alpha * grad_sign
        delta = torch.clamp(perturbed - images, -epsilon, epsilon)
        perturbed = torch.clamp(images + delta, 0.0, 1.0)
    
    return perturbed.detach()


def pgd_attack_spectrum(model, images, labels, epsilon, alpha, num_steps):
    if epsilon == 0:
        return images.clone()
    
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
    if epsilon == 0:
        return images.clone()
    
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


def evaluate_model(model, loader, epsilon, model_type):
    model.eval()
    alpha = epsilon / 4 if epsilon > 0 else 0
    correct = 0
    total = 0
    
    for images, spectra, labels in tqdm(loader, desc=f"eps={epsilon*255:.0f}/255", leave=False):
        images = images.to(DEVICE)
        spectra = spectra.to(DEVICE)
        labels = labels.to(DEVICE)
        
        if model_type == "pixel":
            adv_images = pgd_attack_pixel(model, images, labels, epsilon, alpha, PGD_STEPS)
            with torch.no_grad():
                outputs = model(adv_images)
        
        elif model_type == "spectrum":
            adv_images = pgd_attack_spectrum(model, images, labels, epsilon, alpha, PGD_STEPS)
            with torch.no_grad():
                adv_spectra = compute_spectrum(adv_images)
                outputs = model(adv_spectra)
        
        else:
            adv_images = pgd_attack_dual(model, images, labels, epsilon, alpha, PGD_STEPS)
            with torch.no_grad():
                adv_spectra = compute_spectrum(adv_images)
                outputs = model(adv_images, adv_spectra)
        
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    return correct / total


def main():
    os.makedirs("outputs/adversarial", exist_ok=True)
    
    print(f"Device: {DEVICE}")
    print(f"Loading {N_SAMPLES} test samples...")
    loader = get_test_loader(N_SAMPLES)
    
    print("\nLoading models...")
    pixel_model = PixelClassifier()
    pixel_model.load_state_dict(torch.load("checkpoints/pixel_classifier.pt", map_location=DEVICE, weights_only=True))
    pixel_model.to(DEVICE).eval()
    
    spectrum_model = SpectrumClassifier()
    spectrum_model.load_state_dict(torch.load("checkpoints/spectrum_classifier.pt", map_location=DEVICE, weights_only=True))
    spectrum_model.to(DEVICE).eval()
    
    dual_model = DualBranchClassifier()
    dual_model.load_state_dict(torch.load("checkpoints/dual_classifier.pt", map_location=DEVICE, weights_only=True))
    dual_model.to(DEVICE).eval()
    
    results = {
        "pixel": [],
        "spectrum": [],
        "dual": [],
    }
    
    print("\n" + "="*50)
    print("Running PGD-20 attacks")
    print("="*50)
    
    for eps in EPSILONS:
        print(f"\nEpsilon = {eps*255:.0f}/255")
        
        print("  Attacking pixel classifier...")
        acc = evaluate_model(pixel_model, loader, eps, "pixel")
        results["pixel"].append(acc)
        print(f"    Accuracy: {acc:.4f}")
        
        print("  Attacking spectrum classifier...")
        acc = evaluate_model(spectrum_model, loader, eps, "spectrum")
        results["spectrum"].append(acc)
        print(f"    Accuracy: {acc:.4f}")
        
        print("  Attacking dual classifier...")
        acc = evaluate_model(dual_model, loader, eps, "dual")
        results["dual"].append(acc)
        print(f"    Accuracy: {acc:.4f}")
    
    print("\n" + "="*50)
    print("Results Summary")
    print("="*50)
    print(f"{'Epsilon':<12} {'Pixel':<12} {'Spectrum':<12} {'Dual':<12}")
    print("-"*48)
    for i, eps in enumerate(EPSILONS):
        print(f"{eps*255:.0f}/255{'':<7} {results['pixel'][i]:<12.4f} {results['spectrum'][i]:<12.4f} {results['dual'][i]:<12.4f}")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    eps_labels = [f"{e*255:.0f}/255" for e in EPSILONS]
    
    ax.plot(eps_labels, results["pixel"], "o-", linewidth=2, markersize=8, label="Pixel", color="steelblue")
    ax.plot(eps_labels, results["spectrum"], "s-", linewidth=2, markersize=8, label="Spectrum", color="indianred")
    ax.plot(eps_labels, results["dual"], "^-", linewidth=2, markersize=8, label="Dual", color="seagreen")
    
    ax.set_xlabel("Perturbation (ε)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Adversarial Robustness: PGD-20 Attack", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    save_path = "outputs/adversarial/robustness_curves.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved plot: {save_path}")
    
    with open("outputs/adversarial/results.txt", "w") as f:
        f.write("Adversarial Robustness Results (PGD-20)\n")
        f.write("="*50 + "\n\n")
        f.write(f"{'Epsilon':<12} {'Pixel':<12} {'Spectrum':<12} {'Dual':<12}\n")
        f.write("-"*48 + "\n")
        for i, eps in enumerate(EPSILONS):
            f.write(f"{eps*255:.0f}/255{'':<7} {results['pixel'][i]:<12.4f} {results['spectrum'][i]:<12.4f} {results['dual'][i]:<12.4f}\n")
    print("Saved results: outputs/adversarial/results.txt")


if __name__ == "__main__":
    main()
