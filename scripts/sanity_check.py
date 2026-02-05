"""
Visual sanity check: verify FFT pipeline and confirm spectral differences.

Usage:
    python scripts/sanity_check.py
"""

import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.fft import compute_spectrum, azimuthal_average, band_energy, spectral_residual


REAL_DIR = "data/processed/real"
FAKE_DIR = "data/processed/fake"
OUTPUT_DIR = "outputs/sanity_check"
IMG_SIZE = 256
SEED = 42

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])


def load_random_images(directory: str, n: int, seed: int = SEED) -> list[torch.Tensor]:
    """Load n random images from a directory as tensors."""
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = sorted([
        f for f in os.listdir(directory)
        if os.path.splitext(f)[1].lower() in extensions
    ])
    rng = random.Random(seed)
    selected = rng.sample(files, min(n, len(files)))

    tensors = []
    for f in selected:
        img = Image.open(os.path.join(directory, f)).convert("RGB")
        tensors.append(transform(img))
    return tensors


def plot_image_spectrum_grid(
    real_imgs: list[torch.Tensor],
    fake_imgs: list[torch.Tensor],
    save_path: str,
):
    """Plot 5 real and 5 fake images with their spectra."""
    n = min(5, len(real_imgs), len(fake_imgs))
    fig, axes = plt.subplots(4, n, figsize=(3 * n, 12))

    for i in range(n):
        axes[0, i].imshow(real_imgs[i].permute(1, 2, 0).numpy())
        axes[0, i].set_title("Real" if i == 0 else "")
        axes[0, i].axis("off")

        spec = compute_spectrum(real_imgs[i]).squeeze(0).numpy()
        axes[1, i].imshow(spec, cmap="inferno")
        axes[1, i].set_title("Real Spectrum" if i == 0 else "")
        axes[1, i].axis("off")

        axes[2, i].imshow(fake_imgs[i].permute(1, 2, 0).numpy())
        axes[2, i].set_title("Fake" if i == 0 else "")
        axes[2, i].axis("off")

        spec = compute_spectrum(fake_imgs[i]).squeeze(0).numpy()
        axes[3, i].imshow(spec, cmap="inferno")
        axes[3, i].set_title("Fake Spectrum" if i == 0 else "")
        axes[3, i].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_azimuthal_comparison(
    real_imgs: list[torch.Tensor],
    fake_imgs: list[torch.Tensor],
    save_path: str,
):
    """Plot azimuthally averaged spectra: real vs. fake."""
    real_profiles = []
    for img in real_imgs:
        spec = compute_spectrum(img).squeeze(0)
        profile = azimuthal_average(spec).numpy()
        real_profiles.append(profile)

    fake_profiles = []
    for img in fake_imgs:
        spec = compute_spectrum(img).squeeze(0)
        profile = azimuthal_average(spec).numpy()
        fake_profiles.append(profile)

    real_profiles = np.array(real_profiles)
    fake_profiles = np.array(fake_profiles)
    freqs = np.arange(1, real_profiles.shape[1] + 1)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(freqs, real_profiles.mean(axis=0), "b-", linewidth=2, label="Real (mean)")
    ax.fill_between(
        freqs,
        real_profiles.mean(0) - real_profiles.std(0),
        real_profiles.mean(0) + real_profiles.std(0),
        alpha=0.2, color="blue",
    )
    ax.plot(freqs, fake_profiles.mean(axis=0), "r-", linewidth=2, label="Fake (mean)")
    ax.fill_between(
        freqs,
        fake_profiles.mean(0) - fake_profiles.std(0),
        fake_profiles.mean(0) + fake_profiles.std(0),
        alpha=0.2, color="red",
    )
    ax.set_xlabel("Spatial Frequency")
    ax.set_ylabel("Log-Magnitude")
    ax.set_title("Azimuthally Averaged Spectrum")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_spectrum_difference(
    real_imgs: list[torch.Tensor],
    fake_imgs: list[torch.Tensor],
    save_path: str,
):
    """Plot mean(real spectra) - mean(fake spectra)."""
    real_specs = torch.stack([compute_spectrum(img).squeeze(0) for img in real_imgs])
    fake_specs = torch.stack([compute_spectrum(img).squeeze(0) for img in fake_imgs])

    real_mean = real_specs.mean(dim=0).numpy()
    fake_mean = fake_specs.mean(dim=0).numpy()
    diff = real_mean - fake_mean

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    im0 = axes[0].imshow(real_mean, cmap="inferno")
    axes[0].set_title("Mean Real Spectrum")
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(fake_mean, cmap="inferno")
    axes[1].set_title("Mean Fake Spectrum")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    vmax = max(abs(diff.min()), abs(diff.max()))
    im2 = axes[2].imshow(diff, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[2].set_title("Difference (Real - Fake)")
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_band_energies(
    real_imgs: list[torch.Tensor],
    fake_imgs: list[torch.Tensor],
    save_path: str,
    n_bands: int = 20,
):
    """Plot band energy distributions for real vs. fake."""
    real_energies = []
    for img in real_imgs:
        spec = compute_spectrum(img).squeeze(0)
        energies = band_energy(spec, n_bands=n_bands).numpy()
        real_energies.append(energies)

    fake_energies = []
    for img in fake_imgs:
        spec = compute_spectrum(img).squeeze(0)
        energies = band_energy(spec, n_bands=n_bands).numpy()
        fake_energies.append(energies)

    real_energies = np.array(real_energies)
    fake_energies = np.array(fake_energies)

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(n_bands)
    width = 0.35

    ax.bar(x - width / 2, real_energies.mean(0), width, label="Real", color="steelblue", alpha=0.8)
    ax.bar(x + width / 2, fake_energies.mean(0), width, label="Fake", color="indianred", alpha=0.8)

    ax.set_xlabel("Frequency Band (low to high)")
    ax.set_ylabel("Mean Log-Magnitude")
    ax.set_title("Band Energy Distribution: Real vs. Fake")
    ax.set_xticks(x)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading images...")
    real_5 = load_random_images(REAL_DIR, 5, seed=SEED)
    fake_5 = load_random_images(FAKE_DIR, 5, seed=SEED + 1)
    real_50 = load_random_images(REAL_DIR, 50, seed=SEED + 2)
    fake_50 = load_random_images(FAKE_DIR, 50, seed=SEED + 3)

    print("\n1. Image + Spectrum grid...")
    plot_image_spectrum_grid(real_5, fake_5, os.path.join(OUTPUT_DIR, "01_image_spectrum_grid.png"))

    print("2. Azimuthal average comparison...")
    plot_azimuthal_comparison(real_50, fake_50, os.path.join(OUTPUT_DIR, "02_azimuthal_comparison.png"))

    print("3. 2D spectrum difference map...")
    plot_spectrum_difference(real_50, fake_50, os.path.join(OUTPUT_DIR, "03_spectrum_difference.png"))

    print("4. Band energy distributions...")
    plot_band_energies(real_50, fake_50, os.path.join(OUTPUT_DIR, "04_band_energies.png"))

    print(f"\nAll plots saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
