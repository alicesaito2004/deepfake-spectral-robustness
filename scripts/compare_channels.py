"""
Compare grayscale vs per-channel (RGB) spectra discriminability.

Computes a simple separability metric (difference of means / pooled std)
for both representations to determine which is more discriminative.
"""

import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.fft import compute_spectrum, azimuthal_average


REAL_DIR = "data/processed/real"
FAKE_DIR = "data/processed/fake"
OUTPUT_DIR = "outputs/channel_comparison"
IMG_SIZE = 256
SEED = 42
N_SAMPLES = 100


transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])


def load_random_images(directory: str, n: int, seed: int) -> list[torch.Tensor]:
    extensions = {".jpg", ".jpeg", ".png"}
    files = sorted([f for f in os.listdir(directory) if os.path.splitext(f)[1].lower() in extensions])
    rng = random.Random(seed)
    selected = rng.sample(files, min(n, len(files)))
    tensors = []
    for f in selected:
        img = Image.open(os.path.join(directory, f)).convert("RGB")
        tensors.append(transform(img))
    return tensors


def compute_separability(real_features: np.ndarray, fake_features: np.ndarray) -> float:
    """Compute separability as |mean_real - mean_fake| / pooled_std."""
    mean_real = real_features.mean(axis=0)
    mean_fake = fake_features.mean(axis=0)
    std_real = real_features.std(axis=0)
    std_fake = fake_features.std(axis=0)
    pooled_std = np.sqrt((std_real ** 2 + std_fake ** 2) / 2) + 1e-8
    separability = np.abs(mean_real - mean_fake) / pooled_std
    return separability.mean()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading images...")
    real_imgs = load_random_images(REAL_DIR, N_SAMPLES, SEED)
    fake_imgs = load_random_images(FAKE_DIR, N_SAMPLES, SEED + 1)

    print(f"Computing grayscale spectra for {N_SAMPLES} real and {N_SAMPLES} fake...")
    real_gray_profiles = []
    fake_gray_profiles = []
    for img in real_imgs:
        spec = compute_spectrum(img, per_channel=False).squeeze(0)
        profile = azimuthal_average(spec).numpy()
        real_gray_profiles.append(profile)
    for img in fake_imgs:
        spec = compute_spectrum(img, per_channel=False).squeeze(0)
        profile = azimuthal_average(spec).numpy()
        fake_gray_profiles.append(profile)

    real_gray_profiles = np.array(real_gray_profiles)
    fake_gray_profiles = np.array(fake_gray_profiles)
    gray_separability = compute_separability(real_gray_profiles, fake_gray_profiles)

    print(f"Computing per-channel (RGB) spectra...")
    real_rgb_profiles = []
    fake_rgb_profiles = []
    for img in real_imgs:
        spec = compute_spectrum(img, per_channel=True)  # (3, H, W)
        profiles = []
        for c in range(3):
            profile = azimuthal_average(spec[c:c+1]).numpy()
            profiles.append(profile)
        real_rgb_profiles.append(np.concatenate(profiles))
    for img in fake_imgs:
        spec = compute_spectrum(img, per_channel=True)
        profiles = []
        for c in range(3):
            profile = azimuthal_average(spec[c:c+1]).numpy()
            profiles.append(profile)
        fake_rgb_profiles.append(np.concatenate(profiles))

    real_rgb_profiles = np.array(real_rgb_profiles)
    fake_rgb_profiles = np.array(fake_rgb_profiles)
    rgb_separability = compute_separability(real_rgb_profiles, fake_rgb_profiles)

    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"Grayscale separability:   {gray_separability:.4f}")
    print(f"Per-channel separability: {rgb_separability:.4f}")
    print()
    if rgb_separability > gray_separability:
        winner = "PER-CHANNEL (RGB)"
        ratio = rgb_separability / gray_separability
    else:
        winner = "GRAYSCALE"
        ratio = gray_separability / rgb_separability
    print(f"Winner: {winner} ({ratio:.2f}x more discriminative)")
    print("=" * 50)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    freqs = np.arange(1, real_gray_profiles.shape[1] + 1)
    axes[0].plot(freqs, real_gray_profiles.mean(0), "b-", lw=2, label="Real")
    axes[0].plot(freqs, fake_gray_profiles.mean(0), "r-", lw=2, label="Fake")
    axes[0].fill_between(freqs, 
                          real_gray_profiles.mean(0) - real_gray_profiles.std(0),
                          real_gray_profiles.mean(0) + real_gray_profiles.std(0),
                          alpha=0.2, color="blue")
    axes[0].fill_between(freqs,
                          fake_gray_profiles.mean(0) - fake_gray_profiles.std(0),
                          fake_gray_profiles.mean(0) + fake_gray_profiles.std(0),
                          alpha=0.2, color="red")
    axes[0].set_xlabel("Spatial Frequency")
    axes[0].set_ylabel("Log-Magnitude")
    axes[0].set_title(f"Grayscale (separability: {gray_separability:.3f})")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    n_freq = real_gray_profiles.shape[1]
    colors = ["red", "green", "blue"]
    channel_names = ["R", "G", "B"]
    for c in range(3):
        start = c * n_freq
        end = (c + 1) * n_freq
        real_c = real_rgb_profiles[:, start:end]
        fake_c = fake_rgb_profiles[:, start:end]
        axes[1].plot(freqs, real_c.mean(0), "-", color=colors[c], lw=2, 
                     label=f"Real {channel_names[c]}", alpha=0.7)
        axes[1].plot(freqs, fake_c.mean(0), "--", color=colors[c], lw=2,
                     label=f"Fake {channel_names[c]}", alpha=0.7)
    axes[1].set_xlabel("Spatial Frequency")
    axes[1].set_ylabel("Log-Magnitude")
    axes[1].set_title(f"Per-Channel RGB (separability: {rgb_separability:.3f})")
    axes[1].legend(ncol=2, fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "grayscale_vs_rgb.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved: {save_path}")

    summary_path = os.path.join(OUTPUT_DIR, "result.txt")
    with open(summary_path, "w") as f:
        f.write(f"Grayscale separability:   {gray_separability:.4f}\n")
        f.write(f"Per-channel separability: {rgb_separability:.4f}\n")
        f.write(f"Winner: {winner}\n")
    print(f"Summary saved: {summary_path}")
    print(f"\nNotify Team 2: Use {winner} for spectrum input.")


if __name__ == "__main__":
    main()
