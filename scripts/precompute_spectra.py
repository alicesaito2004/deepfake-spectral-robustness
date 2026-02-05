"""
Precompute spectra for all images and cache to disk.

Avoids recomputing FFTs every epoch during training. Run once after
data setup. Saves each spectrum as a .pt file.

Usage:
    python scripts/precompute_spectra.py \
        --real-dir data/celeba \
        --fake-dir data/stylegan2 \
        --output-dir data/spectra \
        --per-channel          # optional: 3-channel spectra
"""

import argparse
import os
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from src.fft import compute_spectrum


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMG_SIZE = 256


def precompute_dir(
    img_dir: str,
    out_dir: str,
    per_channel: bool = False,
) -> int:
    """Compute and save spectra for all images in a directory."""
    os.makedirs(out_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    files = sorted([
        f for f in os.listdir(img_dir)
        if Path(f).suffix.lower() in IMAGE_EXTENSIONS
    ])

    count = 0
    for fname in tqdm(files, desc=f"  {Path(img_dir).name}"):
        stem = Path(fname).stem
        out_path = os.path.join(out_dir, f"{stem}.pt")

        if os.path.exists(out_path):
            count += 1
            continue

        img = Image.open(os.path.join(img_dir, fname)).convert("RGB")
        tensor = transform(img)
        spectrum = compute_spectrum(tensor, per_channel=per_channel)
        torch.save(spectrum, out_path)
        count += 1

    return count


def main():
    parser = argparse.ArgumentParser(description="Precompute spectra")
    parser.add_argument("--real-dir", required=True)
    parser.add_argument("--fake-dir", required=True)
    parser.add_argument("--output-dir", default="data/spectra")
    parser.add_argument("--per-channel", action="store_true",
                        help="Compute per-channel (R,G,B) spectra instead of grayscale")
    args = parser.parse_args()

    print(f"Per-channel: {args.per_channel}")
    print(f"Output: {args.output_dir}")

    real_out = os.path.join(args.output_dir, "celeba")
    fake_out = os.path.join(args.output_dir, "stylegan2")

    nr = precompute_dir(args.real_dir, real_out, args.per_channel)
    nf = precompute_dir(args.fake_dir, fake_out, args.per_channel)

    print(f"\nDone: {nr} real spectra, {nf} fake spectra")
    print(f"Saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
