"""
Subsample Stable Diffusion face images with gender balance and resize to 256x256.

Usage:
    python balanced_subsample.py --src /path/to/downloaded --dst data/cross_generator

Expected source structure:
    /path/to/downloaded/
    ├── 512/
    │   ├── man/
    │   └── woman/
    ├── 768/
    │   ├── man/
    │   └── woman/
    └── 1024/
        ├── man/
        └── woman/

Output structure:
    data/cross_generator/
    ├── sd15/   (500 images: 250 man + 250 woman, all 256x256 PNG)
    ├── sd21/   (500 images: 250 man + 250 woman, all 256x256 PNG)
    └── sdxl/   (500 images: 250 man + 250 woman, all 256x256 PNG)
"""

import argparse
import random
from pathlib import Path
from PIL import Image
from tqdm import tqdm


VERSIONS = {
    "512": "sd15",
    "768": "sd21",
    "1024": "sdxl",
}

GENDERS = ["man", "woman"]
VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def get_image_files(directory: Path) -> list:
    return sorted([
        f for f in directory.iterdir()
        if f.suffix.lower() in VALID_EXTENSIONS
    ])


def process_version(src_dir: Path, dst_dir: Path, n_per_gender: int):
    dst_dir.mkdir(parents=True, exist_ok=True)
    total = 0

    for gender in GENDERS:
        gender_dir = src_dir / gender
        if not gender_dir.exists():
            print(f"  WARNING: {gender_dir} does not exist, skipping")
            continue

        files = get_image_files(gender_dir)
        if len(files) == 0:
            print(f"  WARNING: no images found in {gender_dir}")
            continue

        n_select = min(n_per_gender, len(files))
        selected = random.sample(files, n_select)

        for f in tqdm(selected, desc=f"  {gender}", leave=False):
            img = Image.open(f).convert("RGB").resize((256, 256), Image.LANCZOS)
            out_name = f"{gender}_{f.stem}.png"
            img.save(dst_dir / out_name)
            total += 1

    return total


def main():
    parser = argparse.ArgumentParser(description="Subsample SD faces with gender balance")
    parser.add_argument("--src", required=True, help="Path to downloaded Kaggle dataset root")
    parser.add_argument("--dst", default="data/cross_generator", help="Output directory")
    parser.add_argument("--n-per-gender", type=int, default=250, help="Images per gender per SD version")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    src_root = Path(args.src)
    dst_root = Path(args.dst)

    print(f"Source: {src_root}")
    print(f"Destination: {dst_root}")
    print(f"Per gender: {args.n_per_gender} (total per version: {args.n_per_gender * 2})")
    print()

    for src_folder, dst_folder in VERSIONS.items():
        src_dir = src_root / src_folder
        dst_dir = dst_root / dst_folder

        if not src_dir.exists():
            print(f"SKIP: {src_dir} not found")
            continue

        print(f"Processing {src_folder} -> {dst_folder}")
        count = process_version(src_dir, dst_dir, args.n_per_gender)
        print(f"  Saved {count} images to {dst_dir}")

    # Verify
    print("\n--- Verification ---")
    for dst_folder in VERSIONS.values():
        d = dst_root / dst_folder
        if d.exists():
            imgs = list(d.glob("*.png"))
            man_count = len([f for f in imgs if f.name.startswith("man_")])
            woman_count = len([f for f in imgs if f.name.startswith("woman_")])
            sample = Image.open(imgs[0]) if imgs else None
            size_str = f"{sample.size[0]}x{sample.size[1]}" if sample else "N/A"
            print(f"  {dst_folder}: {len(imgs)} total ({man_count} man, {woman_count} woman), size: {size_str}")
        else:
            print(f"  {dst_folder}: NOT FOUND")


if __name__ == "__main__":
    main()