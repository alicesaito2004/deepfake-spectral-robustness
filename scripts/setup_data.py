"""
Prepare manually downloaded dataset for training.

Usage:
    1. Download dataset and extract to data/raw/
       Expected structure:
           data/raw/real/  (or data/raw/Real/, case-insensitive)
           data/raw/fake/  (or data/raw/Fake/, or data/raw/ai/, etc.)
    
    2. Run: python scripts/setup_data.py
    
    3. Images will be resized to 256x256 and saved to:
           data/processed/real/
           data/processed/fake/
"""

import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from tqdm import tqdm


DATA_RAW = Path("data/raw")
DATA_PROCESSED = Path("data/processed")
IMG_SIZE = 256


def find_image_dir(base_path: Path, keywords: list[str]) -> Path | None:
    """Find a subdirectory matching any of the keywords (case-insensitive)."""
    if not base_path.exists():
        return None
    for item in base_path.iterdir():
        if item.is_dir():
            name_lower = item.name.lower()
            for kw in keywords:
                if kw in name_lower:
                    return item
    return None


def get_image_files(directory: Path) -> list[Path]:
    """Recursively get all image files from directory."""
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.webp"]
    files = []
    for ext in extensions:
        files.extend(directory.rglob(ext))
    return sorted(files)


def resize_image(args: tuple) -> bool:
    """Resize single image to IMG_SIZE x IMG_SIZE."""
    src, dst = args
    try:
        img = Image.open(src).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
        img.save(dst, "PNG")
        return True
    except Exception as e:
        print(f"Failed to process {src}: {e}")
        return False


def process_images(src_dir: Path, dst_dir: Path, label: str) -> int:
    """Resize all images from src_dir to dst_dir."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    src_files = get_image_files(src_dir)
    if not src_files:
        print(f"No images found in {src_dir}")
        return 0
    
    tasks = []
    for i, src in enumerate(src_files):
        dst = dst_dir / f"{i:05d}.png"
        tasks.append((src, dst))
    
    print(f"Processing {len(tasks)} {label} images...")
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(tqdm(
            executor.map(resize_image, tasks),
            total=len(tasks),
            desc=label
        ))
    
    return sum(results)


def main():
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    
    real_dir = find_image_dir(DATA_RAW, ["real"])
    fake_dir = find_image_dir(DATA_RAW, ["fake", "ai", "synthetic", "generated"])
    
    if real_dir is None:
        print(f"Could not find real images directory in {DATA_RAW}")
        print("Expected: data/raw/real/ or similar")
        print("\nAvailable directories:")
        for item in DATA_RAW.iterdir():
            if item.is_dir():
                print(f"  {item.name}/")
        return
    
    if fake_dir is None:
        print(f"Could not find fake images directory in {DATA_RAW}")
        print("Expected: data/raw/fake/ or data/raw/ai/ or similar")
        print("\nAvailable directories:")
        for item in DATA_RAW.iterdir():
            if item.is_dir():
                print(f"  {item.name}/")
        return
    
    print(f"Real images: {real_dir}")
    print(f"Fake images: {fake_dir}")
    
    n_real = process_images(real_dir, DATA_PROCESSED / "real", "real")
    n_fake = process_images(fake_dir, DATA_PROCESSED / "fake", "fake")
    
    print(f"\nData setup complete.")
    print(f"  Real: {n_real} images")
    print(f"  Fake: {n_fake} images")
    print(f"\nNext step: python scripts/generate_splits.py")


if __name__ == "__main__":
    main()
