"""
Process Stable Diffusion images for cross-generator testing.

Usage:
    1. Download SD images and organize in data/raw/stable_diffusion/:
       data/raw/stable_diffusion/
           sd15/
           sd21/
           sdxl/
    
    2. Run: python scripts/setup_cross_generator.py
    
    3. Images resized to 256x256 and saved to data/cross_generator/
"""

import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from tqdm import tqdm


RAW_DIR = Path("data/raw/stable_diffusion")
OUTPUT_DIR = Path("data/cross_generator")
IMG_SIZE = 256

SD_FOLDERS = ["sd15", "sd21", "sdxl"]


def get_image_files(directory: Path) -> list[Path]:
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.webp"]
    files = []
    for ext in extensions:
        files.extend(directory.rglob(ext))
    return sorted(files)


def resize_image(args: tuple) -> bool:
    src, dst = args
    try:
        img = Image.open(src).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
        img.save(dst, "PNG")
        return True
    except Exception as e:
        print(f"Failed: {src} - {e}")
        return False


def process_folder(name: str) -> int:
    src_dir = RAW_DIR / name
    dst_dir = OUTPUT_DIR / name
    
    if not src_dir.exists():
        alt_names = [name.upper(), name.replace("sd", "SD"), name.replace("sd", "SD_")]
        for alt in alt_names:
            if (RAW_DIR / alt).exists():
                src_dir = RAW_DIR / alt
                break
    
    if not src_dir.exists():
        print(f"Skipping {name}: {src_dir} not found")
        return 0
    
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    files = get_image_files(src_dir)
    if not files:
        print(f"No images in {src_dir}")
        return 0
    
    tasks = [(f, dst_dir / f"{i:05d}.png") for i, f in enumerate(files)]
    
    print(f"Processing {len(tasks)} images for {name}...")
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(tqdm(executor.map(resize_image, tasks), total=len(tasks), desc=name))
    
    return sum(results)


def main():
    if not RAW_DIR.exists():
        print(f"Create {RAW_DIR} and add SD images first.")
        print("Expected structure:")
        print("  data/raw/stable_diffusion/")
        print("      sd15/")
        print("      sd21/")
        print("      sdxl/")
        return
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    total = 0
    for folder in SD_FOLDERS:
        n = process_folder(folder)
        total += n
        print(f"  {folder}: {n} images")
    
    print(f"\nTotal: {total} cross-generator test images")
    print(f"Saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
