"""
Train pixel, spectrum, and dual-branch classifiers.

Usage:
    PYTHONPATH=. python scripts/train.py
    PYTHONPATH=. python scripts/train.py --model pixel --epochs 20
    PYTHONPATH=. python scripts/train.py --model all --batch-size 64
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.classifier import PixelClassifier, SpectrumClassifier, DualBranchClassifier
from src.dataset import DeepfakeDataset, load_splits, get_paths_for_split
from src.fft import compute_spectrum


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


def get_data_loaders(batch_size: int, num_workers: int = 4):
    data_dir = Path("data/processed")
    real_paths = sorted([str(p) for p in (data_dir / "real").glob("*.png")])
    fake_paths = sorted([str(p) for p in (data_dir / "fake").glob("*.png")])
    
    splits = load_splits("config/splits.json")
    
    loaders = {}
    for split_name in ["train", "val", "test"]:
        real_split, fake_split = get_paths_for_split(real_paths, fake_paths, splits, split_name)
        dataset = DeepfakeDataset(real_split, fake_split)
        shuffle = split_name == "train"
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )
    
    return loaders


def train_epoch(model, loader, criterion, optimizer, model_type: str):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Train", leave=False)
    for images, spectra, labels in pbar:
        images = images.to(DEVICE)
        spectra = spectra.to(DEVICE)
        labels = labels.to(DEVICE)
        
        optimizer.zero_grad()
        
        if model_type == "pixel":
            outputs = model(images)
        elif model_type == "spectrum":
            outputs = model(spectra)
        else:
            outputs = model(images, spectra)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix(loss=loss.item(), acc=correct/total)
    
    return total_loss / total, correct / total


def evaluate(model, loader, criterion, model_type: str):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, spectra, labels in loader:
            images = images.to(DEVICE)
            spectra = spectra.to(DEVICE)
            labels = labels.to(DEVICE)
            
            if model_type == "pixel":
                outputs = model(images)
            elif model_type == "spectrum":
                outputs = model(spectra)
            else:
                outputs = model(images, spectra)
            
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    return total_loss / total, correct / total


def train_model(model_type: str, epochs: int, batch_size: int, lr: float):
    print(f"\n{'='*50}")
    print(f"Training {model_type} classifier")
    print(f"{'='*50}")
    print(f"Device: {DEVICE}")
    
    loaders = get_data_loaders(batch_size)
    
    if model_type == "pixel":
        model = PixelClassifier()
    elif model_type == "spectrum":
        model = SpectrumClassifier()
    else:
        model = DualBranchClassifier()
    
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    best_val_acc = 0
    checkpoint_path = f"checkpoints/{model_type}_classifier.pt"
    
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        train_loss, train_acc = train_epoch(model, loaders["train"], criterion, optimizer, model_type)
        val_loss, val_acc = evaluate(model, loaders["val"], criterion, model_type)
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        scheduler.step(val_loss)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint (val_acc: {val_acc:.4f})")
    
    plot_training_curves(history, model_type)
    
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    test_loss, test_acc = evaluate(model, loaders["test"], criterion, model_type)
    print(f"\nTest Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    
    return test_acc


def plot_training_curves(history: dict, model_type: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["train_loss"]) + 1)
    
    axes[0].plot(epochs, history["train_loss"], "b-", label="Train")
    axes[0].plot(epochs, history["val_loss"], "r-", label="Val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{model_type.capitalize()} Classifier - Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(epochs, history["train_acc"], "b-", label="Train")
    axes[1].plot(epochs, history["val_acc"], "r-", label="Val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title(f"{model_type.capitalize()} Classifier - Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs("outputs/training", exist_ok=True)
    save_path = f"outputs/training/{model_type}_curves.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved training curves: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="all", choices=["pixel", "spectrum", "dual", "all"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    
    os.makedirs("checkpoints", exist_ok=True)
    
    models = ["pixel", "spectrum", "dual"] if args.model == "all" else [args.model]
    
    results = {}
    for model_type in models:
        acc = train_model(model_type, args.epochs, args.batch_size, args.lr)
        results[model_type] = acc
    
    print("\n" + "="*50)
    print("Final Results")
    print("="*50)
    for model_type, acc in results.items():
        print(f"{model_type}: {acc:.4f}")
    
    if len(results) > 1:
        plot_comparison(results)


def plot_comparison(results: dict):
    fig, ax = plt.subplots(figsize=(8, 5))
    
    models = list(results.keys())
    accs = [results[m] for m in models]
    colors = ["steelblue", "indianred", "seagreen"]
    
    bars = ax.bar(models, accs, color=colors[:len(models)], alpha=0.8)
    
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Classifier Comparison - Clean Accuracy")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis="y")
    
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{acc:.3f}", ha="center", fontsize=12)
    
    plt.tight_layout()
    save_path = "outputs/training/classifier_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved comparison plot: {save_path}")


if __name__ == "__main__":
    main()
