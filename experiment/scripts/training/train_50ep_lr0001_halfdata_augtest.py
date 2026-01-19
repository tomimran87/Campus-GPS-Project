#!/usr/bin/env python3
"""
Experiment: 50 Epochs, LR=0.0001, Half Data Training with Augmented Test Set
=============================================================================

EXPERIMENT DESIGN (as documented for future reference):
--------------------------------------------------------
DATA SPLIT STRATEGY:
    1. Original dataset: 3,646 images
    2. Split 50/50:
       - First 50% (1,823 images): Used for TRAINING and VALIDATION
         * 70% of these → Training set (~1,276 images)
         * 30% of these → Validation set (~547 images)
       - Second 50% (1,823 images): Used for TEST SET creation
         * Apply data augmentation to these images
         * Take 15% of augmented images as test set (~274 images)

RATIONALE:
    - Training on half the data simulates limited data scenarios
    - Testing on augmented versions of UNSEEN images:
      * Tests generalization to new locations
      * Tests robustness to augmentation variations
      * More challenging than testing on same-distribution data
    - This creates a harder but more realistic evaluation

TRAINING CONFIGURATION:
    - Epochs: 50
    - Learning Rate: 0.0001 (lower for stability)
    - Batch Size: 32
    - Optimizer: AdamW with weight_decay=1e-4
    - Scheduler: ReduceLROnPlateau (patience=5, factor=0.5)
    - Early Stopping: patience=10
    - Random Seed: 42

OUTPUT:
    testing/50ep_lr0001_halfdata_augtest/
    ├── ResNet18/
    ├── EfficientNet/
    ├── ConvNeXt/
    ├── comparison/
    └── experiment_log.txt

Author: Training Pipeline
Date: Generated for experiment comparison
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import transforms
from tqdm import tqdm
from pathlib import Path
import json
from datetime import datetime

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent))

from models import ResNetGPS, EfficientNetGPS, ConvNextGPS
from loss import HaversineLoss


# ============================================
# Configuration
# ============================================
CONFIG = {
    "X_PATH": "/home/liranatt/project/main_project/Latest_data/Latest_data/X.npy",
    "Y_PATH": "/home/liranatt/project/main_project/Latest_data/Latest_data/y.npy",
    "BATCH_SIZE": 32,
    "EPOCHS": 50,
    "LR": 0.0001,
    "DATA_FRACTION": 0.5,  # Use 50% for training
    "TEST_FRACTION_FROM_AUG": 0.15,  # 15% of augmented data for test
    "PATIENCE": 10,
    "SEED": 42,
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "OUTPUT_DIR": Path("testing/50ep_lr0001_halfdata_augtest"),
}

MODELS = {
    "ResNet18": ResNetGPS,
    "EfficientNet": EfficientNetGPS,
    "ConvNeXt": ConvNextGPS,
}

COLORS = {
    "ResNet18": "#e41a1c",
    "EfficientNet": "#377eb8",
    "ConvNeXt": "#4daf4a",
}


# ============================================
# Augmentation Classes
# ============================================
class AugmentedTestDataset(Dataset):
    """Dataset that applies augmentation to images for test set."""
    
    def __init__(self, X, y, transform=None):
        """
        Args:
            X: Images array (N, H, W, 3) in [0, 255]
            y: GPS coordinates (N, 2) normalized to [0, 1]
            transform: Augmentation transform
        """
        self.X = X
        self.y = y
        self.transform = transform
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        image = self.X[idx]
        
        # Convert to uint8 if needed
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        gps = self.y[idx]
        
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        
        gps = torch.from_numpy(gps).float() if isinstance(gps, np.ndarray) else gps
        
        return image, gps


def get_test_augmentation():
    """
    Get augmentation transform for test set.
    
    These augmentations simulate real-world variations:
    - Different lighting conditions
    - Camera sensor differences
    - Slight perspective changes
    """
    return transforms.Compose([
        transforms.ToPILImage(),
        
        # Resize to expected size
        transforms.Resize((224, 224)),
        
        # Color jitter - different times of day/weather
        transforms.ColorJitter(
            brightness=0.25,
            contrast=0.25,
            saturation=0.2,
            hue=0.08
        ),
        
        # Random horizontal flip
        transforms.RandomHorizontalFlip(p=0.5),
        
        # Small rotation - phone not held level
        transforms.RandomRotation(degrees=10),
        
        # Random affine for slight perspective changes
        transforms.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05)
        ),
        
        transforms.ToTensor(),
        
        # Note: NOT using ImageNet normalization to match training pipeline
    ])


# ============================================
# Helper Functions
# ============================================
def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate Haversine distance in meters."""
    R = 6371000
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return R * c


def setup_directories():
    """Create output directory structure."""
    for model_name in MODELS.keys():
        model_dir = CONFIG["OUTPUT_DIR"] / model_name
        (model_dir / "plots").mkdir(parents=True, exist_ok=True)
        (model_dir / "geographic").mkdir(parents=True, exist_ok=True)
    
    comparison_dir = CONFIG["OUTPUT_DIR"] / "comparison"
    (comparison_dir / "plots").mkdir(parents=True, exist_ok=True)
    (comparison_dir / "geographic").mkdir(parents=True, exist_ok=True)
    
    print(f"✓ Created directory structure: {CONFIG['OUTPUT_DIR']}")


def load_and_split_data():
    """
    Load data and split according to experiment design:
    - First 50%: Train (70%) + Val (30%)
    - Second 50%: Apply augmentation, use 15% as test
    """
    print("\n" + "=" * 60)
    print("DATA LOADING AND SPLITTING")
    print("=" * 60)
    
    # Set seeds
    np.random.seed(CONFIG["SEED"])
    torch.manual_seed(CONFIG["SEED"])
    
    # Load full data
    X_full = np.load(CONFIG["X_PATH"])
    y_full = np.load(CONFIG["Y_PATH"])
    
    n_total = len(X_full)
    print(f"Full dataset: {n_total} images")
    
    # Shuffle indices
    indices = np.random.permutation(n_total)
    
    # Split 50/50
    n_half = n_total // 2
    train_val_indices = indices[:n_half]
    augment_test_indices = indices[n_half:]
    
    print(f"\n📌 EXPERIMENT DESIGN:")
    print(f"   First 50% ({len(train_val_indices)} images) → Training + Validation")
    print(f"   Second 50% ({len(augment_test_indices)} images) → Augmented Test Pool")
    
    # First half: Train/Val split
    X_train_val = X_full[train_val_indices]
    y_train_val = y_full[train_val_indices]
    
    # Split first half into train (70%) and val (30%)
    n_train_val = len(X_train_val)
    n_train = int(0.70 * n_train_val)
    
    shuffle_idx = np.random.permutation(n_train_val)
    train_idx = shuffle_idx[:n_train]
    val_idx = shuffle_idx[n_train:]
    
    X_train = X_train_val[train_idx]
    y_train = y_train_val[train_idx]
    X_val = X_train_val[val_idx]
    y_val = y_train_val[val_idx]
    
    print(f"\n   Training set: {len(X_train)} images (70% of first half)")
    print(f"   Validation set: {len(X_val)} images (30% of first half)")
    
    # Second half: Sample for augmented test set
    X_aug_pool = X_full[augment_test_indices]
    y_aug_pool = y_full[augment_test_indices]
    
    # Take 15% of the second half as test
    n_test = int(CONFIG["TEST_FRACTION_FROM_AUG"] * len(X_aug_pool))
    test_indices = np.random.permutation(len(X_aug_pool))[:n_test]
    
    X_test = X_aug_pool[test_indices]
    y_test = y_aug_pool[test_indices]
    
    print(f"   Test set (from augmented pool): {len(X_test)} images (15% of second half)")
    print(f"   Test images will be augmented at evaluation time")
    
    # Normalize GPS using ONLY training data
    min_val = y_train.min(axis=0)
    max_val = y_train.max(axis=0)
    
    y_train_norm = (y_train - min_val) / (max_val - min_val)
    y_val_norm = (y_val - min_val) / (max_val - min_val)
    y_test_norm = (y_test - min_val) / (max_val - min_val)
    
    print(f"\n   GPS normalization (from training set only):")
    print(f"     Lat: [{min_val[0]:.6f}, {max_val[0]:.6f}]")
    print(f"     Lon: [{min_val[1]:.6f}, {max_val[1]:.6f}]")
    
    # Preprocess images for train/val (no augmentation)
    X_train_proc = np.transpose(X_train, (0, 3, 1, 2)).astype(np.float32) / 255.0
    X_val_proc = np.transpose(X_val, (0, 3, 1, 2)).astype(np.float32) / 255.0
    
    # Create train/val dataloaders (standard TensorDataset)
    train_dataset = TensorDataset(
        torch.tensor(X_train_proc),
        torch.tensor(y_train_norm, dtype=torch.float32)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val_proc),
        torch.tensor(y_val_norm, dtype=torch.float32)
    )
    
    # Create test dataset with augmentation
    test_dataset = AugmentedTestDataset(
        X_test,
        y_test_norm.astype(np.float32),
        transform=get_test_augmentation()
    )
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=False)
    
    # Save normalization parameters
    np.save(CONFIG["OUTPUT_DIR"] / "min_val.npy", min_val)
    np.save(CONFIG["OUTPUT_DIR"] / "max_val.npy", max_val)
    
    # Store original test data for analysis
    data_info = {
        "n_total": n_total,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "train_fraction": 0.5 * 0.7,
        "val_fraction": 0.5 * 0.3,
        "test_fraction": 0.5 * CONFIG["TEST_FRACTION_FROM_AUG"],
        "test_is_augmented": True,
        "min_lat": float(min_val[0]),
        "max_lat": float(max_val[0]),
        "min_lon": float(min_val[1]),
        "max_lon": float(max_val[1]),
    }
    
    print(f"\n✓ Data loading complete")
    print("=" * 60)
    
    return train_loader, val_loader, test_loader, min_val, max_val, data_info, y_test


def train_model(model_name, model_class, train_loader, val_loader, test_loader, 
                min_val, max_val, y_test_original):
    """Train a single model and return results."""
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    
    output_dir = CONFIG["OUTPUT_DIR"] / model_name
    
    # Initialize model
    model = model_class().to(CONFIG["DEVICE"])
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["LR"], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Loss functions
    train_criterion = nn.L1Loss()
    eval_criterion = HaversineLoss(
        torch.tensor(min_val, device=CONFIG["DEVICE"]),
        torch.tensor(max_val, device=CONFIG["DEVICE"]),
        CONFIG["DEVICE"]
    )
    
    # Training tracking
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    history = {"train_loss": [], "val_loss": [], "lr": []}
    
    # Training loop
    for epoch in range(CONFIG["EPOCHS"]):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['EPOCHS']}", leave=False)
        for images, labels in pbar:
            images, labels = images.to(CONFIG["DEVICE"]), labels.to(CONFIG["DEVICE"])
            
            optimizer.zero_grad()
            preds = model(images)
            loss = train_criterion(preds, labels)
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(CONFIG["DEVICE"]), labels.to(CONFIG["DEVICE"])
                preds = model(images)
                loss = eval_criterion(preds, labels)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        current_lr = optimizer.param_groups[0]['lr']
        
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["lr"].append(current_lr)
        
        print(f"Epoch {epoch+1}/{CONFIG['EPOCHS']}: Train Loss={avg_train_loss:.4f}, "
              f"Val Error={avg_val_loss:.2f}m, LR={current_lr:.6f}")
        
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  ✓ New best model (Val Error: {avg_val_loss:.2f}m)")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG["PATIENCE"]:
                print(f"\n⚠ Early stopping at epoch {epoch+1}")
                break
    
    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    
    # Save model
    model_path = output_dir / f"{model_name}_30ep_augtest.pth"
    torch.save(model.state_dict(), model_path)
    print(f"✓ Saved {model_path}")
    
    # Evaluate on augmented test set
    print("\nEvaluating on augmented test set...")
    model.eval()
    all_errors = []
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(CONFIG["DEVICE"]), labels.to(CONFIG["DEVICE"])
            preds = model(images)
            
            # Denormalize
            preds_real = preds.cpu().numpy() * (max_val - min_val) + min_val
            labels_real = labels.cpu().numpy() * (max_val - min_val) + min_val
            
            for i in range(len(preds_real)):
                error = haversine_distance(
                    preds_real[i, 0], preds_real[i, 1],
                    labels_real[i, 0], labels_real[i, 1]
                )
                all_errors.append(error)
                all_preds.append(preds_real[i])
                all_labels.append(labels_real[i])
    
    errors = np.array(all_errors)
    preds_arr = np.vstack(all_preds)
    labels_arr = np.vstack(all_labels)
    
    # Calculate metrics
    results = {
        "model": model_name,
        "epochs": CONFIG["EPOCHS"],
        "lr": CONFIG["LR"],
        "data_fraction": CONFIG["DATA_FRACTION"],
        "test_augmented": True,
        "mean_error": float(np.mean(errors)),
        "median_error": float(np.median(errors)),
        "std_error": float(np.std(errors)),
        "min_error": float(np.min(errors)),
        "max_error": float(np.max(errors)),
        "within_5m": float(np.mean(errors < 5) * 100),
        "within_10m": float(np.mean(errors < 10) * 100),
        "within_15m": float(np.mean(errors < 15) * 100),
        "within_20m": float(np.mean(errors < 20) * 100),
        "within_30m": float(np.mean(errors < 30) * 100),
        "within_50m": float(np.mean(errors < 50) * 100),
        "p75_error": float(np.percentile(errors, 75)),
        "p90_error": float(np.percentile(errors, 90)),
        "p95_error": float(np.percentile(errors, 95)),
        "p99_error": float(np.percentile(errors, 99)),
        "test_samples": len(errors),
        "train_samples": len(train_loader.dataset),
        "val_samples": len(val_loader.dataset),
        "best_val_loss": float(best_val_loss),
        "final_epoch": len(history["train_loss"]),
        "timestamp": datetime.now().isoformat(),
    }
    
    print(f"\n{'='*50}")
    print(f"{model_name} Test Results (Augmented Test Set):")
    print(f"{'='*50}")
    print(f"  Mean Error:   {results['mean_error']:.2f}m")
    print(f"  Median Error: {results['median_error']:.2f}m")
    print(f"  Std Error:    {results['std_error']:.2f}m")
    print(f"  Within 5m:    {results['within_5m']:.1f}%")
    print(f"  Within 10m:   {results['within_10m']:.1f}%")
    print(f"  Within 20m:   {results['within_20m']:.1f}%")
    print(f"  P95 Error:    {results['p95_error']:.2f}m")
    print(f"  Max Error:    {results['max_error']:.2f}m")
    
    # Save results
    np.save(output_dir / f"{model_name.lower()}_errors.npy", errors)
    np.save(output_dir / f"{model_name.lower()}_predictions.npy", preds_arr)
    np.save(output_dir / f"{model_name.lower()}_labels.npy", labels_arr)
    
    with open(output_dir / "test_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save training history
    with open(output_dir / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    return results, history, errors, preds_arr, labels_arr


def write_experiment_log(data_info, all_results):
    """Write detailed experiment log."""
    log_path = CONFIG["OUTPUT_DIR"] / "experiment_log.txt"
    
    with open(log_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("EXPERIMENT LOG\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("EXPERIMENT: 30 Epochs, LR=0.0001, Half Data Training, Augmented Test\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("-" * 70 + "\n")
        f.write("DATA SPLIT STRATEGY:\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total Dataset: {data_info['n_total']} images\n\n")
        
        f.write("FIRST 50% ({} images) → Training + Validation:\n".format(
            data_info['n_train'] + data_info['n_val']))
        f.write(f"  • Training set: {data_info['n_train']} images (70%)\n")
        f.write(f"  • Validation set: {data_info['n_val']} images (30%)\n\n")
        
        f.write("SECOND 50% ({} images) → Augmented Test Pool:\n".format(
            int(data_info['n_total'] * 0.5)))
        f.write(f"  • Test set: {data_info['n_test']} images (15% of second half)\n")
        f.write(f"  • Augmentations applied: ColorJitter, RandomHorizontalFlip,\n")
        f.write(f"    RandomRotation(±10°), RandomAffine\n\n")
        
        f.write("-" * 70 + "\n")
        f.write("TRAINING CONFIGURATION:\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Epochs: {CONFIG['EPOCHS']}\n")
        f.write(f"  Learning Rate: {CONFIG['LR']}\n")
        f.write(f"  Batch Size: {CONFIG['BATCH_SIZE']}\n")
        f.write(f"  Optimizer: AdamW (weight_decay=1e-4)\n")
        f.write(f"  Scheduler: ReduceLROnPlateau (patience=5, factor=0.5)\n")
        f.write(f"  Early Stopping Patience: {CONFIG['PATIENCE']}\n")
        f.write(f"  Random Seed: {CONFIG['SEED']}\n\n")
        
        f.write("-" * 70 + "\n")
        f.write("GPS NORMALIZATION (from training set only):\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Latitude range:  [{data_info['min_lat']:.6f}, {data_info['max_lat']:.6f}]\n")
        f.write(f"  Longitude range: [{data_info['min_lon']:.6f}, {data_info['max_lon']:.6f}]\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("RESULTS (on Augmented Test Set):\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"{'Model':<15} {'Mean(m)':<10} {'Median(m)':<10} {'<10m':<10} {'<20m':<10} {'Max(m)':<10}\n")
        f.write("-" * 65 + "\n")
        
        for model, res in all_results.items():
            f.write(f"{model:<15} {res['mean_error']:<10.2f} {res['median_error']:<10.2f} "
                    f"{res['within_10m']:<10.1f} {res['within_20m']:<10.1f} {res['max_error']:<10.2f}\n")
        
        # Find best model
        best_model = min(all_results.items(), key=lambda x: x[1]['mean_error'])
        f.write(f"\n🏆 Best Model: {best_model[0]} (Mean Error: {best_model[1]['mean_error']:.2f}m)\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("NOTES:\n")
        f.write("=" * 70 + "\n")
        f.write("• Test set uses UNSEEN images (second half of dataset)\n")
        f.write("• Test images are augmented at evaluation time\n")
        f.write("• This creates a harder evaluation than same-distribution testing\n")
        f.write("• Results reflect model's ability to generalize to:\n")
        f.write("  - New locations (unseen in training)\n")
        f.write("  - Visual variations (augmentation)\n")
    
    print(f"✓ Experiment log saved: {log_path}")


def main():
    """Main training pipeline."""
    print("=" * 70)
    print("EXPERIMENT: 30 Epochs, LR=0.0001, Half Data, Augmented Test Set")
    print("=" * 70)
    print(f"Device: {CONFIG['DEVICE']}")
    print(f"Output: {CONFIG['OUTPUT_DIR']}")
    print()
    
    # Setup directories
    setup_directories()
    
    # Load and split data
    train_loader, val_loader, test_loader, min_val, max_val, data_info, y_test = load_and_split_data()
    
    # Train all models
    all_results = {}
    all_histories = {}
    all_errors = {}
    all_predictions = {}
    all_labels = {}
    
    for model_name, model_class in MODELS.items():
        results, history, errors, preds, labels = train_model(
            model_name, model_class,
            train_loader, val_loader, test_loader,
            min_val, max_val, y_test
        )
        all_results[model_name] = results
        all_histories[model_name] = history
        all_errors[model_name] = errors
        all_predictions[model_name] = preds
        all_labels[model_name] = labels
    
    # Write experiment log
    write_experiment_log(data_info, all_results)
    
    # Save combined results
    combined_results = {
        "experiment": "30ep_lr0001_halfdata_augtest",
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in CONFIG.items() 
                   if k != "DEVICE"},
        "data_info": data_info,
        "results": all_results,
    }
    
    with open(CONFIG["OUTPUT_DIR"] / "combined_results.json", 'w') as f:
        json.dump(combined_results, f, indent=2)
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\nResults Summary (Augmented Test Set):")
    print(f"{'Model':<15} {'Mean Error':<12} {'Within 10m':<12} {'Within 20m'}")
    print("-" * 50)
    for model, res in all_results.items():
        print(f"{model:<15} {res['mean_error']:.2f}m{'':<6} {res['within_10m']:.1f}%{'':<6} {res['within_20m']:.1f}%")
    
    best_model = min(all_results.items(), key=lambda x: x[1]['mean_error'])
    print(f"\n🏆 Best: {best_model[0]} (Mean: {best_model[1]['mean_error']:.2f}m)")
    print(f"\n📁 Output: {CONFIG['OUTPUT_DIR']}")
    print("\nNext step: Run comparison and plotting scripts")


if __name__ == "__main__":
    main()
