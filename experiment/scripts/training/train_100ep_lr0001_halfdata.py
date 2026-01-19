#!/usr/bin/env python3
"""
Experiment: 100 Epochs, LR=0.0001, Half Data (50%)
===================================================

Training all 3 models with:
- 100 epochs (extended training)
- LR=0.0001 (optimal for ConvNeXt)
- Half data (50%)
- Standard 70/15/15 train/val/test split

Output: testing/100ep_lr0001_halfdata/
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from pathlib import Path
import json
from datetime import datetime

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
    "EPOCHS": 100,
    "LR": 0.0001,
    "DATA_FRACTION": 0.5,
    "PATIENCE": 15,
    "SEED": 42,
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "OUTPUT_DIR": Path("testing/100ep_lr0001_halfdata"),
}

MODELS = {
    "ResNet18": ResNetGPS,
    "EfficientNet": EfficientNetGPS,
    "ConvNeXt": ConvNextGPS,
}


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
    
    print(f"✓ Created directory structure: {CONFIG['OUTPUT_DIR']}")


def load_and_split_data():
    """Load half of the data and split into train/val/test."""
    print("\n" + "=" * 60)
    print("DATA LOADING")
    print("=" * 60)
    
    np.random.seed(CONFIG["SEED"])
    torch.manual_seed(CONFIG["SEED"])
    
    # Load full data
    X = np.load(CONFIG["X_PATH"])
    y = np.load(CONFIG["Y_PATH"])
    
    n_full = len(X)
    print(f"Full dataset: {n_full} images")
    
    # Shuffle first
    indices = np.random.permutation(n_full)
    X = X[indices]
    y = y[indices]
    
    # Take only 50%
    n_half = int(n_full * CONFIG["DATA_FRACTION"])
    X = X[:n_half]
    y = y[:n_half]
    
    n_total = len(X)
    print(f"Using: {n_total} images ({CONFIG['DATA_FRACTION']*100:.0f}%)")
    
    # Split: 70% train, 15% val, 15% test
    n_train = int(0.70 * n_total)
    n_val = int(0.15 * n_total)
    
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]
    
    print(f"Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    # Normalize GPS using ONLY training data
    min_val = y_train.min(axis=0)
    max_val = y_train.max(axis=0)
    
    y_train_norm = (y_train - min_val) / (max_val - min_val)
    y_val_norm = (y_val - min_val) / (max_val - min_val)
    y_test_norm = (y_test - min_val) / (max_val - min_val)
    
    print(f"GPS range: lat [{min_val[0]:.6f}, {max_val[0]:.6f}], lon [{min_val[1]:.6f}, {max_val[1]:.6f}]")
    
    # Preprocess images
    X_train = np.transpose(X_train, (0, 3, 1, 2)).astype(np.float32) / 255.0
    X_val = np.transpose(X_val, (0, 3, 1, 2)).astype(np.float32) / 255.0
    X_test = np.transpose(X_test, (0, 3, 1, 2)).astype(np.float32) / 255.0
    
    # Create dataloaders
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train_norm, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val_norm, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test_norm, dtype=torch.float32))
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=False)
    
    # Save normalization parameters
    np.save(CONFIG["OUTPUT_DIR"] / "min_val.npy", min_val)
    np.save(CONFIG["OUTPUT_DIR"] / "max_val.npy", max_val)
    
    data_info = {
        "n_full": n_full,
        "n_used": n_total,
        "data_fraction": CONFIG["DATA_FRACTION"],
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
    }
    
    print("✓ Data loading complete")
    print("=" * 60)
    
    return train_loader, val_loader, test_loader, min_val, max_val, data_info


def train_model(model_name, model_class, train_loader, val_loader, test_loader, min_val, max_val):
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
        optimizer, mode='min', factor=0.5, patience=7, verbose=True
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
        
        print(f"Epoch {epoch+1}/{CONFIG['EPOCHS']}: Train={avg_train_loss:.4f}, "
              f"Val={avg_val_loss:.2f}m, LR={current_lr:.6f}")
        
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
    model_path = output_dir / f"{model_name}_100ep.pth"
    torch.save(model.state_dict(), model_path)
    print(f"✓ Saved {model_path}")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    model.eval()
    all_errors = []
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(CONFIG["DEVICE"]), labels.to(CONFIG["DEVICE"])
            preds = model(images)
            
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
        "final_epoch": len(history["train_loss"]),
        "best_val_loss": float(best_val_loss),
    }
    
    print(f"\n📊 {model_name} Results:")
    print(f"   Mean Error: {results['mean_error']:.2f}m")
    print(f"   Median Error: {results['median_error']:.2f}m")
    print(f"   Within 10m: {results['within_10m']:.1f}%")
    print(f"   Within 20m: {results['within_20m']:.1f}%")
    
    # Save results
    np.save(output_dir / f"{model_name.lower()}_errors.npy", errors)
    np.save(output_dir / f"{model_name.lower()}_predictions.npy", preds_arr)
    np.save(output_dir / f"{model_name.lower()}_labels.npy", labels_arr)
    
    with open(output_dir / "test_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    with open(output_dir / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    return results, history


def main():
    print("=" * 70)
    print("EXPERIMENT: 100 Epochs, LR=0.0001, Half Data (50%)")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {CONFIG['DEVICE']}")
    
    setup_directories()
    
    # Load data
    train_loader, val_loader, test_loader, min_val, max_val, data_info = load_and_split_data()
    
    # Train all models
    all_results = {}
    all_histories = {}
    
    for model_name, model_class in MODELS.items():
        results, history = train_model(
            model_name, model_class,
            train_loader, val_loader, test_loader,
            min_val, max_val
        )
        all_results[model_name] = results
        all_histories[model_name] = history
    
    # Summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"{'Model':<15} {'Mean Error':<12} {'Median':<10} {'<10m':<10} {'Epochs':<8}")
    print("-" * 55)
    
    best_model = None
    best_error = float('inf')
    
    for model_name, results in all_results.items():
        print(f"{model_name:<15} {results['mean_error']:<12.2f} {results['median_error']:<10.2f} "
              f"{results['within_10m']:<10.1f} {results['final_epoch']:<8}")
        if results['mean_error'] < best_error:
            best_error = results['mean_error']
            best_model = model_name
    
    print("-" * 55)
    print(f"🏆 Best: {best_model} with {best_error:.2f}m mean error")
    
    # Save summary
    with open(CONFIG["OUTPUT_DIR"] / "experiment_summary.json", 'w') as f:
        json.dump({
            'experiment': '100ep_lr0001_halfdata',
            'config': {
                'epochs': CONFIG['EPOCHS'],
                'learning_rate': CONFIG['LR'],
                'data_fraction': CONFIG['DATA_FRACTION'],
                'batch_size': CONFIG['BATCH_SIZE'],
            },
            'data_info': data_info,
            'results': all_results,
            'best_model': best_model,
            'best_error': best_error,
        }, f, indent=2)
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output: {CONFIG['OUTPUT_DIR']}")


if __name__ == "__main__":
    main()
