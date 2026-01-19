"""
Train All Models: 30 Epochs, LR=0.0001, Half Data
==================================================
Trains ResNet18, EfficientNet, and ConvNeXt on half the dataset
with a lower learning rate for comparison.

Output: testing/30epochs_lr0.0001_halfdata/
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from pathlib import Path
import json
from datetime import datetime

import sys
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
    "EPOCHS": 30,
    "LR": 0.0001,  # Lower learning rate
    "DATA_FRACTION": 0.5,  # Half the data
    "PATIENCE": 7,
    "SEED": 42,  # Random seed for reproducibility
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "OUTPUT_DIR": Path("testing/30epochs_lr0.0001_halfdata"),
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
    """Load data, sample half, and split into train/val/test."""
    print("\nLoading and preparing data...")
    
    # Set random seed for reproducibility
    np.random.seed(CONFIG["SEED"])
    torch.manual_seed(CONFIG["SEED"])
    
    # Load full data
    X_full = np.load(CONFIG["X_PATH"])
    y_full = np.load(CONFIG["Y_PATH"])
    
    print(f"  Full dataset: {len(X_full)} images")
    
    # Sample half the data
    n_samples = int(len(X_full) * CONFIG["DATA_FRACTION"])
    indices = np.random.permutation(len(X_full))[:n_samples]
    X = X_full[indices]
    y = y_full[indices]
    
    print(f"  Sampled dataset: {len(X)} images ({CONFIG['DATA_FRACTION']*100:.0f}%)")
    
    # Shuffle again for split
    shuffle_idx = np.random.permutation(len(X))
    X = X[shuffle_idx]
    y = y[shuffle_idx]
    
    # Split: 70% train, 15% val, 15% test
    n = len(X)
    train_end = int(0.70 * n)
    val_end = int(0.85 * n)
    
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    
    print(f"  Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    # Normalize GPS coordinates using ONLY training data (avoid data leakage!)
    min_val = y_train.min(axis=0)
    max_val = y_train.max(axis=0)
    
    y_train_norm = (y_train - min_val) / (max_val - min_val)
    y_val_norm = (y_val - min_val) / (max_val - min_val)
    y_test_norm = (y_test - min_val) / (max_val - min_val)
    
    print(f"  GPS range: lat [{min_val[0]:.6f}, {max_val[0]:.6f}], lon [{min_val[1]:.6f}, {max_val[1]:.6f}]")
    
    # Transpose images from (N, H, W, C) to (N, C, H, W)
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
    
    return train_loader, val_loader, test_loader, min_val, max_val


def train_model(model_name, model_class, train_loader, val_loader, test_loader, min_val, max_val):
    """Train a single model and return results."""
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    
    output_dir = CONFIG["OUTPUT_DIR"] / model_name
    
    # Set seeds for reproducibility
    torch.manual_seed(CONFIG["SEED"])
    np.random.seed(CONFIG["SEED"])
    
    # Initialize model
    model = model_class().to(CONFIG["DEVICE"])
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["LR"], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    train_criterion = nn.L1Loss()
    eval_criterion = HaversineLoss(
        torch.tensor(min_val, device=CONFIG["DEVICE"]),
        torch.tensor(max_val, device=CONFIG["DEVICE"]),
        CONFIG["DEVICE"]
    )
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    history = {"train_loss": [], "val_loss": [], "lr": []}
    
    for epoch in range(CONFIG["EPOCHS"]):
        # Training
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['EPOCHS']}", leave=False)
        for images, labels in pbar:
            images, labels = images.to(CONFIG["DEVICE"]), labels.to(CONFIG["DEVICE"])
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = train_criterion(outputs, labels)
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(CONFIG["DEVICE"]), labels.to(CONFIG["DEVICE"])
                outputs = model(images)
                loss = eval_criterion(outputs, labels)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val_loss)
        
        # Save history
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["lr"].append(current_lr)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
            print(f"Epoch {epoch+1}: Train={avg_train_loss:.4f}, Val={avg_val_loss:.2f}m, LR={current_lr:.6f} ✓ (best)")
        else:
            patience_counter += 1
            print(f"Epoch {epoch+1}: Train={avg_train_loss:.4f}, Val={avg_val_loss:.2f}m, LR={current_lr:.6f} (patience {patience_counter}/{CONFIG['PATIENCE']})")
            if patience_counter >= CONFIG["PATIENCE"]:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    
    # Save model
    model_path = output_dir / f"{model_name}_30epochs_gps.pth"
    torch.save(model.state_dict(), model_path)
    print(f"✓ Saved {model_path}")
    
    # Evaluate on test set
    print("Evaluating on test set...")
    model.eval()
    all_errors = []
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(CONFIG["DEVICE"]), labels.to(CONFIG["DEVICE"])
            outputs = model(images)
            
            preds_real = outputs.cpu().numpy() * (max_val - min_val) + min_val
            labels_real = labels.cpu().numpy() * (max_val - min_val) + min_val
            
            all_preds.append(preds_real)
            all_labels.append(labels_real)
            
            for pred, label in zip(preds_real, labels_real):
                error = haversine_distance(pred[0], pred[1], label[0], label[1])
                all_errors.append(error)
    
    errors = np.array(all_errors)
    preds = np.vstack(all_preds)
    labels_arr = np.vstack(all_labels)
    
    # Results
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
        "within_20m": float(np.mean(errors < 20) * 100),
        "within_50m": float(np.mean(errors < 50) * 100),
        "test_samples": len(errors),
        "train_samples": len(train_loader.dataset),
        "best_val_loss": float(best_val_loss),
        "final_epoch": len(history["train_loss"]),
        "timestamp": datetime.now().isoformat(),
    }
    
    print(f"\n{model_name} Results:")
    print(f"  Mean Error:   {results['mean_error']:.2f}m")
    print(f"  Median Error: {results['median_error']:.2f}m")
    print(f"  Within 10m:   {results['within_10m']:.1f}%")
    print(f"  Within 20m:   {results['within_20m']:.1f}%")
    
    # Save arrays
    np.save(output_dir / f"{model_name.lower()}_errors.npy", errors)
    np.save(output_dir / f"{model_name.lower()}_predictions.npy", preds)
    np.save(output_dir / f"{model_name.lower()}_labels.npy", labels_arr)
    
    # Save history and results
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    with open(output_dir / "test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results, errors, preds, labels_arr


def main():
    print("=" * 70)
    print("TRAINING ALL MODELS")
    print(f"Epochs: {CONFIG['EPOCHS']}, LR: {CONFIG['LR']}, Data: {CONFIG['DATA_FRACTION']*100:.0f}%")
    print(f"Random Seed: {CONFIG['SEED']} (for reproducibility)")
    print(f"Device: {CONFIG['DEVICE']}")
    print("=" * 70)
    
    setup_directories()
    train_loader, val_loader, test_loader, min_val, max_val = load_and_split_data()
    
    all_results = {}
    
    for model_name, model_class in MODELS.items():
        results, errors, preds, labels = train_model(
            model_name, model_class,
            train_loader, val_loader, test_loader,
            min_val, max_val
        )
        all_results[model_name] = results
    
    # Save combined results
    with open(CONFIG["OUTPUT_DIR"] / "comparison" / "combined_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE - SUMMARY")
    print("=" * 70)
    print(f"{'Model':<15} {'Mean Error':<12} {'Median':<10} {'Within 10m':<12} {'Within 20m':<12}")
    print("-" * 60)
    for name, res in all_results.items():
        print(f"{name:<15} {res['mean_error']:.2f}m{'':<6} {res['median_error']:.2f}m{'':<4} {res['within_10m']:.1f}%{'':<7} {res['within_20m']:.1f}%")
    
    best_model = min(all_results.keys(), key=lambda x: all_results[x]["mean_error"])
    print(f"\n🏆 Best Model: {best_model} (Mean Error: {all_results[best_model]['mean_error']:.2f}m)")
    print(f"\n✓ Results saved to: {CONFIG['OUTPUT_DIR']}")
    print("✓ Run comparison scripts to generate plots!")


if __name__ == "__main__":
    main()
