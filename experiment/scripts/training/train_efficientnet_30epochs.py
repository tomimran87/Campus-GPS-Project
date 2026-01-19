"""
Train EfficientNet with same settings as ResNet18 and ConvNeXt
for fair comparison (30 epochs, LR=0.001)

Output saved to: testing/30epochs/EfficientNet/
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from pathlib import Path
import json
from datetime import datetime

from data_loader import GPSDataManager
from models import EfficientNetGPS
from loss import HaversineLoss


CONFIG = {
    "X_PATH": "/home/liranatt/project/main_project/Latest_data/Latest_data/X.npy",
    "Y_PATH": "/home/liranatt/project/main_project/Latest_data/Latest_data/y.npy",
    "BATCH_SIZE": 32,
    "EPOCHS": 30,
    "LR": 0.001,
    "PATIENCE": 7,
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "OUTPUT_DIR": Path("testing/30epochs/EfficientNet"),
}


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return R * c


def setup_output_dir():
    """Create output directory structure."""
    CONFIG["OUTPUT_DIR"].mkdir(parents=True, exist_ok=True)
    return CONFIG["OUTPUT_DIR"]


def main():
    output_dir = setup_output_dir()
    
    print("=" * 60)
    print("TRAINING EfficientNet (30 Epochs, LR=0.001)")
    print("=" * 60)
    print(f"Device: {CONFIG['DEVICE']}")
    print(f"Epochs: {CONFIG['EPOCHS']}, LR: {CONFIG['LR']}")
    print(f"Output: {output_dir}")
    print()
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load data
    print("Loading data...")
    data_manager = GPSDataManager(CONFIG["X_PATH"], CONFIG["Y_PATH"], CONFIG["BATCH_SIZE"])
    train_loader, val_loader, test_loader = data_manager.get_loaders()
    min_val, max_val = data_manager.min_val, data_manager.max_val
    
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
    
    # Model setup
    model = EfficientNetGPS().to(CONFIG["DEVICE"])
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
    
    # Training history
    history = {"train_loss": [], "val_loss": [], "lr": []}
    
    print("\n" + "="*50)
    print("Training EfficientNet")
    print("="*50)
    
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
    model_path = output_dir / "EfficientNet_30epochs_gps.pth"
    torch.save(model.state_dict(), model_path)
    print(f"✓ Saved {model_path}")
    
    # Also save to main directory for compatibility
    torch.save(model.state_dict(), "EfficientNet_30epochs_gps.pth")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    model.eval()
    all_errors = []
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
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
        "model": "EfficientNet",
        "epochs": CONFIG["EPOCHS"],
        "lr": CONFIG["LR"],
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
        "best_val_loss": float(best_val_loss),
        "final_epoch": len(history["train_loss"]),
        "timestamp": datetime.now().isoformat(),
    }
    
    print(f"\n{'='*50}")
    print(f"EfficientNet Test Results:")
    print(f"{'='*50}")
    print(f"  Mean Error:   {results['mean_error']:.2f}m")
    print(f"  Median Error: {results['median_error']:.2f}m")
    print(f"  Std Error:    {results['std_error']:.2f}m")
    print(f"  Within 5m:    {results['within_5m']:.1f}%")
    print(f"  Within 10m:   {results['within_10m']:.1f}%")
    print(f"  Within 20m:   {results['within_20m']:.1f}%")
    print(f"  Within 50m:   {results['within_50m']:.1f}%")
    print(f"  Max Error:    {results['max_error']:.2f}m")
    
    # Save arrays to output directory
    np.save(output_dir / "efficientnet_errors.npy", errors)
    np.save(output_dir / "efficientnet_predictions.npy", preds)
    np.save(output_dir / "efficientnet_labels.npy", labels_arr)
    
    # Also save to main directory for compatibility with generate_comparison_reports.py
    np.save("efficientnet_errors.npy", errors)
    np.save("efficientnet_predictions.npy", preds)
    np.save("efficientnet_labels.npy", labels_arr)
    
    # Save history
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    # Save results
    with open(output_dir / "test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Saved efficientnet_errors.npy, efficientnet_predictions.npy, efficientnet_labels.npy")
    print(f"✓ Saved training_history.json, test_results.json")
    print(f"✓ All outputs saved to: {output_dir}")
    print("✓ Training complete!")
    
    return results


if __name__ == "__main__":
    main()
