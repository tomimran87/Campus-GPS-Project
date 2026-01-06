"""
Evaluation Script for GPS Localization Models

This script loads trained models and evaluates them on the test set with
comprehensive metrics. Use this for final model evaluation after training.

Usage:
    python evaluate.py

Output:
    - Detailed metrics for each model
    - Ensemble performance
    - Comparison table
"""

import torch
import numpy as np
from data_loader import GPSDataManager
from loss import HaversineLoss
from models import ResNetGPS, EfficientNetGPS, ConvNextGPS
from metrics import GPSMetrics


# Configuration
CONFIG = {
    "X_PATH": "/home/liranatt/project/claude_exmpirement/X_gps.npy",
    "Y_PATH": "/home/liranatt/project/claude_exmpirement/y_gps.npy",
    "BATCH_SIZE": 32,
    "MODEL_PATHS": {
        "ResNet18": "/home/liranatt/project/ResNet18_gps.pth",
        "EfficientNet": "/home/liranatt/project/EfficientNet_gps.pth",
        "ConvNeXt": "/home/liranatt/project/ConvNeXt_gps.pth"
    }
}


def evaluate_model(model, loader, min_val, max_val, device):
    """
    Evaluate a single model on a dataset
    
    Args:
        model: Trained model
        loader: Data loader
        min_val, max_val: Normalization parameters
        device: torch.device
        
    Returns:
        dict: Metrics dictionary
    """
    model.eval()
    all_distances = []
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            
            # Get predictions
            preds = model(images)
            
            # Compute distances
            distances = GPSMetrics.compute_haversine_distances(
                preds, labels, min_val, max_val
            )
            all_distances.extend(distances)
    
    # Convert to array and compute metrics
    all_distances = np.array(all_distances)
    return GPSMetrics.compute_all_metrics(all_distances)


def evaluate_ensemble(models, loader, min_val, max_val, device):
    """
    Evaluate ensemble of models
    
    Args:
        models: List of trained models
        loader: Data loader
        min_val, max_val: Normalization parameters
        device: torch.device
        
    Returns:
        dict: Metrics dictionary
    """
    for m in models:
        m.eval()
    
    all_distances = []
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            
            # Get predictions from all models
            batch_preds = [m(images) for m in models]
            
            # Average predictions
            avg_pred = torch.stack(batch_preds).mean(dim=0)
            
            # Compute distances
            distances = GPSMetrics.compute_haversine_distances(
                avg_pred, labels, min_val, max_val
            )
            all_distances.extend(distances)
    
    # Convert to array and compute metrics
    all_distances = np.array(all_distances)
    return GPSMetrics.compute_all_metrics(all_distances)


def main():
    """Main evaluation pipeline"""
    print(f"\n{'='*60}")
    print(f"GPS Localization Model Evaluation")
    print(f"{'='*60}\n")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    # Load data
    print("Loading data...")
    data_manager = GPSDataManager(CONFIG["X_PATH"], CONFIG["Y_PATH"], CONFIG["BATCH_SIZE"])
    train_loader, val_loader, test_loader = data_manager.get_loaders()
    min_t, max_t = data_manager.get_scaling_tensors()
    min_t, max_t = min_t.to(device), max_t.to(device)
    
    # Load models
    print("Loading trained models...\n")
    models = {}
    model_classes = {
        "ResNet18": ResNetGPS,
        "EfficientNet": EfficientNetGPS,
        "ConvNeXt": ConvNextGPS
    }
    
    for name, model_class in model_classes.items():
        try:
            model = model_class().to(device)
            model.load_state_dict(torch.load(CONFIG["MODEL_PATHS"][name]))
            models[name] = model
            print(f"✓ Loaded {name}")
        except FileNotFoundError:
            print(f"✗ Model file not found: {CONFIG['MODEL_PATHS'][name]}")
    
    if not models:
        print("\n❌ No models found. Please train models first using main.py")
        return
    
    # Evaluate each model
    print(f"\n{'='*60}")
    print("INDIVIDUAL MODEL EVALUATION")
    print(f"{'='*60}")
    
    results = {}
    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        metrics = evaluate_model(model, test_loader, min_t, max_t, device)
        results[name] = metrics
        GPSMetrics.print_metrics(metrics, title=f"{name} - Test Set")
    
    # Evaluate ensemble
    if len(models) > 1:
        print(f"\n{'='*60}")
        print("ENSEMBLE EVALUATION")
        print(f"{'='*60}")
        
        ensemble_metrics = evaluate_ensemble(
            list(models.values()), test_loader, min_t, max_t, device
        )
        results["Ensemble"] = ensemble_metrics
        GPSMetrics.print_metrics(ensemble_metrics, title="Ensemble - Test Set")
    
    # Comparison table
    print(f"\n{'='*60}")
    print("MODEL COMPARISON (Test Set)")
    print(f"{'='*60}")
    print(f"{'Model':<15} {'Mean (m)':>10} {'Median (m)':>12} {'95th %ile':>12} {'Acc@10m':>10}")
    print(f"-" * 60)
    
    for name, metrics in results.items():
        print(f"{name:<15} {metrics['mean_error']:>10.2f} {metrics['median_error']:>12.2f} "
              f"{metrics['p95_error']:>12.2f} {metrics['accuracy_10m']:>9.1f}%")
    
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
