import torch
import numpy as np
from data_loader import GPSDataManager
from loss import HaversineLoss
from models import ResNetGPS, EfficientNetGPS, ConvNextGPS
from trainer import Trainer
from metrics import GPSMetrics
import torch.nn as nn


# --- CONFIGURATION ---
# Hyperparameters for training the GPS localization ensemble
CONFIG = {
    "X_PATH": "/home/liranatt/project/claude_exmpirement/X_gps.npy",        # Path to image data (N, 224, 224, 3)
    "Y_PATH": "/home/liranatt/project/claude_exmpirement/y_gps.npy",        # Path to GPS coordinates (N, 2)
    "BATCH_SIZE": 32,             # Batch size for training
    "EPOCHS": 50,                 # Maximum number of epochs
    "LR": 0.003,                  # Learning rate: increased to 0.003 for faster convergence
    "ConvLR": 0.005               # Model needs stronger signal to learn GPS variance
                                  # Tanh activation can handle higher LR without instability
}

def run_pipeline():
    """
    Main training pipeline for GPS localization ensemble
    
    Steps:
        1. Setup device (GPU if available)
        2. Set random seeds for reproducibility
        3. Load and split data with proper normalization
        4. Initialize Haversine loss function
        5. Train each model in the ensemble
        6. Evaluate ensemble on test set
    """
    # Set random seeds for reproducibility
    # This ensures consistent results across multiple runs
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{'='*60}")
    print(f"GPS Image Localization Training Pipeline")
    print(f"{'='*60}")
    print(f"System Check: Running on {device}")
    print(f"Random Seed: 42 (for reproducibility)")
    print(f"{'='*60}\n")

    # 1. Data Preparation with proper train/val/test splitting
    data_manager = GPSDataManager(CONFIG["X_PATH"], CONFIG["Y_PATH"], CONFIG["BATCH_SIZE"])
    train_loader, val_loader, test_loader = data_manager.get_loaders()  # Now returns 3 loaders
    min_t, max_t = data_manager.get_scaling_tensors()

    # 2. Loss Function - Haversine distance in meters
    criterion = nn.L1Loss()  # HaversineLoss expects normalized coords

    # 3. Model Zoo - Ensemble of different CNN architectures
    models_to_train = {
        "ResNet18": ResNetGPS(),
        "EfficientNet": EfficientNetGPS()
    }

    trained_models = {}

    # 4. Training Loop - Train each model sequentially
    for name, model in models_to_train.items():
        print(f"\n{'='*60}")
        print(f"Training: {name}")
        print(f"{'='*60}")
        trainer = Trainer(model, train_loader, val_loader, criterion, device, CONFIG["LR"])
        history = trainer.fit(CONFIG["EPOCHS"])
        
        # Save model checkpoint
        torch.save(model.state_dict(), f"{name}_gps.pth")
        print(f"✓ Model saved to {name}_gps.pth")
        
        trained_models[name] = model

    # 5. Final Ensemble Evaluation on Test Set
    print(f"\n{'='*60}")
    print(f"FINAL EVALUATION ON TEST SET")
    print(f"{'='*60}")
    test_loss = evaluate_ensemble(list(trained_models.values()), test_loader, criterion, device)
    print(f"Ensemble Test Error: {test_loss:.2f} meters")
    print(f"{'='*60}\n")
    
    # 7. Show detailed predictions on last test sample
    show_last_sample_predictions(
        trained_models,
        test_loader,
        min_t,
        max_t,
        device
    )

def evaluate_ensemble(models, loader, loss_fn, device):
    """
    Evaluate ensemble of models on a dataset
    
    Combines predictions from multiple models by averaging their outputs.
    This ensemble approach reduces variance and typically improves accuracy
    compared to individual models.
    
    Process:
        1. Set all models to evaluation mode
        2. For each batch:
            - Get predictions from all models
            - Average the predictions (ensemble)
            - Compute Haversine loss
        3. Return average loss across all batches
    
    Args:
        models (list): List of trained models
        loader (DataLoader): Data loader (validation or test)
        loss_fn (HaversineLoss): Loss function for computing distance
        device (torch.device): Device for computation
        
    Returns:
        float: Average Haversine distance error in meters
    """
    total_loss = 0.0
    
    # Set all models to evaluation mode
    for m in models:
        m.eval()
    
    # Disable gradient computation for efficiency
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            
            # Get predictions from all models in the ensemble
            batch_preds = [m(images) for m in models]
            
            # Average predictions across models
            # torch.stack creates tensor of shape (num_models, batch_size, 2)
            # .mean(dim=0) averages across models → (batch_size, 2)
            avg_pred = torch.stack(batch_preds).mean(dim=0)
            
            # Compute Haversine distance between ensemble prediction and target
            total_loss += loss_fn(avg_pred, labels).item()
            
    # Return average loss across all batches
    return total_loss / len(loader)


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points on Earth
    
    Args:
        lat1, lon1: First point coordinates in degrees
        lat2, lon2: Second point coordinates in degrees
        
    Returns:
        float: Distance in meters
    """
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # Earth's radius in meters
    return 6371000 * c


def show_last_sample_predictions(models_dict, test_loader, min_val, max_val, device):
    """
    Show detailed predictions for the last test sample
    
    Displays:
        - Actual GPS coordinates
        - Each model's individual prediction
        - Ensemble (mean) prediction
        - Distance errors for each
    
    Args:
        models_dict (dict): Dictionary of model_name -> model
        test_loader (DataLoader): Test data loader
        min_val, max_val (torch.Tensor): Denormalization parameters
        device (torch.device): Device for computation
    """
    # Get the last batch from test loader
    last_batch = None
    for batch in test_loader:
        last_batch = batch
    
    images, labels = last_batch
    # Get last sample from the last batch
    last_image = images[-1:].to(device)  # Shape: (1, 3, 224, 224)
    last_label = labels[-1:].to(device)  # Shape: (1, 2)
    
    # Set all models to eval mode
    for model in models_dict.values():
        model.eval()
    
    # Get predictions from each model
    predictions = {}
    with torch.no_grad():
        for name, model in models_dict.items():
            pred_norm = model(last_image)
            predictions[name] = pred_norm
    
    # Compute ensemble (mean) prediction
    all_preds = torch.stack(list(predictions.values()))  # Shape: (3, 1, 2)
    ensemble_pred = all_preds.mean(dim=0)  # Shape: (1, 2)
    
    # Denormalize everything
    min_val_cpu = min_val.cpu()
    max_val_cpu = max_val.cpu()
    
    actual_norm = last_label.cpu()
    actual_gps = (actual_norm * (max_val_cpu - min_val_cpu) + min_val_cpu).numpy()[0]
    
    ensemble_norm = ensemble_pred.cpu()
    ensemble_gps = (ensemble_norm * (max_val_cpu - min_val_cpu) + min_val_cpu).numpy()[0]
    
    # Print header
    print("\n" + "="*60)
    print("DETAILED PREDICTION ON LAST TEST SAMPLE")
    print("="*60)
    print(f"\nActual GPS Location:")
    print(f"  Latitude:  {actual_gps[0]:.6f}°")
    print(f"  Longitude: {actual_gps[1]:.6f}°")
    print(f"\n" + "-"*60)
    
    # Show each model's prediction
    model_errors = {}
    for name, pred_norm in predictions.items():
        pred_gps = (pred_norm.cpu() * (max_val_cpu - min_val_cpu) + min_val_cpu).numpy()[0]
        
        # Calculate distance using haversine formula
        distance = haversine_distance(
            actual_gps[0], actual_gps[1],
            pred_gps[0], pred_gps[1]
        )
        
        model_errors[name] = distance
        
        print(f"\n{name} Prediction:")
        print(f"  Latitude:  {pred_gps[0]:.6f}° (Δ{abs(pred_gps[0] - actual_gps[0]):.6f}°)")
        print(f"  Longitude: {pred_gps[1]:.6f}° (Δ{abs(pred_gps[1] - actual_gps[1]):.6f}°)")
        print(f"  Error: {distance:.2f} meters")
    
    # Show ensemble prediction
    ensemble_distance = haversine_distance(
        actual_gps[0], actual_gps[1],
        ensemble_gps[0], ensemble_gps[1]
    )
    
    print(f"\n" + "-"*60)
    print(f"\nEnsemble (Mean of 3 models) Prediction:")
    print(f"  Latitude:  {ensemble_gps[0]:.6f}° (Δ{abs(ensemble_gps[0] - actual_gps[0]):.6f}°)")
    print(f"  Longitude: {ensemble_gps[1]:.6f}° (Δ{abs(ensemble_gps[1] - actual_gps[1]):.6f}°)")
    print(f"  Error: {ensemble_distance:.2f} meters")
    
    # Summary
    print(f"\n" + "-"*60)
    print("\nSummary:")
    print(f"  Best Individual Model: {min(model_errors, key=model_errors.get)} ({min(model_errors.values()):.2f}m)")
    print(f"  Worst Individual Model: {max(model_errors, key=model_errors.get)} ({max(model_errors.values()):.2f}m)")
    print(f"  Ensemble Error: {ensemble_distance:.2f}m")
    print("="*60 + "\n")


if __name__ == "__main__":
    run_pipeline()