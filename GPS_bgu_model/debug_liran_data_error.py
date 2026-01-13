"""
Debug Script for Liran Data Error with EfficientNet Model

This script investigates issues when loading EfficientNet_gps.pth with liran_data
"""

import torch
import numpy as np
import os
import traceback
from pathlib import Path

# Import project modules
from models import EfficientNetGPS
from data_loader import GPSDataManager

def check_liran_data_structure():
    """Check the structure and contents of liran_data directory"""
    print("=== LIRAN DATA DIRECTORY ANALYSIS ===")
    liran_path = Path("/home/tommimra/GPS_BGU_model/liran_data")
    
    if not liran_path.exists():
        print(f"❌ ERROR: {liran_path} does not exist!")
        return False
    
    print(f"✅ Directory exists: {liran_path}")
    print(f"Directory contents:")
    
    for item in liran_path.iterdir():
        if item.is_file():
            size = item.stat().st_size / (1024*1024)  # Size in MB
            print(f"  📄 {item.name} ({size:.2f} MB)")
        elif item.is_dir():
            num_files = len(list(item.iterdir()))
            print(f"  📁 {item.name}/ ({num_files} items)")
    
    return True

def load_efficientnet_model():
    """Load EfficientNet model and check for issues"""
    print("\n=== EFFICIENTNET MODEL LOADING ===")
    
    model_path = "/home/tommimra/GPS_BGU_model/GPS_bgu_model/EfficientNet_gps.pth"
    
    if not os.path.exists(model_path):
        print(f"❌ ERROR: Model file not found: {model_path}")
        return None
    
    try:
        # Create model instance
        print("Creating EfficientNetGPS model...")
        model = EfficientNetGPS()
        
        # Load weights
        print("Loading model weights...")
        checkpoint = torch.load(model_path, map_location='cpu')
        
        print(f"Checkpoint keys: {list(checkpoint.keys())}")
        
        # Check if it's a full checkpoint or just state_dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("✅ Loaded from 'model_state_dict'")
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
            print("✅ Loaded from 'state_dict'")
        else:
            model.load_state_dict(checkpoint)
            print("✅ Loaded directly as state_dict")
        
        model.eval()
        print("✅ EfficientNet model loaded successfully")
        return model
        
    except Exception as e:
        print(f"❌ ERROR loading EfficientNet model: {e}")
        traceback.print_exc()
        return None

def test_liran_data_loading():
    """Test loading data from liran_data directory"""
    print("\n=== LIRAN DATA LOADING TEST ===")
    
    try:
        # Look for X_gps.npy and y_gps.npy in liran_data
        liran_path = "/home/tommimra/GPS_BGU_model/liran_data"
        
        x_path = os.path.join(liran_path, "X_gps.npy")
        y_path = os.path.join(liran_path, "y_gps.npy")
        
        print(f"Looking for X data: {x_path}")
        print(f"Looking for Y data: {y_path}")
        
        if os.path.exists(x_path):
            X_data = np.load(x_path)
            print(f"✅ X data loaded: shape {X_data.shape}, dtype {X_data.dtype}")
        else:
            print(f"❌ X data not found at {x_path}")
            
        if os.path.exists(y_path):
            y_data = np.load(y_path)
            print(f"✅ Y data loaded: shape {y_data.shape}, dtype {y_data.dtype}")
        else:
            print(f"❌ Y data not found at {y_path}")
            
        # Try using GPSDataManager
        print("\nTesting GPSDataManager with liran_data...")
        data_manager = GPSDataManager(
            x_path=x_path if os.path.exists(x_path) else None,
            y_path=y_path if os.path.exists(y_path) else None,
            batch_size=32,
            test_size=0.15,
            val_size=0.15
        )
        
        if hasattr(data_manager, 'get_loaders'):
            train_loader, val_loader, test_loader = data_manager.get_loaders()
            print("✅ Data loaders created successfully")
            
            # Test first batch
            for batch_idx, (images, targets) in enumerate(train_loader):
                print(f"✅ First batch loaded: images {images.shape}, targets {targets.shape}")
                break
        
    except Exception as e:
        print(f"❌ ERROR with liran data loading: {e}")
        traceback.print_exc()

def test_model_with_liran_data():
    """Test running EfficientNet model with liran data"""
    print("\n=== MODEL + LIRAN DATA INTEGRATION TEST ===")
    
    try:
        # Load model
        model = load_efficientnet_model()
        if model is None:
            print("❌ Cannot proceed - model loading failed")
            return
            
        # Load data
        liran_path = "/home/tommimra/GPS_BGU_model/liran_data"
        x_path = os.path.join(liran_path, "X_gps.npy")
        y_path = os.path.join(liran_path, "y_gps.npy")
        
        if not (os.path.exists(x_path) and os.path.exists(y_path)):
            print("❌ Cannot proceed - data files not found")
            return
            
        # Create data manager
        data_manager = GPSDataManager(x_path, y_path, batch_size=8, test_size=0.15, val_size=0.15)
        train_loader, val_loader, test_loader = data_manager.get_loaders()
        
        # Test inference
        print("Testing model inference...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(val_loader):
                images = images.to(device)
                targets = targets.to(device)
                
                print(f"Input shape: {images.shape}")
                print(f"Target shape: {targets.shape}")
                
                # Forward pass
                predictions = model(images)
                print(f"✅ Predictions shape: {predictions.shape}")
                print(f"Sample prediction: {predictions[0].cpu().numpy()}")
                print(f"Sample target: {targets[0].cpu().numpy()}")
                break
                
        print("✅ Model inference with liran data successful!")
        
    except Exception as e:
        print(f"❌ ERROR during model+data integration: {e}")
        traceback.print_exc()

def main():
    """Main debug function"""
    print("🔍 DEBUGGING LIRAN DATA ERROR WITH EFFICIENTNET")
    print("=" * 60)
    
    # Step 1: Check data structure
    if not check_liran_data_structure():
        return
    
    # Step 2: Test model loading
    load_efficientnet_model()
    
    # Step 3: Test data loading
    test_liran_data_loading()
    
    # Step 4: Test integration
    test_model_with_liran_data()
    
    print("\n🏁 DEBUG COMPLETE")

if __name__ == "__main__":
    main()