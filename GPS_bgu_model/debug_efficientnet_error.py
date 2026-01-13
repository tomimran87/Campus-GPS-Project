#!/usr/bin/env python3
"""
Debug Script for EfficientNet_gps.pth Model Loading Issues

This script diagnoses and displays errors related to the EfficientNet model,
particularly issues with 'liran_data' paths or model loading problems.
"""

import torch
import traceback
import os
from pathlib import Path

def check_model_file():
    """Check if EfficientNet model file exists and get basic info"""
    model_path = "/home/tommimra/GPS_BGU_model/GPS_bgu_model/EfficientNet_gps.pth"
    
    print("=" * 60)
    print("EFFICIENTNET MODEL FILE DIAGNOSTICS")
    print("=" * 60)
    
    if os.path.exists(model_path):
        file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        print(f"✓ Model file exists: {model_path}")
        print(f"✓ File size: {file_size:.1f} MB")
        return True
    else:
        print(f"✗ Model file NOT found: {model_path}")
        return False

def check_liran_data_paths():
    """Check for hardcoded 'liran_data' or 'liranatt' paths in the model"""
    model_path = "/home/tommimra/GPS_BGU_model/GPS_bgu_model/EfficientNet_gps.pth"
    
    print("\n" + "=" * 60)
    print("CHECKING FOR LIRAN_DATA PATH ISSUES")
    print("=" * 60)
    
    try:
        # Load model checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Convert checkpoint to string and check for liran paths
        checkpoint_str = str(checkpoint)
        
        if 'liran' in checkpoint_str.lower():
            print("🔍 Found 'liran' references in model checkpoint:")
            lines = checkpoint_str.split('\n')
            for i, line in enumerate(lines):
                if 'liran' in line.lower():
                    print(f"  Line {i}: {line.strip()}")
        else:
            print("✓ No 'liran' references found in model checkpoint")
            
        # Check for specific problematic paths
        problematic_paths = [
            '/home/liranatt/',
            'liran_data',
            'liranatt'
        ]
        
        for path in problematic_paths:
            if path in checkpoint_str:
                print(f"⚠️  Found problematic path: {path}")
        
        return checkpoint
        
    except Exception as e:
        print(f"✗ Error loading checkpoint: {e}")
        traceback.print_exc()
        return None

def try_load_model():
    """Attempt to load the EfficientNet model and catch errors"""
    print("\n" + "=" * 60)
    print("ATTEMPTING TO LOAD EFFICIENTNET MODEL")
    print("=" * 60)
    
    try:
        # Import model class
        from models import EfficientNetGPS
        
        print("✓ Successfully imported EfficientNetGPS class")
        
        # Create model instance
        model = EfficientNetGPS()
        print("✓ Successfully created model instance")
        
        # Try to load state dict
        model_path = "/home/tommimra/GPS_BGU_model/GPS_bgu_model/EfficientNet_gps.pth"
        state_dict = torch.load(model_path, map_location='cpu')
        
        print("✓ Successfully loaded state dict from file")
        
        # Try to load state dict into model
        model.load_state_dict(state_dict)
        print("✓ Successfully loaded state dict into model")
        
        # Test model forward pass
        test_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(test_input)
        print(f"✓ Model forward pass successful, output shape: {output.shape}")
        
        return model
        
    except Exception as e:
        print(f"✗ ERROR loading model: {e}")
        print("\n📋 FULL ERROR TRACEBACK:")
        traceback.print_exc()
        
        # Check if it's a path-related error
        error_str = str(e).lower()
        if 'liran' in error_str or 'path' in error_str or 'file' in error_str:
            print(f"\n⚠️  This appears to be a PATH-RELATED ERROR")
            print("   The model might contain hardcoded paths to 'liran_data' or similar")
            
        return None

def check_config_paths():
    """Check for hardcoded paths in configuration files"""
    print("\n" + "=" * 60)
    print("CHECKING CONFIGURATION FILES FOR LIRAN PATHS")
    print("=" * 60)
    
    files_to_check = [
        'main.py',
        'evaluate.py',
        'data_loader.py',
        'trainer.py'
    ]
    
    for filename in files_to_check:
        filepath = f"/home/tommimra/GPS_BGU_model/GPS_bgu_model/{filename}"
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                if 'liran' in content.lower():
                    print(f"\n📁 Found 'liran' in {filename}:")
                    lines = content.split('\n')
                    for i, line in enumerate(lines, 1):
                        if 'liran' in line.lower():
                            print(f"  Line {i}: {line.strip()}")
                            
            except Exception as e:
                print(f"Error reading {filename}: {e}")

def main():
    """Main debugging function"""
    print("EFFICIENTNET MODEL DEBUG SCRIPT")
    print("Checking for 'liran_data' errors and model loading issues...")
    
    # Check if model file exists
    if not check_model_file():
        return
    
    # Check for liran paths in checkpoint
    checkpoint = check_liran_data_paths()
    
    # Try to load the model
    model = try_load_model()
    
    # Check config files for hardcoded paths
    check_config_paths()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if model is not None:
        print("✅ EfficientNet model loaded successfully - no errors found")
    else:
        print("❌ EfficientNet model failed to load")
        print("💡 Suggestions:")
        print("   1. Check if model was trained on different machine with different paths")
        print("   2. Retrain model or fix hardcoded paths in checkpoint")
        print("   3. Check data paths in configuration files")

if __name__ == "__main__":
    main()