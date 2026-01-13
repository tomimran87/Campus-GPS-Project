#!/usr/bin/env python3
"""
Script to combine tom_data and new_data datasets into a single combined dataset.
"""

import numpy as np
import os

def combine_datasets():
    """Combine tom_data and new_data into a single dataset."""
    
    # Load tom_data
    print("Loading tom_data...")
    tom_X = np.load('/home/tommimra/GPS_BGU_model/tom_data/X_gps.npy')
    tom_y = np.load('/home/tommimra/GPS_BGU_model/tom_data/y_gps.npy')
    
    print(f"Tom data - X shape: {tom_X.shape}, y shape: {tom_y.shape}")
    
    # Load new_data
    print("Loading new_data...")
    new_X = np.load('/home/tommimra/GPS_BGU_model/new_data/X_gps_new.npy')
    new_y = np.load('/home/tommimra/GPS_BGU_model/new_data/y_gps_new.npy')
    
    print(f"New data - X shape: {new_X.shape}, y shape: {new_y.shape}")
    
    # Combine datasets
    print("Combining datasets...")
    combined_X = np.concatenate([tom_X, new_X], axis=0)
    combined_y = np.concatenate([tom_y, new_y], axis=0)
    
    print(f"Combined data - X shape: {combined_X.shape}, y shape: {combined_y.shape}")
    
    # Create combined_data directory
    combined_dir = '/home/tommimra/GPS_BGU_model/combined_data'
    os.makedirs(combined_dir, exist_ok=True)
    
    # Save combined dataset
    print("Saving combined dataset...")
    np.save(os.path.join(combined_dir, 'X_gps.npy'), combined_X)
    np.save(os.path.join(combined_dir, 'y_gps.npy'), combined_y)
    
    print(f"Combined dataset saved to {combined_dir}")
    print("Summary:")
    print(f"  - Tom data samples: {tom_X.shape[0]}")
    print(f"  - New data samples: {new_X.shape[0]}")
    print(f"  - Total combined samples: {combined_X.shape[0]}")
    
    return combined_X, combined_y

if __name__ == "__main__":
    combined_X, combined_y = combine_datasets()