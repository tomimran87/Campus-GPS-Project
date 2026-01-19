#!/usr/bin/env python3
"""
Master Script: Run All 30-Epoch Analysis
=========================================
1. Train EfficientNet (30 epochs) if not already done
2. Organize all model files into proper structure
3. Generate all comparison plots
4. Run geographic analysis for all models

Usage:
    python run_all_30epochs.py
    
Output:
    testing/30epochs/
    ├── ResNet18/
    │   ├── ResNet18_30epochs_gps.pth
    │   ├── plots/
    │   └── geographic/
    ├── EfficientNet/
    │   ├── EfficientNet_30epochs_gps.pth
    │   ├── plots/
    │   └── geographic/
    ├── ConvNeXt/
    │   ├── ConvNeXt_30epochs_gps.pth
    │   ├── plots/
    │   └── geographic/
    └── comparison/
        ├── plots/
        └── geographic/
"""

import subprocess
import sys
from pathlib import Path


def run_script(script_name, description):
    """Run a Python script and handle errors."""
    print("\n" + "=" * 70)
    print(f"RUNNING: {description}")
    print("=" * 70)
    
    result = subprocess.run(
        [sys.executable, script_name],
        capture_output=False
    )
    
    if result.returncode != 0:
        print(f"\n✗ ERROR: {script_name} failed with code {result.returncode}")
        return False
    
    return True


def check_efficientnet_trained():
    """Check if EfficientNet 30-epoch model exists."""
    paths_to_check = [
        Path("testing/30epochs/EfficientNet/efficientnet_errors.npy"),
        Path("efficientnet_errors.npy"),
    ]
    return any(p.exists() for p in paths_to_check)


def main():
    print("=" * 70)
    print("MASTER SCRIPT: 30-EPOCH ANALYSIS PIPELINE")
    print("=" * 70)
    
    # Step 1: Train EfficientNet if needed
    if not check_efficientnet_trained():
        print("\n📌 EfficientNet not trained yet. Starting training...")
        if not run_script("train_efficientnet_30epochs.py", "Train EfficientNet (30 Epochs)"):
            print("✗ EfficientNet training failed. Exiting.")
            return
    else:
        print("\n✓ EfficientNet already trained. Skipping training.")
    
    # Step 2: Organize files and generate comparison plots
    if not run_script("organize_and_compare_30epochs.py", "Organize & Generate Comparison Plots"):
        print("✗ Comparison generation failed. Exiting.")
        return
    
    # Step 3: Run geographic analysis
    if not run_script("geographic_analysis_30epochs.py", "Geographic Analysis"):
        print("✗ Geographic analysis failed. Exiting.")
        return
    
    print("\n" + "=" * 70)
    print("✅ ALL 30-EPOCH ANALYSIS COMPLETE!")
    print("=" * 70)
    print("\nOutput structure:")
    print("  testing/30epochs/")
    print("  ├── ResNet18/")
    print("  │   ├── plots/ (6 individual plots)")
    print("  │   └── geographic/ (5 plots + analysis)")
    print("  ├── EfficientNet/")
    print("  │   ├── plots/")
    print("  │   └── geographic/")
    print("  ├── ConvNeXt/")
    print("  │   ├── plots/")
    print("  │   └── geographic/")
    print("  └── comparison/")
    print("      ├── plots/ (8 comparison plots)")
    print("      └── geographic/ (3 comparison plots)")
    print("\nReady for 50-epoch training when you decide the learning rate!")


if __name__ == "__main__":
    main()
