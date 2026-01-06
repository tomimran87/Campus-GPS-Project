# Campus Image-to-GPS Localization Project

## Overview
This project implements a Deep Learning system to predict precise GPS coordinates (Latitude, Longitude) from a single ground-level image. It is designed for campus-scale localization tasks where GPS signals might be degraded or unavailable.

The system uses an **Ensemble of Convolutional Neural Networks (CNNs)** with different backbones (ResNet18, EfficientNet-B0, ConvNeXt-Tiny) to achieve high accuracy. It minimizes the **Haversine Distance** (real-world meters) rather than standard mean squared error.

## Key Features
* **Multi-Backbone Support:** Switch easily between ResNet18, EfficientNet-B0, and ConvNeXt-Tiny
* **Physics-Informed Loss:** Uses a differentiable Haversine Loss function to optimize for physical distance (meters)
* **Ensemble Inference:** Combines predictions from multiple models to reduce variance and improve accuracy
* **Modular Architecture:** Clean separation of data loading, model definition, and training logic using Abstract Base Classes
* **Numerical Stability:** Double-clamping strategy, gradient clipping, and NaN detection prevent training collapse
* **Data Hygiene:** Proper train/val/test split with normalization computed only from training set
* **Comprehensive Metrics:** Beyond mean error - includes median, percentiles, and accuracy thresholds

## Recent Improvements (January 2026)

### 🔧 Critical Fixes
- **NaN Collapse Prevention**: Added double-clamping in Haversine loss and gradient clipping (max_norm=1.0)
- **Data Leakage Fix**: Normalization parameters now computed only from training set
- **Training Stability**: Reduced initial LR to 0.0001, added early stopping with patience=5
- **Reproducibility**: Random seeds set for PyTorch and NumPy

### 📊 New Features
- **Test Set Evaluation**: Proper 70/15/15 train/val/test split
- **Comprehensive Metrics**: GPSMetrics class with percentiles and accuracy thresholds
- **Model Improvements**: Added LayerNorm and Sigmoid output activation to all models
- **Evaluation Script**: Dedicated `evaluate.py` for model comparison
- **Enhanced Documentation**: Detailed docstrings explaining all math and design choices

## Installation
Ensure you have Python 3.8+ and install dependencies:

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install torch torchvision numpy scikit-learn matplotlib tqdm
```

## Quick Start

### Training
Train all three models in the ensemble:

```bash
python main.py
```

This will:
1. Load and split data (70% train, 15% val, 15% test)
2. Train ResNet18, EfficientNet-B0, and ConvNeXt-Tiny
3. Save model checkpoints (`*_gps.pth`)
4. Report ensemble performance on test set

### Evaluation
Evaluate trained models with comprehensive metrics:

```bash
python evaluate.py
```

This provides:
- Individual model performance
- Ensemble performance
- Detailed metrics (mean, median, percentiles, accuracy@thresholds)
- Model comparison table

## Project Structure

```
project/
├── main.py              # Main training pipeline
├── evaluate.py          # Comprehensive evaluation script
├── base_model.py        # Abstract base class for GPS models
├── models.py            # ResNet, EfficientNet, ConvNeXt implementations
├── data_loader.py       # Data loading with proper splitting
├── loss.py              # Haversine loss with numerical stability
├── trainer.py           # Training orchestration with gradient clipping
├── metrics.py           # Comprehensive evaluation metrics
├── requirements.txt     # Project dependencies
├── X_gps.npy           # Image data (2962, 224, 224, 3)
├── y_gps.npy           # GPS coordinates (2962, 2)
└── README.md           # This file
```

## Architecture Details

### Models

**ResNet18GPS**
- Backbone: ResNet18 pretrained on ImageNet
- Features: 512-dim after global average pooling
- Head: 512 → 128 → 64 → 2 with LayerNorm and SiLU
- Parameters: ~11M
- Speed: Fast training (~2-3s/epoch on GPU)

**EfficientNetGPS**
- Backbone: EfficientNet-B0 pretrained on ImageNet
- Features: 1280-dim after global average pooling
- Head: 1280 → 256 → 2 with LayerNorm and SiLU
- Parameters: ~5M
- Efficiency: Best accuracy per parameter

**ConvNextGPS**
- Backbone: ConvNeXt-Tiny pretrained on ImageNet
- Features: 768-dim after global average pooling
- Head: 768 → 128 → 2 with LayerNorm and SiLU
- Parameters: ~28M
- Stability: Most stable training (internal LayerNorm)

### Haversine Loss Function

The Haversine formula computes the great-circle distance between two points on Earth:

```
a = sin²(Δlat/2) + cos(lat₁) × cos(lat₂) × sin²(Δlon/2)
c = 2 × asin(√a)
distance = R × c
```

Where R = 6,371,000 meters (Earth's mean radius)

**Numerical Stability Features:**
- Double-clamping: Clamp both `a` and `√a` to prevent domain errors
- Gradient clipping: Prevents explosion when predictions become accurate
- Output clamping: Limits maximum distance to prevent overflow

## Training Configuration

Default hyperparameters (in `main.py`):

```python
CONFIG = {
    "X_PATH": "X_gps.npy",
    "Y_PATH": "y_gps.npy",
    "BATCH_SIZE": 32,
    "EPOCHS": 30,
    "LR": 0.0001  # Reduced for stability
}
```

**Training Features:**
- Optimizer: AdamW with weight decay 1e-4
- LR Scheduler: ReduceLROnPlateau (factor=0.5, patience=3)
- Early Stopping: Patience=5 epochs
- Gradient Clipping: max_norm=1.0
- Data Augmentation: None (images are pre-normalized)

## Performance Metrics

The evaluation script reports:

**Error Statistics:**
- Mean Error (meters)
- Median Error (meters) - robust to outliers
- Standard Deviation
- Min/Max Error

**Percentiles:**
- 50th, 90th, 95th, 99th percentile errors
- Shows error distribution shape

**Accuracy Thresholds:**
- % predictions within 5m
- % predictions within 10m
- % predictions within 50m

## Expected Performance

After fixes (approximate on test set):

| Model | Mean Error | Median Error | 90th %ile | Acc@10m |
|-------|-----------|-------------|-----------|---------|
| ResNet18 | 20-30m | 15-25m | 40-60m | 60-75% |
| EfficientNet | 15-25m | 12-20m | 30-50m | 70-80% |
| ConvNeXt | 18-28m | 14-22m | 35-55m | 65-78% |
| **Ensemble** | **12-22m** | **10-18m** | **25-45m** | **75-85%** |

*Note: Performance depends on dataset quality and geographic distribution*

## Troubleshooting

### NaN Loss During Training

If you still encounter NaN:
1. Reduce learning rate further: `CONFIG["LR"] = 5e-5`
2. Increase gradient clipping: `max_norm=0.5` in trainer.py
3. Check data for corrupted samples
4. Try training individual models separately

### Memory Issues

If you run out of GPU memory:
1. Reduce batch size: `CONFIG["BATCH_SIZE"] = 16`
2. Train models one at a time
3. Use mixed precision training (add to trainer.py)

### Poor Performance

If models don't improve:
1. Check data quality and distribution
2. Verify GPS coordinates are in correct format
3. Add data augmentation (see data_loader.py comments)
4. Increase training epochs
5. Try different learning rates

## Citation

If you use this code in your research, please cite:

```
@misc{gps_localization_2026,
  title={Campus Image-to-GPS Localization with Ensemble CNNs},
  author={Your Name},
  year={2026},
  howpublished={\\url{https://github.com/yourusername/gps-localization}}
}
```

## References

- He et al., "Deep Residual Learning for Image Recognition", CVPR 2016
- Tan & Le, "EfficientNet: Rethinking Model Scaling for CNNs", ICML 2019
- Liu et al., "A ConvNet for the 2020s", CVPR 2022
- Haversine Formula: https://en.wikipedia.org/wiki/Haversine_formula

## License

MIT License - See LICENSE file for details

## Contact

For questions or issues, please open an issue on GitHub or contact [your email]