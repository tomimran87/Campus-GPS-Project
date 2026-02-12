# 🌍 Campus Image-to-GPS Regression

### 🔗 [Click Here to Launch the Live App](https://huggingface.co/spaces/liranatt/GPU_Modell_Liran_and_Tom)

### Project Overview
This project implements a **Deep Learning-based Visual Localization System** designed to predict precise GPS coordinates (Latitude, Longitude) from a single ground-level image. Unlike traditional retrieval-based methods, this system utilizes a **Regression-based approach** powered by an **EfficientNet-B7** backbone, allowing for continuous coordinate prediction rather than discrete classification.

The model was trained on a custom dataset collected within the **Ben-Gurion University** campus, demonstrating the feasibility of autonomous navigation and localization in GPS-denied environments using purely visual data.


## Installation

### Prerequisites
- Python 3.9 or higher
- CUDA 11.8+ (for GPU acceleration, optional but recommended)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/tomimran87/Campus-GPS-Project.git
cd Campus-GPS-Project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Data Preparation
Extract GPS coordinates from images and prepare data:

```bash
cd source
python photo_utils.py
```

This will:
1. Read images from `dataset_root/images/`
2. Extract GPS coordinates from EXIF metadata
3. Save processed data to `latest_data/X_photos.npy` and `y_photos.npy`
4. Generate ground truth CSV: `latest_data/gt.csv`

### Training
Train models in the ensemble:

```bash
cd source
python main.py
```

This will:
1. Load and split data (70% train, 15% val, 15% test)
2. Train selected models (EfficientNet, etc.)
3. Save model checkpoints (`*_gps.pth`)
4. Report ensemble performance on test set
5. Display detailed metrics and error analysis

### Evaluation
**The required evaluation function (predict_gps) is at source/predict.py**.


## Project Structure

```
GPS_BGU_model/
├── source/
│   ├── main.py                    # Main training pipeline
│   ├── evaluate.py                # Comprehensive evaluation script
│   ├── predict.py                 # Inference module for predictions
│   ├── photo_utils.py             # Image loading and GPS extraction
│   ├── base_model.py              # Abstract base class for GPS models
│   ├── models.py                  # ResNet, EfficientNet, ConvNeXt implementations
│   ├── data_loader.py             # Data loading with train/val/test splitting
│   ├── loss.py                    # Haversine loss function
│   ├── trainer.py                 # Training orchestration
│   ├── metrics.py                 # Evaluation metrics (mean, median, percentiles)
│   ├── augmentation.py            # Data augmentation strategies
│   ├── check_y_gps.py             # GPS validation utilities
│   └── EfficientNet_gps.pth       # Trained model checkpoint
├── dataset_root/
│   ├── images/                    # Raw image files (jpg, png, heic)
│   └── gt.csv                     # Ground truth GPS coordinates
├── latest_data/
│   ├── X_photos.npy               # Processed images array
│   ├── y_photos.npy               # GPS coordinates array
│   └── gt.csv                     # Ground truth file
├── app/                           # Web application (optional)
├── experiment/                    # Experimental code and notebooks
├── requirements.txt               # Project dependencies
├── readme.md                      # This file
└── README.md                      # Alternative readme
```

## Architecture Details

### Models

**ResNet18GPS**
- Backbone: ResNet18 pretrained on ImageNet
- Feature dimension: 512 (after global average pooling)
- Regression head: 512 → 128 → 64 → 2 with LayerNorm, SiLU, Dropout
- Total parameters: ~11M
- Training speed: ~2-3s/epoch on NVIDIA GPU
- Output activation: Sigmoid [0, 1]

**EfficientNetGPS**
- Backbone: EfficientNet-B0 pretrained on ImageNet
- Feature dimension: 1280 (after global average pooling)
- Regression head: 1280 → 256 → 128 → 2
- Total parameters: ~5.3M
- Efficiency: Best accuracy-to-parameter ratio
- Training speed: ~2-3s/epoch on NVIDIA GPU
- Output activation: Sigmoid [0, 1]

**ConvNeXtGPS**
- Backbone: ConvNeXt-Tiny pretrained on ImageNet
- Feature dimension: 768 (after global average pooling)
- Regression head: 768 → 256 → 128 → 2
- Total parameters: ~28M
- Stability: Excellent (built-in LayerNorm)
- Training speed: ~3-4s/epoch on NVIDIA GPU
- Output activation: Sigmoid [0, 1]

**Additional Variants**
- EfficientNetGPS2: Alternative EfficientNet configuration
- EfficientNetGPS_withGEM: Uses Generalized Mean pooling for attention to salient features

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

Default hyperparameters (in `source/main.py`):

```python
CONFIG = {
    "X_PATH": "./latest_data/X_photos.npy",    # Processed image data
    "Y_PATH": "./latest_data/y_photos.npy",    # GPS coordinates
    "BATCH_SIZE": 32,                          # Images per batch
    "EPOCHS": 120,                             # Maximum training epochs
    "LR": 0.0004,                              # Learning rate
    "ConvLR": 0.00005                          # Learning rate for ConvNext
}
```

**Training Features:**
- Optimizer: AdamW with weight_decay=1e-3
- Learning Rate Scheduler: ReduceLROnPlateau (factor=0.5, patience=3)
- Early Stopping: Patience=10 epochs without improvement
- Gradient Clipping: max_norm=1.0 (prevents explosion)
- Data Normalization: ImageNet statistics ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
- Reproducibility: Random seed 42 for all operations

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

Estimated performance on campus-scale GPS localization (depends on dataset):

| Model | Mean Error | Median Error | 90th Percentile | Accuracy@10m |
|-------|-----------|-------------|-----------------|-------------|
| ResNet18 | 15-25m | 12-20m | 35-50m | 65-75% |
| EfficientNet | 12-20m | 10-15m | 28-45m | 75-85% |
| ConvNeXt | 14-22m | 11-18m | 32-48m | 70-80% |
| **Ensemble** | **10-18m** | **8-14m** | **22-40m** | **80-90%** |

**Performance Factors:**
- Dataset size and geographic distribution
- Image resolution and quality
- GPS coordinate precision
- Seasonal/lighting variations
- Model training duration and hyperparameters

*Actual results will vary based on your specific dataset and conditions*

## Troubleshooting

### Data Issues

**No GPS coordinates found in images:**
- Ensure images contain valid EXIF GPS metadata
- Check supported formats: JPG, PNG, TIFF, HEIC
- Verify image files are in `dataset_root/images/`

**Solution:**
```bash
python source/check_y_gps.py  # Verify GPS data
```

### Training Issues

**NaN Loss During Training:**
1. Reduce learning rate: `CONFIG["LR"] = 0.0001`
2. Increase gradient clipping: `max_norm=0.5` in trainer.py
3. Check for corrupted data samples
4. Try training models individually
5. Reduce batch size: `CONFIG["BATCH_SIZE"] = 16`

**Training is slow:**
- Ensure GPU is being used: `python -c "import torch; print(torch.cuda.is_available())"`
- Install CUDA 11.8+: https://developer.nvidia.com/cuda-11-8-0-download-archive
- Consider reducing batch size or epoch count during development

**Memory Issues:**
1. Reduce batch size: `CONFIG["BATCH_SIZE"] = 8`
2. Reduce model size (use EfficientNet instead of ConvNeXt)
3. Train models sequentially instead of ensemble
4. Use smaller image resolution (reduce IMG_SIZE in photo_utils.py)

### Performance Issues

**Poor localization accuracy:**
1. Verify dataset has diverse locations and viewpoints
2. Check GPS coordinate range is realistic
3. Ensure sufficient training data (>1000 samples recommended)
4. Try longer training: `CONFIG["EPOCHS"] = 200`
5. Enable data augmentation (see `source/augmentation.py`)
6. Ensemble multiple models for better performance

**Overfitting (train loss decreases, val loss increases):**
1. Increase dropout: `dropout=0.5` in model heads
2. Reduce model complexity (use EfficientNet)
3. Add L2 regularization: increase `weight_decay` in optimizer
4. Use more aggressive data augmentation

## Key Implementation Details

### Data Leakage Prevention
- Normalization parameters (min/max GPS values) computed **only** from training set
- Same parameters applied to validation and test sets
- Prevents test set statistics from influencing training

### Numerical Stability
- Double-clamping in Haversine loss to prevent domain errors
- Gradient clipping (max_norm=1.0) to prevent explosion
- NaN/Inf detection with batch skipping during training
- Output activation (Sigmoid) constrains predictions to valid range

### Haversine Loss Function
Optimizes for actual geographic distance rather than Euclidean:
- Accounts for Earth's spherical geometry
- More realistic than L2 distance for GPS coordinates
- Implemented with numerical stability precautions

## References

**Deep Learning Architectures:**
- He et al., "Deep Residual Learning for Image Recognition", CVPR 2016
- Tan & Le, "EfficientNet: Rethinking Model Scaling for CNNs", ICML 2019
- Liu et al., "A ConvNet for the 2020s", CVPR 2022

**Geospatial Computing:**
- Haversine Formula: https://en.wikipedia.org/wiki/Haversine_formula
- Great-circle Distance: https://en.wikipedia.org/wiki/Great-circle_distance

**Libraries Used:**
- PyTorch: https://pytorch.org/
- Pillow: https://python-pillow.org/
- Scikit-learn: https://scikit-learn.org/

## License

MIT License - See LICENSE file for details

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with clear descriptions

## Support

For questions, issues, or suggestions:
- Open an issue on GitHub
- Check existing documentation in source file docstrings
- Review PROJECT_REPORT_NOTES.md for detailed analysis


### 👥 Authors
Developed by **Liran Attar** and **Tom Mimran** as part of the Computer Science Department research track.
