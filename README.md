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

### 1. Data Setup (Required Before Anything Else)

The `dataset_root/images/` folder is **empty by default** — images are not included in the Git repository due to their size. You must populate it before running any scripts.

**Option A: Download the BGU campus dataset**

Download the images from the shared Google Drive folder:
🔗 **[Download Dataset Images](https://drive.google.com/drive/folders/1-sgYP8CwPJzz3IkQI049IjyAc8SmlZhR)**

After downloading, place **all image files** into:
```
<your-path>/Campus-GPS-Project/dataset_root/images/
```

The filenames must match those listed in `dataset_root/gt.csv` (e.g., `IMG_1725.JPG`, `IMG_1082.JPG`, etc.).

**Option B: Use your own photos from the BGU area**

You can also take your own geotagged photos around the Ben-Gurion University campus. Ensure each photo has valid **EXIF GPS metadata** embedded (most smartphone cameras do this automatically). Place them in the same `dataset_root/images/` folder and also make sure you add the matching ground truth and gps location to gt.csv 

> ⚠️ **If this folder is empty, `photo_utils.py` will produce empty arrays and `main.py` will crash.**

### 2. Data Preparation
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

### 3. Training
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

- **Step 1 — Place the trained model in the project root:**
  - Put the checkpoint file `EfficientNet_gps.pth` in the repository root (CAMPUS-GPS-Project). The trained model can be downloaded from:
    https://drive.google.com/drive/u/0/folders/1wyGhNcXOpeMbq_Intn9gNEH0A8diybbX

- **Step 2 — Create an evaluation script in `source/`:**
  - The inference function `predict_gps` is implemented in `source/predict.py` and will load `../EfficientNet_gps.pth` by default (relative to `source/`).
  - We provide a minimal example script `source/eval_example.py` that demonstrates how to call `predict_gps` on a single image.

Example usage (create `source/eval_example.py` or run the provided example):

```python
from predict import predict_gps
import numpy as np
from PIL import Image

# Put the path to the image here:
img_path = "../dataset_root/images/IMG_2322.JPG"
image = np.array(Image.open(img_path).convert('RGB'))

gps_prediction = predict_gps(image)

print(f"predicted gps: {gps_prediction}")
```

Run the example from the repository root:

```bash
cd source
python eval_example.py
```

Notes:
- `source/predict.py` exposes `predict_gps(image: np.ndarray) -> np.ndarray` which expects an RGB `uint8` numpy array with shape `(H, W, 3)` and values in `[0, 255]`.
- The default checkpoint path in `source/predict.py` is `../EfficientNet_gps.pth` (project root). If you store the checkpoint elsewhere, pass the path when instantiating `GPSPredictor` or modify `MODEL_CHECKPOINT` in `source/predict.py`.



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

## Advanced Configuration: Training Options

### Single vs. Multi-Model Training

By default, `main.py` trains **only EfficientNet-B0**. The code supports three architectures that can be trained individually or together as an ensemble.

In `source/main.py`, find the `models_to_train` dictionary (around line 70):

```python
# Current default — single model:
models_to_train = {
    "EfficientNet": EfficientNetGPS(),
}
```

To enable **multi-model ensemble training**, replace it with:

```python
# Full ensemble — trains all three architectures:
models_to_train = {
    "EfficientNet": EfficientNetGPS(),
    "ResNet": ResNetGPS(),
    "ConvNeXt": ConvNextGPS(),
}
```

You can also train any single alternative model:

```python
# Train only ResNet:
models_to_train = {
    "ResNet": ResNetGPS(),
}
```

> **Note:** ConvNeXt is sensitive to learning rate. If you enable it, consider using `CONFIG["ConvLR"] = 0.00005` or lowering `CONFIG["LR"]` to `0.0001`. See the project report for details on ConvNeXt instability at higher learning rates.

### Additional Model Variants

The codebase also includes experimental variants imported in `main.py`:
- `EfficientNetGPS2` — Alternative EfficientNet configuration
- `EfficientNetGPS_withGEM` — Uses Generalized Mean pooling for attention to salient features

These can be added to `models_to_train` in the same way.

---

## Understanding the Output

When you run `python main.py`, the logs contain several important sections. Here's what they mean:

### Normalization Parameters
```
Normalization parameters (from training set only):
  Min: lat=31.261851, lon=34.804034
  Max: lat=31.262177, lon=34.804194
```
The model predicts GPS coordinates normalized to the range [0, 1]. To convert between real GPS coordinates and this normalized range, the pipeline computes the **min/max latitude and longitude from the training split only**. These same parameters are then applied to the validation and test sets. This prevents **data leakage** — the model never "sees" test set statistics during training.

### Ensemble Prediction
```
Ensemble (Mean of 3 models) Prediction:
```
The code is architecturally designed to support an **ensemble of multiple models**. It averages predictions from all models in `models_to_train`. If you only train one model (the default EfficientNet), the "Ensemble" result is simply the output of that single model — the averaging has no effect since there's only one prediction to average. This is expected behavior, not a bug.

### Summary Block (Best/Worst Model)
```
Summary:
  Best Individual Model: EfficientNet (6.74m)
  Worst Individual Model: EfficientNet (6.74m)
  Ensemble Error: 6.74m
```
When only one model is trained, the Best and Worst model will be identical (since there's only one to compare). This comparison is meaningful when multiple models are trained in ensemble mode — it helps identify which architecture performs best. When running a single model, this section is automatically simplified to avoid redundant output.

---

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

## Actual Performance (Research Results)

The following results were obtained from our research using 3,646 campus images with the best configuration per model (100 epochs, full data). See the [project report (PDF)](FINAL_GPS_PROJECT%20(3).pdf) for the complete analysis.

### Best Configuration per Model (Phase 1)

| Model | Mean Error | Median Error | P75 | P90 | P95 | Max Error |
|-------|-----------|-------------|-----|-----|-----|-----------|
| **EfficientNet-B0** | **5.24m** | **2.75m** | 5.39m | 11.91m | 18.01m | 94.05m |
| ConvNeXt-Tiny | 6.12m | 3.65m | 7.38m | 12.93m | 16.74m | 99.66m |
| ResNet18 | 6.43m | 3.77m | 7.89m | 15.15m | 20.28m | 94.76m |

### Final Optimized Model (Phase 2 — EfficientNet-B0)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Mean Error | 5.24m | Average distance from ground truth |
| Median Error | 4.01m | 50% of predictions within this radius |
| P75 | 5.35m | 75% of predictions are highly accurate |
| P95 | 10.22m | 95% of predictions are usable (<10m) |
| Accuracy (<10m) | 94.5% | High reliability |
| Accuracy (<20m) | 98.2% | Near-perfect coarse localization |
| Max Error | 90.07m | Improved stability vs. Phase 1 baselines |

### Model Robustness to Augmented Testing (100 Epochs, Half Data)

| Model | Regular Test | Augmented Test | Degradation |
|-------|-------------|---------------|-------------|
| ConvNeXt | 9.62m | 14.35m | 1.49× |
| EfficientNet | 10.67m | 24.34m | 2.28× |
| ResNet18 | 11.34m | 24.38m | 2.15× |

> **Key Finding:** EfficientNet-B0 achieves the best overall accuracy, while ConvNeXt-Tiny is the most robust to viewpoint changes. See the full report for details on all 10 experiments.

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
