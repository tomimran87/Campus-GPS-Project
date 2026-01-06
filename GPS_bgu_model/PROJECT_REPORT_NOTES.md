# GPS Image Localization - Project Report Notes

**Project**: Campus GPS Coordinate Prediction from Images  
**Dataset**: 2,962 images from Technion campus (224×224×3 RGB)  
**GPS Range**: 31.261-31.263°N, 34.801-34.804°E (~165m × 290m area, 0.048 km²)  
**Models**: ResNet18, EfficientNet-B0, ConvNeXt-Tiny (ensemble)

---

## 1. Data Collection Strategy

### Collection Process
- **Manual collection**: Students walked around campus taking photos at different locations
- **GPS tagging**: Each image tagged with precise GPS coordinates from smartphone
- **Coverage**: 2,962 images across ~0.048 km² campus area
- **Density**: 61,708 images/km² (very high density for small area)

### Collection Challenges
1. **Limited geographic area**: Entire dataset covers only 165m × 290m
   - Makes errors very sensitive: 0.01 normalized error = ~50m real distance
   - Small coordinate range (0.002° latitude, 0.003° longitude)

2. **Real-world variations**:
   - Different times of day (lighting conditions vary)
   - Different weather (sunny, cloudy, rainy)
   - Different smartphones (camera sensors vary)
   - Different orientations (students hold phones differently)
   - Dynamic objects (people walking, cars, changing shadows)

### Data Preprocessing
1. **Image standardization**: Resize to 224×224 (required by pretrained models)
2. **Normalization**: GPS coordinates normalized to [0, 1] using min-max scaling
   - **CRITICAL**: Scaling parameters computed ONLY from training set
   - Computing from full dataset causes **data leakage** (test info leaks into training)
3. **Data splitting**: 70% train (2,073), 15% val (444), 15% test (445)
   - Stratified by GPS coordinates to ensure geographic coverage in all splits

---

## 2. Data Augmentation

### Why Augmentation is Critical

Without augmentation, model would need:
- Thousands of photos from each exact location
- Photos at every time of day
- Photos from every possible phone model
- Photos at every angle and tilt

With only 2,962 images, this is impossible. Augmentation simulates these variations.

### Augmentation Techniques

#### 1. **ColorJitter** (brightness, contrast, saturation, hue)
- **Purpose**: Simulates different lighting conditions and camera sensors
- **Parameters**:
  - Brightness ±30%: Different times of day, shadows, cloud cover
  - Contrast ±30%: Different phone cameras (Samsung vs iPhone vs Huawei)
  - Saturation ±20%: Camera post-processing variations
  - Hue ±10%: White balance differences between cameras
- **Justification**: Students collect images with different phones at different times
- **Effect**: Model learns features independent of lighting/color (e.g., building shapes, not colors)

#### 2. **RandomHorizontalFlip** (p=0.5)
- **Purpose**: Doubles effective dataset size
- **Justification**: Buildings look similar from left/right, GPS coordinates symmetric
- **Effect**: Model learns orientation-invariant features
- **Warning**: Vertical flip NOT used (upside-down buildings look unnatural)

#### 3. **RandomRotation** (±15°)
- **Purpose**: Handles phone tilt while walking
- **Justification**: Students don't hold phones perfectly level
- **Parameters**: Conservative 15° (more would distort vertical buildings)
- **Effect**: Model robust to camera roll angle

#### 4. **RandomPerspective** (distortion=0.2, p=0.3)
- **Purpose**: Simulates camera tilt and height variations
- **Justification**: Short vs tall students, phone at different heights
- **Parameters**: 20% distortion, applied to 30% of images
- **Effect**: Model handles perspective changes

#### 5. **RandomGrayscale** (p=0.1)
- **Purpose**: Handles extreme lighting (very dark scenes)
- **Parameters**: Low 10% probability (color is informative)
- **Justification**: Worst-case scenario (night, heavy shadows)
- **Effect**: Forces model to learn shape features, not just color

#### 6. **ImageNet Normalization** (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- **Purpose**: MANDATORY for pretrained models (ResNet, EfficientNet, ConvNeXt)
- **Justification**: Pretrained backbones trained on ImageNet with these exact values
- **Effect**: Without this, pretrained features completely fail
- **Warning**: Using wrong normalization drops accuracy by 50%+

### Augmentation Pipeline
```
Training:   ColorJitter → HFlip → Rotation → Perspective → Grayscale → Resize → Normalize
Validation: Resize → Normalize (NO augmentation for consistent evaluation)
```

---

## 3. Model Architecture Decisions

### Why Ensemble of Three Models?

1. **Diversity**: Each architecture has different biases
   - ResNet18: Simple residual connections, fast
   - EfficientNet-B0: Compound scaling, efficient
   - ConvNeXt-Tiny: Modern architecture, strong features

2. **Robustness**: Averaging predictions reduces individual model errors
   - If one model fails on edge case, others compensate
   - Ensemble variance lower than single model

3. **Performance**: Ensemble typically 10-20% better than best single model

### Pretrained Backbones
- **All models**: Initialized with ImageNet pretrained weights
- **Justification**: Transfer learning from 1.2M images to our 2,962
- **Effect**: Converges 5-10× faster, achieves better accuracy
- **Fine-tuning strategy**: All layers trainable (full fine-tuning)

### Regression Head Design

**Previous design (BROKEN)**:
```python
Linear → LayerNorm → SiLU → Linear → Sigmoid  # Output constrained to [0,1]
```

**Problem**: Sigmoid saturates gradients near 0 and 1
- For small GPS area, small errors = huge meter errors
- Model got stuck at 0.5 output → predicted campus center → 12,742m error

**Fixed design (CURRENT)**:
```python
Linear → LayerNorm → SiLU → Linear  # NO Sigmoid!
# Initialization: Xavier uniform (gain=0.01), bias=0.5
```

**Why this works**:
- Network learns to output [0,1] via loss function (not forced by activation)
- Small initialization (gain=0.01) starts predictions near center
- Bias=0.5 gives good starting point
- Gradients flow freely (no saturation)

### Loss Function: Haversine Distance

**Formula**:
```
a = sin²(Δlat/2) + cos(lat1) * cos(lat2) * sin²(Δlon/2)
distance = 2 * R * arcsin(√a)
```
Where R = 6,371,000 meters (Earth radius)

**Why Haversine?**
- Accounts for Earth's spherical geometry
- Outputs physical distance in meters (interpretable)
- Better than MSE (MSE doesn't account for latitude/longitude curvature)

**Numerical Stability**:
```python
a = torch.clamp(a, min=1e-10, max=1.0)      # Clamp 'a' first
sqrt_a = torch.clamp(torch.sqrt(a), min=0.0, max=1.0)  # Then clamp sqrt(a)
```
- Double-clamping prevents NaN in arcsin gradient
- Without this, training collapses with NaN at epochs 6-8

---

## 4. Training Strategy

### Hyperparameters
- **Optimizer**: AdamW (weight_decay=0.01 for regularization)
- **Learning Rate**: 0.001 (increased after removing Sigmoid)
  - Higher LR needed because network must learn output range
  - Previously Sigmoid forced [0,1], now network learns this
- **Scheduler**: ReduceLROnPlateau (patience=3, factor=0.5)
  - Reduces LR by 50% if validation doesn't improve for 3 epochs
- **Gradient Clipping**: max_norm=1.0
  - Prevents gradient explosion in Haversine loss
  - Critical for training stability
- **Early Stopping**: Patience=5 epochs
  - Stops if validation doesn't improve for 5 epochs
  - Prevents overfitting on small dataset

### Regularization
1. **Dropout**: 30% in regression head (prevents overfitting)
2. **Weight Decay**: 0.01 in AdamW (L2 regularization)
3. **Data Augmentation**: As described above
4. **Early Stopping**: Prevents training too long

### Training Process
1. **Warm-up**: First few epochs learn output range
2. **Convergence**: Loss decreases, accuracy improves
3. **Plateau**: Learning rate reduced when stuck
4. **Early Stop**: Training ends when validation plateaus

---

## 5. Evaluation Metrics

### Primary Metric: Mean Distance Error (meters)
- **Definition**: Average Haversine distance between predicted and true GPS
- **Interpretation**: Directly measures localization accuracy
- **Target**: 8-15m final error (good for 165m×290m area)
- **Baseline**: Random prediction would be ~80m (campus radius)

### Secondary Metrics
1. **Median Error**: Robust to outliers (better than mean for skewed distributions)
2. **95th Percentile**: Worst-case performance (5% of predictions worse than this)
3. **Max Error**: Absolute worst prediction (identifies failure modes)
4. **Accuracy @ Xm**: Percentage of predictions within X meters
   - Accuracy @ 10m: % within 10 meters
   - Accuracy @ 25m: % within 25 meters
   - Accuracy @ 50m: % within 50 meters

### Error Analysis
- **Spatial distribution**: Which campus areas have highest errors?
- **Temporal patterns**: Do errors correlate with time of day?
- **Model comparison**: Which model (ResNet/EfficientNet/ConvNeXt) performs best?

---

## 6. Ablation Studies (RECOMMENDED)

### Study 1: Effect of Augmentation
**Hypothesis**: Augmentation significantly improves generalization

**Experiment**:
- Train model WITH augmentation (current setup)
- Train model WITHOUT augmentation (remove ColorJitter, flips, rotation, etc.)
- Compare validation errors

**Expected**: With augmentation ~30% better than without

**Why**: Demonstrates augmentation's importance for small dataset

---

### Study 2: Effect of Pretrained Weights
**Hypothesis**: Pretrained weights crucial for small dataset

**Experiment**:
- Train model with ImageNet pretrained weights (current)
- Train model from scratch (random initialization)
- Compare training speed and final accuracy

**Expected**: Pretrained converges 5× faster, 20% better accuracy

**Why**: Shows transfer learning value with only 2,962 images

---

### Study 3: Sigmoid vs No Sigmoid
**Hypothesis**: Removing Sigmoid fixes saturation problem

**Experiment**:
- Train with Sigmoid activation (previous broken version)
- Train without Sigmoid (current fixed version)
- Compare training curves and final errors

**Expected**: 
- With Sigmoid: Stuck at 12,742m error
- Without Sigmoid: Achieves <15m error

**Why**: Demonstrates critical bug fix

---

### Study 4: Ensemble vs Single Model
**Hypothesis**: Ensemble better than best individual model

**Experiment**:
- Train ResNet18 alone
- Train EfficientNet-B0 alone
- Train ConvNeXt-Tiny alone
- Compare with ensemble average

**Expected**: Ensemble 10-15% better than best single model

**Why**: Shows ensemble value for production system

---

## 7. Known Issues and Fixes

### Issue 1: NaN Collapse (FIXED)
**Symptom**: Training loss became NaN at epochs 6-8  
**Cause**: Haversine formula's arcsin gradient explodes when argument near 1  
**Fix**: Double-clamping in loss.py (clamp 'a', then clamp sqrt(a))  
**Validation**: No NaN in recent training runs

---

### Issue 2: Data Leakage (FIXED)
**Symptom**: Unrealistically good validation performance initially  
**Cause**: GPS normalization computed from entire dataset before split  
**Fix**: Compute min/max ONLY from training set (data_loader.py lines 91-105)  
**Validation**: Val error now realistic (not artificially low)

---

### Issue 3: Sigmoid Saturation (FIXED)
**Symptom**: Model stuck at 12,742m error, all predictions identical  
**Cause**: Sigmoid activation saturates at 0.5 → predicts campus center  
**Fix**: Removed Sigmoid from all three models, added Xavier init (gain=0.01)  
**Validation**: TBD (need to retrain after this fix)

---

### Issue 4: Missing Augmentation (FIXED)
**Symptom**: Model overfits to training lighting/angles  
**Cause**: No data augmentation, only 2,962 images  
**Fix**: Created augmentation.py with comprehensive pipeline  
**Validation**: TBD (need to retrain with augmentation)

---

## 8. Expected Results After Fixes

### Training Curves
- **Loss**: Should decrease smoothly from ~100m to <15m
- **Validation**: Should track training (gap indicates overfitting)
- **Early epochs**: High error (~80m) as model learns
- **Middle epochs**: Rapid improvement (80m → 20m)
- **Late epochs**: Slow refinement (20m → 10m)

### Final Performance Targets
- **Mean Error**: 8-15 meters
- **Median Error**: 5-10 meters  
- **95th Percentile**: <30 meters
- **Accuracy @ 10m**: >60%
- **Accuracy @ 25m**: >85%

### Comparison to Baseline
- **Random prediction**: ~80m (center of campus)
- **Nearest neighbor**: ~40m (closest training image)
- **Our model**: 8-15m (5-10× better than baseline)

---

## 9. Future Improvements

1. **More data**: Collect 10,000+ images for better coverage
2. **Test-time augmentation**: Average predictions over augmented versions
3. **Attention mechanisms**: Focus on discriminative landmarks
4. **Multi-task learning**: Predict building ID + GPS jointly
5. **Confidence estimation**: Output uncertainty for each prediction
6. **Temporal modeling**: Use image sequences (video) for better localization

---

## 10. Conclusion

This project demonstrates:
1. **Transfer learning**: Pretrained models work with small datasets
2. **Data augmentation**: Critical for robustness with limited data
3. **Loss function design**: Haversine better than MSE for GPS
4. **Debugging**: Fixed NaN collapse, data leakage, Sigmoid saturation
5. **Engineering**: Proper train/val/test split, early stopping, gradient clipping

**Key takeaway**: Small datasets require careful engineering (augmentation, pretrained weights, proper regularization) to achieve good performance.
