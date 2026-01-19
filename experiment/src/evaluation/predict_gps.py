"""
GPS Prediction Module - Required Interface for Project Evaluation

This module implements the predict_gps function as specified in the project requirements.
Includes Test-Time Augmentation (TTA) and ensemble prediction for improved accuracy.
"""

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

# Import models - adjust path based on your structure
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.models import GPSResNet18, GPSEfficientNet, GPSConvNeXt


class GPSPredictor:
    """
    GPS Predictor with ensemble and test-time augmentation support.
    
    Creative Features:
    1. Ensemble of multiple models (ResNet18, EfficientNet, ConvNeXt)
    2. Test-Time Augmentation (TTA) - average predictions over augmented versions
    3. Weighted ensemble based on validation performance
    4. Uncertainty estimation via prediction variance
    """
    
    def __init__(self, 
                 model_dir: str = None,
                 use_ensemble: bool = True,
                 use_tta: bool = True,
                 device: str = None):
        """
        Initialize the GPS predictor.
        
        Args:
            model_dir: Directory containing trained model weights
            use_ensemble: Whether to use ensemble of all 3 models
            use_tta: Whether to use test-time augmentation
            device: 'cuda' or 'cpu'
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_ensemble = use_ensemble
        self.use_tta = use_tta
        
        # Default model directory
        if model_dir is None:
            model_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                'models', 'pretrained'
            )
        
        # Load normalization parameters (computed from training set only - no data leakage!)
        results_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'results'
        )
        self.min_val = np.load(os.path.join(results_dir, 'min_val.npy'))
        self.max_val = np.load(os.path.join(results_dir, 'max_val.npy'))
        
        # Model weights based on validation performance (from experiments)
        # Higher weight = better performance on validation set
        self.model_weights = {
            'efficientnet': 0.45,  # Best overall
            'convnext': 0.35,      # Second best with lr=0.0001
            'resnet18': 0.20       # Good but not as strong
        }
        
        # Load models
        self.models = {}
        self._load_models(model_dir)
        
        # Standard ImageNet normalization (required for pretrained models)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # Base transform (no augmentation)
        self.base_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            self.normalize
        ])
        
        # TTA transforms - multiple views of the same image
        self.tta_transforms = self._create_tta_transforms()
    
    def _load_models(self, model_dir: str):
        """Load all trained models."""
        model_configs = [
            ('resnet18', GPSResNet18, 'ResNet18_gps.pth'),
            ('efficientnet', GPSEfficientNet, 'EfficientNet_gps.pth'),
            ('convnext', GPSConvNeXt, 'ConvNeXt_gps.pth'),
        ]
        
        for name, model_class, weight_file in model_configs:
            weight_path = os.path.join(model_dir, weight_file)
            if os.path.exists(weight_path):
                model = model_class()
                model.load_state_dict(torch.load(weight_path, map_location=self.device))
                model.to(self.device)
                model.eval()
                self.models[name] = model
                print(f"Loaded {name} from {weight_path}")
            else:
                print(f"Warning: {weight_path} not found, skipping {name}")
        
        if not self.models:
            raise RuntimeError("No models loaded! Check model_dir path.")
    
    def _create_tta_transforms(self):
        """
        Create Test-Time Augmentation transforms.
        
        TTA Strategy:
        - Original image
        - Horizontal flip
        - Small rotations (±5°)
        - Slight brightness variations
        
        This simulates the variations we might see in real test images
        and averages out random noise in predictions.
        """
        tta_list = []
        
        # 1. Original (no augmentation)
        tta_list.append(transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            self.normalize
        ]))
        
        # 2. Horizontal flip
        tta_list.append(transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            self.normalize
        ]))
        
        # 3. Small rotation left
        tta_list.append(transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(degrees=(5, 5)),
            transforms.ToTensor(),
            self.normalize
        ]))
        
        # 4. Small rotation right
        tta_list.append(transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(degrees=(-5, -5)),
            transforms.ToTensor(),
            self.normalize
        ]))
        
        # 5. Slight brightness increase
        tta_list.append(transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ColorJitter(brightness=0.1),
            transforms.ToTensor(),
            self.normalize
        ]))
        
        return tta_list
    
    def _denormalize_gps(self, normalized_coords: np.ndarray) -> np.ndarray:
        """
        Convert normalized [0,1] coordinates back to real GPS values.
        
        Args:
            normalized_coords: Shape (2,) with normalized lat/lon
            
        Returns:
            Real GPS coordinates (latitude, longitude)
        """
        real_coords = normalized_coords * (self.max_val - self.min_val) + self.min_val
        return real_coords
    
    def _predict_single_model(self, model: nn.Module, image_tensor: torch.Tensor) -> np.ndarray:
        """Get prediction from a single model."""
        with torch.no_grad():
            output = model(image_tensor)
            return output.cpu().numpy().flatten()
    
    def _predict_with_tta(self, pil_image: Image.Image) -> tuple:
        """
        Predict with Test-Time Augmentation.
        
        Returns:
            tuple: (mean_prediction, std_prediction) for uncertainty estimation
        """
        all_predictions = []
        
        for transform in self.tta_transforms:
            image_tensor = transform(pil_image).unsqueeze(0).to(self.device)
            
            if self.use_ensemble:
                # Weighted ensemble prediction
                ensemble_pred = np.zeros(2)
                total_weight = 0
                
                for name, model in self.models.items():
                    pred = self._predict_single_model(model, image_tensor)
                    weight = self.model_weights.get(name, 1.0)
                    ensemble_pred += pred * weight
                    total_weight += weight
                
                ensemble_pred /= total_weight
                all_predictions.append(ensemble_pred)
            else:
                # Use only EfficientNet (best single model)
                if 'efficientnet' in self.models:
                    pred = self._predict_single_model(self.models['efficientnet'], image_tensor)
                else:
                    pred = self._predict_single_model(list(self.models.values())[0], image_tensor)
                all_predictions.append(pred)
        
        all_predictions = np.array(all_predictions)
        mean_pred = np.mean(all_predictions, axis=0)
        std_pred = np.std(all_predictions, axis=0)
        
        return mean_pred, std_pred
    
    def predict(self, image: np.ndarray, return_uncertainty: bool = False) -> np.ndarray:
        """
        Predict GPS coordinates from an image.
        
        Args:
            image: numpy array of shape (H, W, 3), RGB, uint8, [0, 255]
            return_uncertainty: If True, also return prediction uncertainty
            
        Returns:
            np.ndarray of shape (2,) with [latitude, longitude]
            If return_uncertainty=True, returns (prediction, uncertainty)
        """
        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(image)
        
        if self.use_tta:
            normalized_pred, uncertainty = self._predict_with_tta(pil_image)
        else:
            # Simple prediction without TTA
            image_tensor = self.base_transform(pil_image).unsqueeze(0).to(self.device)
            
            if self.use_ensemble:
                normalized_pred = np.zeros(2)
                total_weight = 0
                
                for name, model in self.models.items():
                    pred = self._predict_single_model(model, image_tensor)
                    weight = self.model_weights.get(name, 1.0)
                    normalized_pred += pred * weight
                    total_weight += weight
                
                normalized_pred /= total_weight
            else:
                if 'efficientnet' in self.models:
                    normalized_pred = self._predict_single_model(self.models['efficientnet'], image_tensor)
                else:
                    normalized_pred = self._predict_single_model(list(self.models.values())[0], image_tensor)
            
            uncertainty = None
        
        # Denormalize to real GPS coordinates
        gps_prediction = self._denormalize_gps(normalized_pred)
        
        # Ensure output format matches specification
        result = np.array([gps_prediction[0], gps_prediction[1]], dtype=np.float32)
        
        if return_uncertainty:
            return result, uncertainty
        return result


# Global predictor instance (lazy initialization)
_predictor = None


def predict_gps(image: np.ndarray) -> np.ndarray:
    """
    Predict GPS latitude and longitude from a single RGB image.
    
    This is the REQUIRED function interface for project evaluation.
    
    Args:
        image: numpy.ndarray
            - Shape: (H, W, 3)
            - Channel order: RGB (NOT BGR)
            - Dtype: uint8
            - Value range: [0, 255]
    
    Returns:
        numpy.ndarray
            - Shape: (2,)
            - Dtype: float32
            - Format: [latitude, longitude]
    
    Example:
        >>> import numpy as np
        >>> from PIL import Image
        >>> img = np.array(Image.open('test_image.jpg'))
        >>> gps = predict_gps(img)
        >>> print(gps)  # e.g., array([31.262345, 34.803210], dtype=float32)
    """
    global _predictor
    
    # Lazy initialization of predictor
    if _predictor is None:
        _predictor = GPSPredictor(
            use_ensemble=True,   # Use all 3 models
            use_tta=True         # Use test-time augmentation
        )
    
    # Validate input
    assert isinstance(image, np.ndarray), "Input must be numpy.ndarray"
    assert len(image.shape) == 3, "Image must have 3 dimensions (H, W, 3)"
    assert image.shape[2] == 3, "Image must have 3 channels (RGB)"
    assert image.dtype == np.uint8, "Image dtype must be uint8"
    
    # Get prediction
    prediction = _predictor.predict(image)
    
    # Validate output
    assert isinstance(prediction, np.ndarray), "Output must be numpy.ndarray"
    assert prediction.shape == (2,), "Output shape must be (2,)"
    assert prediction.dtype == np.float32, "Output dtype must be float32"
    assert np.isfinite(prediction).all(), "Output must contain finite values"
    
    return prediction


# ============================================================================
# CREATIVE ADDITIONS FOR TESTING/COMPARISON
# ============================================================================

def predict_gps_with_confidence(image: np.ndarray) -> tuple:
    """
    Predict GPS with confidence/uncertainty estimation.
    
    Uses variance across TTA augmentations to estimate prediction confidence.
    Higher variance = lower confidence = model is uncertain about this image.
    
    Returns:
        tuple: (gps_prediction, confidence_score)
            - gps_prediction: np.array([lat, lon], dtype=float32)
            - confidence_score: float in [0, 1], higher = more confident
    """
    global _predictor
    
    if _predictor is None:
        _predictor = GPSPredictor(use_ensemble=True, use_tta=True)
    
    prediction, uncertainty = _predictor.predict(image, return_uncertainty=True)
    
    # Convert uncertainty (std) to confidence score
    # Lower std = higher confidence
    if uncertainty is not None:
        # Normalize: typical std range is 0.01-0.1 in normalized coordinates
        avg_std = np.mean(uncertainty)
        confidence = max(0, min(1, 1 - avg_std * 10))  # Scale to [0, 1]
    else:
        confidence = 0.5  # Unknown confidence
    
    return prediction, confidence


def predict_gps_ablation(image: np.ndarray, 
                         use_ensemble: bool = True,
                         use_tta: bool = True,
                         model_name: str = None) -> dict:
    """
    Predict GPS with ablation options for comparison experiments.
    
    This function allows testing different configurations:
    - Single model vs ensemble
    - With vs without TTA
    - Specific model selection
    
    Args:
        image: Input image (H, W, 3), RGB, uint8
        use_ensemble: Whether to use model ensemble
        use_tta: Whether to use test-time augmentation
        model_name: Specific model to use ('resnet18', 'efficientnet', 'convnext')
                    If None, uses ensemble or best model based on use_ensemble
    
    Returns:
        dict with prediction details for analysis
    """
    predictor = GPSPredictor(
        use_ensemble=use_ensemble and model_name is None,
        use_tta=use_tta
    )
    
    pil_image = Image.fromarray(image)
    
    results = {
        'config': {
            'use_ensemble': use_ensemble,
            'use_tta': use_tta,
            'model_name': model_name
        }
    }
    
    if model_name:
        # Single model prediction
        if model_name in predictor.models:
            image_tensor = predictor.base_transform(pil_image).unsqueeze(0).to(predictor.device)
            with torch.no_grad():
                normalized_pred = predictor.models[model_name](image_tensor).cpu().numpy().flatten()
            gps_pred = predictor._denormalize_gps(normalized_pred)
            results['prediction'] = np.array(gps_pred, dtype=np.float32)
        else:
            raise ValueError(f"Model {model_name} not found. Available: {list(predictor.models.keys())}")
    else:
        # Use predictor with configured ensemble/tta settings
        if use_tta:
            normalized_pred, uncertainty = predictor._predict_with_tta(pil_image)
            results['uncertainty'] = uncertainty
        else:
            image_tensor = predictor.base_transform(pil_image).unsqueeze(0).to(predictor.device)
            if use_ensemble:
                normalized_pred = np.zeros(2)
                total_weight = 0
                for name, model in predictor.models.items():
                    with torch.no_grad():
                        pred = model(image_tensor).cpu().numpy().flatten()
                    weight = predictor.model_weights.get(name, 1.0)
                    normalized_pred += pred * weight
                    total_weight += weight
                normalized_pred /= total_weight
            else:
                model = predictor.models.get('efficientnet', list(predictor.models.values())[0])
                with torch.no_grad():
                    normalized_pred = model(image_tensor).cpu().numpy().flatten()
        
        gps_pred = predictor._denormalize_gps(normalized_pred)
        results['prediction'] = np.array(gps_pred, dtype=np.float32)
    
    return results


if __name__ == "__main__":
    # Test the predict_gps function
    print("Testing predict_gps function...")
    
    # Create a dummy image for testing
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    try:
        gps = predict_gps(dummy_image)
        print(f"✓ Prediction: lat={gps[0]:.6f}, lon={gps[1]:.6f}")
        print(f"✓ Shape: {gps.shape}")
        print(f"✓ Dtype: {gps.dtype}")
        print("✓ All checks passed!")
    except Exception as e:
        print(f"✗ Error: {e}")
