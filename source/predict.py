"""
GPS Localization Inference Module

This module provides a simple interface to use the trained GPS localization model
to predict GPS coordinates from images.

Usage:
    from predict import predict_gps
    import numpy as np
    
    # Load an image as RGB numpy array (H, W, 3) with values in [0, 255]
    image = np.array(...)  # shape (H, W, 3), dtype uint8
    
    # Predict GPS coordinates
    gps_coords = predict_gps(image)
    # Returns: np.array([latitude, longitude], dtype=float32)
"""

import torch
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple
from models import EfficientNetGPS


# --- MODEL CONFIGURATION ---
# Hardcoded normalization parameters from training data
# These are computed from the training set only to prevent data leakage
MIN_VAL = np.array([31.261283, 34.801083], dtype=np.float32)  # [lat_min, lon_min]
MAX_VAL = np.array([31.262683, 34.804469], dtype=np.float32)  # [lat_max, lon_max]

# Model checkpoint path
MODEL_CHECKPOINT = "../EfficientNet_gps.pth"

# Image size used during training
IMG_SIZE = (255, 255)


class GPSPredictor:
    """
    GPS Coordinate Predictor using Trained Neural Network
    
    Loads a pre-trained model and provides inference capabilities for
    predicting GPS coordinates from RGB images.
    
    The predictor handles:
    - Image preprocessing (resizing, normalization)
    - Model loading and inference
    - Output denormalization to real GPS coordinates
    """
    
    def __init__(self, 
                 checkpoint_path: str = MODEL_CHECKPOINT,
                 min_val: np.ndarray = MIN_VAL,
                 max_val: np.ndarray = MAX_VAL,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the GPS Predictor
        
        Args:
            checkpoint_path (str): Path to the trained model checkpoint
            min_val (np.ndarray): Minimum values for denormalization [lat_min, lon_min]
            max_val (np.ndarray): Maximum values for denormalization [lat_max, lon_max]
            device (str): Device to run inference on ('cuda' or 'cpu')
            
        Raises:
            FileNotFoundError: If checkpoint file does not exist
        """
        self.checkpoint_path = checkpoint_path
        self.min_val = min_val
        self.max_val = max_val
        self.device = torch.device(device)
        
        # Check if checkpoint exists
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
        
        # Initialize model
        self.model = EfficientNetGPS()
        self.model.to(self.device)
        
        # Load pretrained weights
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        
        # Set to evaluation mode
        self.model.eval()
        print(f"✓ Model loaded from {checkpoint_path}")
        print(f"✓ Running on device: {self.device}")
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for model input
        
        Steps:
            1. Validate input format
            2. Resize to (224, 224)
            3. Transpose from (H, W, 3) to (1, 3, H, W) for batch processing
            4. Normalize pixel values from [0, 255] to [0, 1]
            5. Convert to PyTorch tensor
        
        Args:
            image (np.ndarray): Input image with shape (H, W, 3), dtype uint8, values [0, 255]
            
        Returns:
            torch.Tensor: Preprocessed image with shape (1, 3, 224, 224)
            
        Raises:
            ValueError: If input format is incorrect
        """
        # Validate input
        if not isinstance(image, np.ndarray):
            raise ValueError(f"Expected np.ndarray, got {type(image)}")
        
        if image.dtype != np.uint8:
            raise ValueError(f"Expected dtype uint8, got {image.dtype}")
        
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected shape (H, W, 3), got {image.shape}")
        
        if image.min() < 0 or image.max() > 255:
            raise ValueError(f"Expected pixel values in [0, 255], got [{image.min()}, {image.max()}]")
        
        # Resize to expected input size
        image_resized = cv2.resize(image, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
        
        # Transpose from (H, W, C) to (C, H, W) for PyTorch
        image_transposed = np.transpose(image_resized, (2, 0, 1))
        
        # Normalize pixel values from [0, 255] to [0, 1]
        image_normalized = image_transposed.astype(np.float32) / 255.0
        
        # Add batch dimension: (C, H, W) -> (1, C, H, W)
        image_batch = np.expand_dims(image_normalized, axis=0)
        
        # Convert to PyTorch tensor
        tensor = torch.tensor(image_batch, dtype=torch.float32, device=self.device)
        
        return tensor
    
    def _denormalize_gps(self, normalized_gps: np.ndarray) -> np.ndarray:
        """
        Denormalize GPS coordinates from [0, 1] range to real values
        
        The model outputs coordinates normalized to [0, 1] range based on
        training data statistics. This method reverses that normalization.
        
        Formula: real_gps = normalized_gps * (max_val - min_val) + min_val
        
        Args:
            normalized_gps (np.ndarray): Normalized coordinates, shape (2,), values in [0, 1]
            
        Returns:
            np.ndarray: Real GPS coordinates [latitude, longitude], dtype float32
        """
        range_val = self.max_val - self.min_val
        real_gps = normalized_gps * range_val + self.min_val
        return real_gps.astype(np.float32)
    
    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Predict GPS coordinates from an image
        
        This is the main prediction method. It handles the full pipeline:
        preprocessing, inference, and denormalization.
        
        Args:
            image (np.ndarray): Input RGB image
                - Shape: (H, W, 3)
                - Dtype: uint8
                - Values: [0, 255]
        
        Returns:
            np.ndarray: Predicted GPS coordinates
                - Shape: (2,)
                - Dtype: float32
                - Format: [latitude, longitude]
                - Values: Real GPS coordinates (not normalized)
        """
        # Preprocess image
        tensor = self._preprocess_image(image)
        
        # Run inference
        with torch.no_grad():
            normalized_output = self.model(tensor)
        
        # Move to CPU and convert to numpy
        normalized_gps = normalized_output.cpu().numpy()[0]  # Remove batch dimension
        
        # Denormalize to real GPS coordinates
        real_gps = self._denormalize_gps(normalized_gps)
        
        return real_gps


# Global predictor instance (lazy-loaded)
_predictor = None


def predict_gps(image: np.ndarray) -> np.ndarray:
    """
    Predict GPS latitude and longitude from a single RGB image.
    
    This is the main inference function. It uses a pre-trained deep learning model
    to estimate the GPS coordinates where a photo was taken.
    
    IMPORTANT INPUT REQUIREMENTS:
    - Type: numpy.ndarray
    - Shape: (H, W, 3) where H and W are image dimensions
    - Channel order: RGB (NOT BGR)
    - Dtype: uint8
    - Value range: [0, 255]
    
    OUTPUT SPECIFICATION:
    - Type: numpy.ndarray
    - Shape: (2,)
    - Dtype: float32
    - Format: [latitude, longitude]
    - Values: Real GPS coordinates (not normalized)
    
    Args:
        image (np.ndarray): Input RGB image with specifications above
        
    Returns:
        np.ndarray: Predicted GPS coordinates [latitude, longitude]
        
    Raises:
        ValueError: If input format is incorrect
        FileNotFoundError: If model checkpoint is not found
        RuntimeError: If CUDA is not available when expected
        
    Example:
        >>> import numpy as np
        >>> from predict import predict_gps
        >>> 
        >>> # Load or create an image (H, W, 3) with values 0-255
        >>> image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        >>> 
        >>> # Predict GPS coordinates
        >>> gps = predict_gps(image)
        >>> print(f"Latitude: {gps[0]:.6f}, Longitude: {gps[1]:.6f}")
        Latitude: 31.262000, Longitude: 34.803000
    """
    global _predictor
    
    # Lazy-load predictor on first use
    if _predictor is None:
        _predictor = GPSPredictor(
            checkpoint_path=MODEL_CHECKPOINT,
            min_val=MIN_VAL,
            max_val=MAX_VAL
        )
    
    # Run prediction
    return _predictor.predict(image)


def predict_gps_from_path(image_path: str) -> np.ndarray:
    """
    Predict GPS latitude and longitude from an image file path.
    
    This function loads an image from disk, validates it, and predicts GPS coordinates
    using the trained model. It handles various image formats including JPEG, PNG, etc.
    
    INPUT SPECIFICATION:
    - Type: str (file path)
    - File must exist and be a valid image file
    - Supported formats: JPEG, PNG, TIFF, HEIC, BMP, etc.
    
    OUTPUT SPECIFICATION:
    - Type: numpy.ndarray
    - Shape: (2,)
    - Dtype: float32
    - Format: [latitude, longitude]
    - Values: Real GPS coordinates
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        np.ndarray: Predicted GPS coordinates [latitude, longitude]
        
    Raises:
        FileNotFoundError: If the image file does not exist
        ValueError: If the image cannot be loaded or is invalid
        
    Example:
        >>> from predict import predict_gps_from_path
        >>> 
        >>> gps = predict_gps_from_path("path/to/photo.jpg")
        >>> print(f"Latitude: {gps[0]:.6f}, Longitude: {gps[1]:.6f}")
        Latitude: 31.262000, Longitude: 34.803000
    """
    # Check if file exists
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Load image using OpenCV
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Convert BGR to RGB (OpenCV loads in BGR format by default)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # Ensure correct dtype
    if image_rgb.dtype != np.uint8:
        raise ValueError(f"Image has unexpected dtype {image_rgb.dtype}, expected uint8")
    
    # Predict GPS coordinates
    return predict_gps(image_rgb)


if __name__ == "__main__":
    """
    Simple test script demonstrating usage of the predict_gps and predict_gps_from_path functions
    """
    import cv2
    import tempfile
    
    print("="*60)
    print("GPS Prediction Test")
    print("="*60)
    
    # Test 1: Create a synthetic image
    print("\n1. Testing predict_gps with synthetic random image...")
    test_image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    try:
        gps = predict_gps(test_image)
        print(f"   ✓ Prediction successful!")
        print(f"   Latitude:  {gps[0]:.6f}")
        print(f"   Longitude: {gps[1]:.6f}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test 2: Test input validation for predict_gps
    print("\n2. Testing predict_gps input validation...")
    
    # Wrong shape
    try:
        bad_image = np.zeros((224, 224), dtype=np.uint8)  # Missing channel dimension
        predict_gps(bad_image)
        print("   ✗ Should have rejected image with wrong shape")
    except ValueError as e:
        print(f"   ✓ Correctly rejected wrong shape: {str(e)[:50]}...")
    
    # Wrong dtype
    try:
        bad_image = np.zeros((224, 224, 3), dtype=np.float32)
        predict_gps(bad_image)
        print("   ✗ Should have rejected image with wrong dtype")
    except ValueError as e:
        print(f"   ✓ Correctly rejected wrong dtype: {str(e)[:50]}...")
    
    # Wrong value range
    try:
        bad_image = np.zeros((224, 224, 3), dtype=np.uint8) + 256  # Out of range
        predict_gps(bad_image)
        print("   ✗ Should have rejected image with out-of-range values")
    except ValueError as e:
        print(f"   ✓ Correctly rejected out-of-range values: {str(e)[:50]}...")
    
    # Test 3: Test predict_gps_from_path
    print("\n3. Testing predict_gps_from_path with temporary image file...")
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        tmp_path = tmp.name
        test_img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        cv2.imwrite(tmp_path, cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR))
        
        try:
            gps = predict_gps_from_path(tmp_path)
            print(f"   ✓ Prediction successful!")
            print(f"   Latitude:  {gps[0]:.6f}")
            print(f"   Longitude: {gps[1]:.6f}")
        except Exception as e:
            print(f"   ✗ Error: {e}")
        finally:
            Path(tmp_path).unlink()  # Clean up
    
    # Test 4: Test error handling for predict_gps_from_path
    print("\n4. Testing predict_gps_from_path error handling...")
    
    # Non-existent file
    try:
        predict_gps_from_path("/non/existent/path.jpg")
        print("   ✗ Should have rejected non-existent file")
    except FileNotFoundError as e:
        print(f"   ✓ Correctly rejected non-existent file: {str(e)[:50]}...")
    
    print("\n" + "="*60)
    print("Test completed!")
    print("="*60)
