"""
Inference module for GPS prediction from single images.
Provides the predict_gps() function required for project submission.
"""

import torch
import numpy as np
from torchvision import transforms
from models import EfficientNetGPS  # or whichever model performed best
from pathlib import Path

class GPSPredictor:
    """
    Singleton predictor for GPS localization.
    Loads model once and reuses for multiple predictions.
    """
    
    def __init__(self, model_path: str, min_val_path: str, max_val_path: str):
        """
        Initialize predictor with trained model and normalization params.
        
        Args:
            model_path: Path to saved model checkpoint (.pth file)
            min_val_path: Path to saved min_val.npy
            max_val_path: Path to saved max_val.npy
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        self.model = EfficientNetGPS().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # Load normalization parameters
        self.min_val = torch.tensor(np.load(min_val_path), dtype=torch.float32, device=self.device)
        self.max_val = torch.tensor(np.load(max_val_path), dtype=torch.float32, device=self.device)
        
        # Preprocessing pipeline (matches training validation transforms)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"✓ GPS Predictor initialized on {self.device}")
        print(f"✓ Model loaded from {model_path}")
        print(f"✓ GPS range: lat [{self.min_val[0]:.6f}, {self.max_val[0]:.6f}], "
              f"lon [{self.min_val[1]:.6f}, {self.max_val[1]:.6f}]")
    
    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Predict GPS coordinates from a single RGB image.
        
        Args:
            image (np.ndarray): RGB image of shape (H, W, 3), dtype uint8, range [0, 255]
        
        Returns:
            np.ndarray: GPS coordinates [latitude, longitude], dtype float32
        """
        # Validate input
        assert isinstance(image, np.ndarray), "Input must be numpy array"
        assert image.ndim == 3, f"Expected 3D array (H, W, 3), got shape {image.shape}"
        assert image.shape[2] == 3, f"Expected RGB image (3 channels), got {image.shape[2]}"
        assert image.dtype == np.uint8, f"Expected uint8 dtype, got {image.dtype}"
        
        # Preprocess image
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)  # (1, 3, 224, 224)
        
        # Run inference
        with torch.no_grad():
            pred_norm = self.model(img_tensor)  # (1, 2) in [0, 1]
        
        # Denormalize: pred_real = pred_norm * (max - min) + min
        pred_real = pred_norm.cpu() * (self.max_val.cpu() - self.min_val.cpu()) + self.min_val.cpu()
        
        return pred_real.numpy().squeeze().astype(np.float32)


# Global predictor instance (initialized once)
_predictor = None


def predict_gps(image: np.ndarray) -> np.ndarray:
    """
    Predict GPS latitude and longitude from a single RGB image.
    
    This is the main function required for project submission.
    
    Args:
        image (np.ndarray): RGB image
            - Shape: (H, W, 3)
            - Channel order: RGB (NOT BGR)
            - Dtype: uint8
            - Value range: [0, 255]
    
    Returns:
        np.ndarray: GPS coordinates
            - Shape: (2,)
            - Dtype: float32
            - Format: [latitude, longitude]
            - Example: np.array([31.262345, 34.803210], dtype=np.float32)
    """
    global _predictor
    
    # Initialize predictor on first call
    if _predictor is None:
        # Get paths relative to this file
        base_dir = Path(__file__).parent
        model_path = base_dir / "EfficientNet_gps.pth"
        min_val_path = base_dir / "min_val.npy"
        max_val_path = base_dir / "max_val.npy"
        
        _predictor = GPSPredictor(
            model_path=str(model_path),
            min_val_path=str(min_val_path),
            max_val_path=str(max_val_path)
        )
    
    return _predictor.predict(image)


# Example usage
if __name__ == "__main__":
    # Test with random image
    test_image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    
    prediction = predict_gps(test_image)
    
    print(f"\nPrediction: {prediction}")
    print(f"Type: {type(prediction)}")
    print(f"Shape: {prediction.shape}")
    print(f"Dtype: {prediction.dtype}")
    print(f"Latitude: {prediction[0]:.6f}")
    print(f"Longitude: {prediction[1]:.6f}")