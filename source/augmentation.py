"""
Data augmentation for GPS image localization.

This module provides robust data augmentation tailored for outdoor campus GPS localization.
The augmentation strategies address real-world challenges:
1. Varying lighting conditions (different times of day, weather)
2. Different camera sensors (student smartphones vary widely)
3. Orientation differences (students hold phones at different angles)
4. Perspective variations (height, tilt)
5. Color shifts (camera post-processing differences)

These transformations make the model generalize better without requiring thousands
of images from each location at different times/conditions.
"""

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np


class GPSAugmentation:
    """
    Provides train and validation transforms for GPS localization.
    
    Philosophy:
    - Train: Aggressive augmentation to simulate real-world variations
    - Validation: Only normalization (no augmentation) to measure true performance
    
    All transforms preserve the GPS coordinate (label doesn't change with augmentation).
    """
    
    def __init__(self, image_size: int = 224):
        """
        Initialize augmentation pipelines.
        
        Args:
            image_size (int): Target image size (ResNet/EfficientNet/ConvNeXt use 224)
        """
        self.image_size = image_size
        
        # ImageNet statistics: CRITICAL for pretrained models!
        # Pretrained backbones were trained with these exact values
        # Using different normalization will cause severe accuracy drop
        self.mean = [0.485, 0.456, 0.406]  # ImageNet RGB means
        self.std = [0.229, 0.224, 0.225]   # ImageNet RGB std devs
        
      # Training augmentation pipeline
        # AGGRESSIVE augmentation - maximize data diversity
        # GPS stays the same regardless of image orientation/lighting
        self.train_transforms = transforms.Compose([
            transforms.ToPILImage(),
            
            # Random crop and resize - different perspectives of same location
            transforms.RandomResizedCrop(
                self.image_size,
                scale=(0.85, 1.0),  # Crop 85-100% of image
                ratio=(0.95, 1.05)  # Keep aspect ratio nearly square
            ),
            
            # Strong color jitter - different times of day/weather/cameras
            transforms.ColorJitter(
                brightness=0.3,   # Sunrise vs sunset
                contrast=0.3,     # Hazy vs clear days
                saturation=0.25,  # Camera differences
                hue=0.1           # Color temperature
            ),
            
            # Horizontal flip - location same if you turn around!
            transforms.RandomHorizontalFlip(p=0.5),
            
            # Vertical flip - phone upside down
            transforms.RandomVerticalFlip(p=0.3),
            
            # Small rotation - phone not held level
            transforms.RandomRotation(degrees=15),  # ±15 degrees
            
            # Random grayscale - poor lighting
            transforms.RandomGrayscale(p=0.1),
            
            transforms.ToTensor(),
            
            # Random erasing - occlusions (people/trees blocking)
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
            
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        
        # Validation/test transforms: ONLY normalization, NO augmentation
        # We want to measure true model performance without random variations
        self.val_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
    
    def get_train_transforms(self):
        """Get training transforms (with augmentation)."""
        return self.train_transforms
    
    def get_val_transforms(self):
        """Get validation transforms (no augmentation)."""
        return self.val_transforms


class AugmentedGPSDataset(Dataset):
    """
    GPS dataset with configurable augmentation.
    
    Wraps (X, y) numpy arrays with transform pipeline.
    Applies augmentation to images while keeping GPS coordinates unchanged.
    """
    
    def __init__(self, X: np.ndarray, y: np.ndarray, transform=None):
        """
        Initialize dataset.
        
        Args:
            X (np.ndarray): Images array of shape (N, H, W, 3) in [0, 255]
            y (np.ndarray): GPS coordinates of shape (N, 2) normalized to [0, 1]
            transform (callable, optional): Transform pipeline to apply to images
        """
        self.X = X
        self.y = y
        self.transform = transform
    
    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.X)
    
    def __getitem__(self, idx: int):
        """
        Get single sample.
        
        Args:
            idx (int): Sample index
            
        Returns:
            tuple: (image_tensor, gps_coordinates)
                - image_tensor: Transformed image of shape (3, 224, 224)
                - gps_coordinates: GPS coords of shape (2,) in [0, 1]
        """
        # Get image - handle both float [0,1] and uint8 [0,255] formats
        image = self.X[idx]
        
        # Convert to uint8 [0, 255] for PIL transforms
        if image.dtype == np.float32 or image.dtype == np.float64:
            # Image is in [0, 1], scale to [0, 255]
            image = (image * 255).astype(np.uint8)
        elif image.dtype != np.uint8:
            # Unknown format, assume needs scaling
            image = image.astype(np.uint8)
        
        # Get GPS coordinates
        gps = self.y[idx]
        
        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)
        else:
            # Fallback: just convert to tensor
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()
            if image.max() > 1.0:
                image = image / 255.0
        
        # Convert GPS to tensor
        gps = torch.from_numpy(gps).float()
        
        return image, gps


# Example usage for reference:
if __name__ == "__main__":
    print("GPS Augmentation Module")
    print("=" * 50)
    
    # Create augmentation
    aug = GPSAugmentation(image_size=224)
    
    # Create dummy data
    dummy_X = np.random.randint(0, 255, (100, 224, 224, 3), dtype=np.uint8)
    dummy_y = np.random.rand(100, 2).astype(np.float32)
    
    # Create datasets
    train_dataset = AugmentedGPSDataset(
        dummy_X, dummy_y,
        transform=aug.get_train_transforms()
    )
    val_dataset = AugmentedGPSDataset(
        dummy_X, dummy_y,
        transform=aug.get_val_transforms()
    )
    
    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Val dataset: {len(val_dataset)} samples")
    
    # Show sample
    img, gps = train_dataset[0]
    print(f"\nSample shape: {img.shape}")
    print(f"GPS shape: {gps.shape}")
    print(f"Image range: [{img.min():.3f}, {img.max():.3f}]")
    print(f"GPS coords: {gps.numpy()}")
