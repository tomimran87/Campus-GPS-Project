# Core modules for GPS localization

# Models
from .models import ResNetGPS, EfficientNetGPS, ConvNextGPS, TanhTo01

# Loss
from .loss import HaversineLoss

# Data
from .data_loader import GPSDataManager

# Augmentation
from .augmentation import GPSAugmentation, AugmentedGPSDataset

# Metrics
from .metrics import GPSMetrics

# Base model
from .base_model import BaseGPSModel


def get_model(model_name):
    """
    Factory function to get a model by name.
    
    Args:
        model_name (str): One of 'resnet18', 'resnet', 'efficientnet', 'convnext'
        
    Returns:
        nn.Module: Instantiated GPS model
    """
    models_dict = {
        'resnet18': ResNetGPS,
        'resnet': ResNetGPS,
        'efficientnet': EfficientNetGPS,
        'convnext': ConvNextGPS,
    }
    model_name = model_name.lower()
    if model_name not in models_dict:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models_dict.keys())}")
    return models_dict[model_name]()


# Convenience aliases
GPSResNet18 = ResNetGPS
GPSEfficientNet = EfficientNetGPS
GPSConvNeXt = ConvNextGPS


__all__ = [
    # Models
    'ResNetGPS', 'EfficientNetGPS', 'ConvNextGPS',
    'GPSResNet18', 'GPSEfficientNet', 'GPSConvNeXt',  # Aliases
    'get_model',
    'TanhTo01',
    'BaseGPSModel',
    # Loss
    'HaversineLoss',
    # Data
    'GPSDataManager',
    # Augmentation
    'GPSAugmentation', 'AugmentedGPSDataset',
    # Metrics
    'GPSMetrics',
]
