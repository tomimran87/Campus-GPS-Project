# GPS Localization Source Package
from .core import (
    ResNetGPS, EfficientNetGPS, ConvNextGPS,
    GPSResNet18, GPSEfficientNet, GPSConvNeXt,  # Aliases
    get_model,
    HaversineLoss,
    GPSDataManager,
    GPSAugmentation, AugmentedGPSDataset,
    GPSMetrics,
    BaseGPSModel,
)

__all__ = [
    'ResNetGPS', 'EfficientNetGPS', 'ConvNextGPS',
    'GPSResNet18', 'GPSEfficientNet', 'GPSConvNeXt',
    'get_model',
    'HaversineLoss',
    'GPSDataManager',
    'GPSAugmentation', 'AugmentedGPSDataset',
    'GPSMetrics',
    'BaseGPSModel',
]
