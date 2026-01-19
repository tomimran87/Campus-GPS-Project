import torch.nn as nn
from abc import ABC, abstractmethod

class BaseGPSModel(nn.Module, ABC):
    """
    Abstract Base Class for GPS Regression Models
    
    This class defines the interface that all GPS localization models must implement.
    It follows the Template Method design pattern, where the forward pass is standardized
    but subclasses must implement their own backbone and head architectures.
    
    Architecture Pattern:
        Input Image (3, 224, 224)
            ↓
        Backbone (Feature Extractor)
            ↓
        Features (N-dimensional vector)
            ↓
        Head (Regression Layers)
            ↓
        Output (2D coordinates: latitude, longitude)
    
    The backbone extracts visual features from images (typically a pretrained CNN),
    while the head regresses these features to GPS coordinates.
    
    Subclasses must implement:
        - _get_backbone(): Return feature extractor (e.g., ResNet, EfficientNet)
        - _get_head(input_features): Return regression head (Linear layers)
    
    Example Usage:
        class MyGPSModel(BaseGPSModel):
            def __init__(self):
                super().__init__()
                self.backbone = self._get_backbone()
                self.head = self._get_head(512)  # 512 = backbone output size
            
            def _get_backbone(self):
                return some_pretrained_model
            
            def _get_head(self, input_features):
                return nn.Linear(input_features, 2)
    
    Inherits from:
        nn.Module: PyTorch's base class for all neural network modules
        ABC: Python's Abstract Base Class for interface enforcement
    """
    def __init__(self):
        super(BaseGPSModel, self).__init__()
        
    @abstractmethod
    def _get_backbone(self):
        """
        Create and return the feature extraction backbone
        
        The backbone is typically a pretrained CNN (ResNet, EfficientNet, etc.)
        with the classification head removed. It should:
            1. Take images of shape (batch, 3, 224, 224)
            2. Extract spatial features through convolutions
            3. Apply global pooling to get fixed-size vector
            4. Return features of shape (batch, feature_dim)
        
        Returns:
            nn.Module: Feature extractor network
            
        Example:
            resnet = models.resnet18(weights='DEFAULT')
            backbone = nn.Sequential(*list(resnet.children())[:-1], nn.Flatten())
            return backbone
        """
        pass

    @abstractmethod
    def _get_head(self, input_features):
        """
        Create and return the regression head
        
        The head maps extracted features to GPS coordinates. It should:
            1. Take features of size input_features
            2. Apply Linear layers with activations
            3. Output 2D tensor: [latitude, longitude] in normalized [0,1] range
        
        Args:
            input_features (int): Dimension of backbone output features
            
        Returns:
            nn.Module: Regression head network
            
        Example:
            return nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(input_features, 128),
                nn.ReLU(),
                nn.Linear(128, 2)
            )
        """
        pass

    def forward(self, x):
        """
        Standard forward pass for all GPS models
        
        This method is automatically inherited by all subclasses and should
        not be overridden unless you have a specific reason.
        
        Flow:
            x: Input images (batch, 3, 224, 224)
                ↓
            self.backbone(x): Extract features (batch, feature_dim)
                ↓
            self.head(features): Regress to GPS (batch, 2)
                ↓
            Output: GPS coordinates [lat, lon] (batch, 2)
        
        Args:
            x (torch.Tensor): Batch of images, shape (batch_size, 3, 224, 224)
            
        Returns:
            torch.Tensor: Predicted GPS coordinates, shape (batch_size, 2)
                         Values are in normalized [0, 1] range
        """
        # Extract visual features using the backbone
        x = self.backbone(x)
        
        # Regress features to GPS coordinates using the head
        return self.head(x)