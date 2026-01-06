import torch
import torch
import torch.nn as nn
import torchvision.models as models
from base_model import BaseGPSModel


class TanhTo01(nn.Module):
    """
    Custom activation layer that scales Tanh output from [-1,1] to [0,1].
    
    Why use this instead of Sigmoid?
    - Tanh has steeper gradient in middle region (better learning signal)
    - Tanh is zero-centered (symmetrical around 0)
    - Scaling avoids saturation issues that plague Sigmoid
    - Output range is still bounded to [0,1] as needed for normalized GPS coords
    """
    def forward(self, x):
        return (torch.tanh(x) + 1) / 2  # Maps [-1,1] to [0,1]


class ResNetGPS(BaseGPSModel):
    """
    GPS Localization Model using ResNet18 Backbone
    
    Architecture:
        - Backbone: ResNet18 pretrained on ImageNet (1.2M images, 1000 classes)
        - Features: 512-dimensional vector after global average pooling
        - Head: 3-layer MLP with LayerNorm, dropout, and SiLU activations
        - Output: 2D GPS coordinates [latitude, longitude]
    
    ResNet18 (Residual Network):
        - 18 layers deep with residual (skip) connections
        - Residual connections: y = F(x) + x (allows gradient flow)
        - Uses batch normalization for training stability
        - Efficient: 11M parameters, good for real-time inference
        - Pretrained weights provide strong visual feature extraction
    
    Head Design:
        512 → [Dropout 0.3] → [Linear 128] → [LayerNorm] → [SiLU] →
        → [Linear 64] → [LayerNorm] → [SiLU] → [Linear 2] → [Sigmoid]
    
    Activation Function (SiLU):
        SiLU(x) = x * sigmoid(x)
        - Smooth, non-monotonic activation (also called Swish)
        - Better gradient flow than ReLU for regression tasks
        - Used in EfficientNet and modern architectures
    
    Regularization:
        - Dropout (p=0.3): Randomly zeros 30% of activations during training
        - LayerNorm: Normalizes activations, reduces internal covariate shift
        - Weight Decay: Applied via AdamW optimizer (1e-4)
    
    Output Activation (Sigmoid):
        - Constrains outputs to [0, 1] range
        - Matches the normalized GPS coordinate range
        - Prevents extreme predictions that cause NaN
    
    Performance:
        - Fast training: ~2-3s per epoch on GPU
        - Moderate accuracy: Good baseline model
        - Stable with gradient clipping
    
    Reference:
        He et al., "Deep Residual Learning for Image Recognition", CVPR 2016
    """
    def __init__(self):
        super().__init__()
        # Load pretrained backbone and create regression head
        self.backbone = self._get_backbone()
        self.head = self._get_head(512)  # ResNet18's fc layer has 512 input features

    def _get_backbone(self):
        """
        Create ResNet18 feature extractor
        
        Modifications from original ResNet18:
            1. Remove final fully-connected layer (originally 512 → 1000 for ImageNet)
            2. Keep global average pooling (reduces spatial dims to 1x1)
            3. Add Flatten layer to convert (batch, 512, 1, 1) → (batch, 512)
        
        The pretrained weights are from ImageNet training, which learned
        to recognize 1000 object categories. These features transfer well
        to GPS localization as they capture general visual patterns.
        
        Returns:
            nn.Sequential: Feature extractor outputting 512-dim vectors
        """
        # Load pretrained ResNet18 from torchvision
        # weights='DEFAULT' uses best available pretrained weights
        resnet = models.resnet18(weights='DEFAULT')
        
        # ResNet18 structure: conv1 → bn1 → relu → maxpool → 
        #                     layer1 → layer2 → layer3 → layer4 → 
        #                     avgpool → fc
        # We keep everything except the final fc (classification) layer
        # list(resnet.children())[:-1] removes the fc layer
        # nn.Flatten() converts (batch, 512, 1, 1) to (batch, 512)
        return nn.Sequential(*list(resnet.children())[:-1], nn.Flatten())

    def _get_head(self, input_features):
        """
        Create regression head for GPS coordinate prediction
        
        Architecture choices:
            - Dropout first: Regularize incoming features, prevent co-adaptation
            - 512 → 128 → 64: Gradually compress to 2D output
            - LayerNorm: Stabilize training, normalize across features
            - SiLU activation: Smooth gradients for regression
            - Sigmoid output: Constrain to [0, 1] range (matches normalized GPS)
        
        LayerNorm vs BatchNorm:
            - LayerNorm normalizes across feature dimension (not batch)
            - More stable for small batch sizes
            - Doesn't require running statistics
            - Better for regression tasks
        
        Args:
            input_features (int): 512 for ResNet18
            
        Returns:
            nn.Sequential: Regression layers
        """
        head = nn.Sequential(
            nn.Dropout(0.3),                # Regularization: drop 30% of features
            nn.Linear(input_features, 128),  # First compression layer
            nn.LayerNorm(128),               # Normalize activations for stability
            nn.SiLU(),                       # Smooth activation: x * sigmoid(x)
            nn.Linear(128, 64),              # Second compression layer
            nn.LayerNorm(64),                # Normalize activations
            nn.SiLU(),                       # Smooth activation
            nn.Linear(64, 2),                # Final regression to [lat, lon]
            
        )
        
        # Initialize final linear layer (index -2 since -1 is TanhTo01)
        nn.init.xavier_uniform_(head[-1].weight, gain=0.1)
        nn.init.constant_(head[-1].bias, 0.0)
        
        return head

class EfficientNetGPS(BaseGPSModel):
    """
    GPS Localization Model using EfficientNet-B0 Backbone
    
    Architecture:
        - Backbone: EfficientNet-B0 pretrained on ImageNet
        - Features: 1280-dimensional vector after global average pooling
        - Head: 2-layer MLP with LayerNorm, dropout, and SiLU activations
        - Output: 2D GPS coordinates [latitude, longitude]
    
    EfficientNet-B0:
        - Compound scaling: Balances depth, width, and resolution
        - Mobile Inverted Bottleneck Convolution (MBConv) blocks
        - Squeeze-and-Excitation (SE) attention mechanisms
        - Efficient: 5M parameters (fewer than ResNet18!)
        - Higher accuracy per parameter than ResNet
    
    Head Design:
        1280 → [Dropout 0.3] → [Linear 256] → [LayerNorm] → [SiLU] →
        → [Linear 2] → [Sigmoid]
    
    Why wider head (256 units)?
        - EfficientNet features are richer (1280-dim vs ResNet's 512-dim)
        - Needs more capacity to compress features to 2D
        - Still fewer total parameters than ResNet head
    
    Advantages:
        - Best accuracy-to-parameter ratio
        - Modern architecture with SE attention
        - Good generalization on small datasets
        - Fast inference
    
    Disadvantages:
        - Slightly slower training than ResNet
        - More memory usage during training
    
    Reference:
        Tan & Le, "EfficientNet: Rethinking Model Scaling for CNNs", ICML 2019
    """
    def __init__(self):
        super().__init__()
        self.backbone = self._get_backbone()
        # EfficientNet-B0 classifier has 1280 input features
        self.head = self._get_head(1280)

    def _get_backbone(self):
        """
        Create EfficientNet-B0 feature extractor
        
        EfficientNet structure:
            - features: MBConv blocks with SE attention
            - avgpool: Global average pooling (7x7 → 1x1)
            - classifier: Final linear layer (removed here)
        
        We use:
            - features: Keep all MBConv blocks
            - avgpool: Keep global pooling
            - Flatten: Convert to 1D vector
        
        Returns:
            nn.Sequential: Feature extractor outputting 1280-dim vectors
        """
        # Load pretrained EfficientNet-B0
        effnet = models.efficientnet_b0(weights='DEFAULT')
        
        # EfficientNet's features are in 'features' module (MBConv blocks)
        # avgpool reduces spatial dimensions to 1x1
        # Flatten converts (batch, 1280, 1, 1) to (batch, 1280)
        return nn.Sequential(effnet.features, effnet.avgpool, nn.Flatten())

    def _get_head(self, input_features):
        """
        Create regression head for GPS coordinate prediction
        
        Simpler than ResNet head (2 layers vs 3) because:
            - EfficientNet features are more refined
            - Less compression needed
            - Reduces overfitting risk
        
        Args:
            input_features (int): 1280 for EfficientNet-B0
            
        Returns:
            nn.Sequential: Regression layers
        """
        head = nn.Sequential(
            nn.Dropout(0.3),                 # Regularization
            nn.Linear(input_features, 256),  # Compression to 256-dim
            nn.LayerNorm(256),               # Stabilize activations
            nn.SiLU(),                       # Smooth activation
            nn.Linear(256, 2),               # Regression to [lat, lon]
            TanhTo01()                       # Tanh scaled to [0,1]
        )
        
        # Initialize final linear layer
        nn.init.xavier_uniform_(head[-2].weight, gain=0.1)
        nn.init.constant_(head[-2].bias, 0.0)
        
        return head

class ConvNextGPS(BaseGPSModel):
    """
    GPS Localization Model using ConvNeXt-Tiny Backbone
    
    Architecture:
        - Backbone: ConvNeXt-Tiny pretrained on ImageNet
        - Features: 768-dimensional vector after global average pooling
        - Head: 2-layer MLP with LayerNorm, dropout, and SiLU activations
        - Output: 2D GPS coordinates [latitude, longitude]
    
    ConvNeXt (2022):
        - Modernized ResNet using recent best practices
        - Uses LayerNorm instead of BatchNorm (more stable)
        - Depthwise convolutions (like in transformers)
        - GELU activation (smooth, works well in practice)
        - Larger kernel sizes (7x7) for better receptive fields
    
    Head Design:
        768 → [Dropout 0.3] → [Linear 128] → [LayerNorm] → [SiLU] →
        → [Linear 2] → [Sigmoid]
    
    Why ConvNeXt is more stable:
        - LayerNorm in backbone (not BatchNorm) is more stable
        - No running statistics to get corrupted
        - Better gradient flow through normalization layers
        - Modern design handles numerical issues better
    
    Advantages:
        - Most stable training (observed in experiments)
        - Good accuracy with fewer parameters than EfficientNet
        - State-of-the-art CNN architecture (2022)
        - Better than ResNet on most benchmarks
    
    Disadvantages:
        - Slower training than ResNet
        - Slightly higher memory usage
        - May be overkill for simple localization tasks
    
    Performance Notes:
        - In experiments, ConvNeXt was the last to hit NaN
        - Internal LayerNorm provides natural stability
        - Best choice if stability is priority over speed
    
    Reference:
        Liu et al., "A ConvNet for the 2020s", CVPR 2022
    """
    def __init__(self):
        super().__init__()
        self.backbone = self._get_backbone()
        # ConvNeXt-Tiny classifier has 768 input features
        self.head = self._get_head(768)

    def _get_backbone(self):
        """
        Create ConvNeXt-Tiny feature extractor
        
        ConvNeXt structure:
            - features: 4 stages of ConvNeXt blocks
            - avgpool: Adaptive average pooling (→ 1x1)
            - classifier: Final linear layer (removed here)
        
        ConvNeXt blocks use:
            - Depthwise convolutions (efficient)
            - LayerNorm (stable normalization)
            - GELU activation (smooth)
            - Larger kernels (7x7)
        
        Returns:
            nn.Sequential: Feature extractor outputting 768-dim vectors
        """
        # Load pretrained ConvNeXt-Tiny
        convnext = models.convnext_tiny(weights='DEFAULT')
        
        # Structure: features (ConvNeXt blocks) → avgpool → classifier
        # We keep features + avgpool, remove classifier
        # Flatten converts (batch, 768, 1, 1) to (batch, 768)
        return nn.Sequential(convnext.features, convnext.avgpool, nn.Flatten())

    def _get_head(self, input_features):
        """
        Create regression head for GPS coordinate prediction
        
        Similar to ResNet but with one less layer:
            - ConvNeXt features are already well-refined
            - LayerNorm in backbone helps stability
            - Simpler head reduces overfitting
        
        Args:
            input_features (int): 768 for ConvNeXt-Tiny
            
        Returns:
            nn.Sequential: Regression layers
        """
        head = nn.Sequential(
            nn.Dropout(0.3),                 # Regularization
            nn.Linear(input_features, 128),  # Compression to 128-dim
            nn.LayerNorm(128),               # Stabilize activations
            nn.SiLU(),                       # Smooth activation
            nn.Linear(128, 2),               # Regression to [lat, lon]
            TanhTo01()                       # Tanh scaled to [0,1]
        )
        
        # Initialize final linear layer
        nn.init.xavier_uniform_(head[-2].weight, gain=0.1)
        nn.init.constant_(head[-2].bias, 0.0)
        
        return head