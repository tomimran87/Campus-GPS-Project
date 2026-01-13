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
    GPS Localization Model using Modified ResNet18 Architecture
    
    MODIFICATIONS FROM STANDARD ResNet18:
        1. Backbone: Use standard ResNet18 pre-trained on ImageNet
        2. Remove: Final classification layer (originally 512 → 1000 classes)
        3. Add: Custom regression head for GPS coordinate prediction
    
    Architecture:
        - Backbone: ResNet18 pretrained on ImageNet (removes final fc layer)
        - Features: 512-dimensional vector after global average pooling  
        - Head: Custom 3-layer regression MLP
        - Output: 2D GPS coordinates [latitude, longitude] in [0,1] range
    
    ResNet18 Backbone Details:
        - 18 convolutional layers with residual connections
        - Pretrained on ImageNet (1.2M images, 1000 classes)
        - Final feature dimension: 512 (after removing classification head)
        - Residual connections: y = F(x) + x (enables deep training)
        - Global Average Pooling reduces spatial dimensions to 1×1
    
    Custom Regression Head Design:
        512 → [Dropout 0.3] → [Linear 128] → [LayerNorm] → [SiLU] →
        → [Linear 64] → [LayerNorm] → [SiLU] → [Linear 2] → [Sigmoid]
        
        - Progressive dimension reduction: 512 → 128 → 64 → 2
        - LayerNorm for training stability
        - SiLU activation for smooth gradients
        - Sigmoid output to constrain GPS coordinates to [0,1]
        - Dropout for regularization
    
    Key Benefits:
        - Leverages ImageNet pretrained features (transfer learning)
        - Custom head specifically designed for GPS regression
        - Stable training with proper normalization
        - Efficient: ~11M parameters total
    
    Reference:
        He et al., "Deep Residual Learning for Image Recognition", CVPR 2016
    """
    def __init__(self):
        super().__init__()
        # Create modified ResNet18: pretrained backbone + custom regression head
        self.backbone = self._get_backbone()
        self.head = self._get_head(512)  # ResNet18 outputs 512 features after GAP

    def _get_backbone(self):
        """
        Create Modified ResNet18 Feature Extractor
        
        STEP 1: Load standard ResNet18 pretrained on ImageNet
        STEP 2: Remove final classification layer (512 → 1000 classes)
        STEP 3: Keep global average pooling + add flatten layer
        
        Original ResNet18 Architecture:
            Input (3×224×224) → Conv1 → BN → ReLU → MaxPool →
            → ResBlock1 → ResBlock2 → ResBlock3 → ResBlock4 →
            → GlobalAvgPool → [REMOVED: Linear(512→1000)] 
        
        Our Modified Backbone:
            Input (3×224×224) → Conv1 → BN → ReLU → MaxPool →
            → ResBlock1 → ResBlock2 → ResBlock3 → ResBlock4 →
            → GlobalAvgPool → Flatten → Output (512-dim vector)
        
        Key Modifications:
            ❌ REMOVED: Final fc layer (classification head for ImageNet)
            ✅ KEPT: All convolutional layers + global average pooling  
            ✅ ADDED: Flatten layer to convert (batch,512,1,1) → (batch,512)
        
        Transfer Learning Benefits:
            - Pretrained weights encode powerful visual features
            - Lower layers detect edges, textures, simple shapes
            - Higher layers detect complex objects, spatial patterns
            - Saves training time vs. training from scratch
        
        Returns:
            nn.Sequential: Modified ResNet18 backbone (input: images, output: 512-dim features)
        """
        # Load standard ResNet18 with ImageNet pretrained weights
        resnet = models.resnet18(weights='DEFAULT')
        
        # Remove the final classification layer but keep everything else:
        # list(resnet.children()) = [conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool, fc]
        # [:-1] removes the fc layer, keeping: [conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool]
        backbone_layers = list(resnet.children())[:-1]
        
        # Add Flatten layer to convert (batch, 512, 1, 1) → (batch, 512)
        backbone_layers.append(nn.Flatten())
        
        return nn.Sequential(*backbone_layers)

    def _get_head(self, input_features):
        """
        Create Custom Regression Head for GPS Coordinate Prediction
        
        INPUT: 512-dimensional features from ResNet18 backbone
        OUTPUT: 2D GPS coordinates [latitude, longitude] in [0,1] range
        
        Custom Head Architecture:
            512 → [Dropout(0.3)] → [Linear(512→128)] → [LayerNorm(128)] → [SiLU] 
                → [Linear(128→64)] → [LayerNorm(64)] → [SiLU]
                → [Linear(64→2)] → [Sigmoid] → [lat, lon]
        
        Design Rationales:
        
        🎯 PROGRESSIVE COMPRESSION: 512 → 128 → 64 → 2
           - Gradually reduces dimensionality to avoid information bottlenecks
           - Each layer learns increasingly specialized GPS-relevant features
        
        🛡️ REGULARIZATION: Dropout(0.3) at input
           - Prevents overfitting by randomly zeroing 30% of backbone features
           - Forces model to not rely on specific feature combinations
        
        📊 NORMALIZATION: LayerNorm after each linear layer
           - Normalizes activations across feature dimension
           - More stable than BatchNorm for small batch sizes
           - Accelerates training convergence
        
        ⚡ ACTIVATION: SiLU (Sigmoid Linear Unit)
           - SiLU(x) = x * sigmoid(x) = x / (1 + e^(-x))
           - Smooth, differentiable (better gradients than ReLU)
           - Non-monotonic (can decrease, unlike ReLU)
           - Proven effective in EfficientNet and Transformers
        
        🎯 OUTPUT: Sigmoid activation
           - Constrains outputs to [0,1] range (matches normalized GPS coords)
           - Prevents extreme predictions that could cause training instability
           - Ensures valid coordinate predictions
        
        🔧 WEIGHT INITIALIZATION: 
           - Xavier uniform for final layer with small gain (0.1)
           - Zero bias initialization for final layer
           - Prevents initial predictions from being too extreme
        
        Args:
            input_features (int): 512 for ResNet18 backbone
            
        Returns:
            nn.Sequential: Custom regression head
        """
        head = nn.Sequential(
            # Input regularization
            nn.Dropout(0.3),                # Drop 30% of backbone features
            
            # First compression layer: 512 → 128
            nn.Linear(input_features, 128),
            nn.LayerNorm(128),               # Normalize across feature dim
            nn.SiLU(),                       # Smooth activation
            
            # Second compression layer: 128 → 64  
            nn.Linear(128, 64),
            nn.LayerNorm(64),               # Normalize across feature dim
            nn.SiLU(),                       # Smooth activation
            
            # Final regression layer: 64 → 2 (lat, lon)
            nn.Linear(64, 2),
            nn.Sigmoid()                     # Constrain to [0,1] range
        )
        
        # Initialize final linear layer with small weights to prevent extreme initial predictions
        final_layer = head[-2]  # -2 because -1 is Sigmoid
        nn.init.xavier_uniform_(final_layer.weight, gain=0.1)
        nn.init.constant_(final_layer.bias, 0.0)
        
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

    
class EfficientNetGPS2(BaseGPSModel):
    """
    EfficientNetV2-B0 GPS Localization Model with Custom Architecture
    
    This model uses EfficientNetV2-B0 as backbone with custom modifications:
    - Removes global average pooling to preserve spatial information
    - Adds channel reduction to make the head manageable
    - Uses scaled sigmoid output activation
    
    Architecture:
        - Backbone: EfficientNetV2-B0 features (without classifier)
        - Channel Reducer: 1280 → 128 channels via 1x1 convolution
        - Head: Flattened features → regression layers → GPS coordinates
    """
    def __init__(self, input_shape=(3, 224, 224)):
        super().__init__()
        self.input_shape = input_shape
        # Initialize backbone and head using the abstract methods
        self.backbone = self._get_backbone()
        
        # Calculate head input features dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            features = self.backbone(dummy)
            head_input_features = features.numel()
        
        self.head = self._get_head(head_input_features)
    
    def _get_backbone(self):
        """
        Create EfficientNetV2-B0 feature extractor with channel reduction
        
        Returns:
            nn.Sequential: Modified EfficientNet backbone
        """
        # Load pretrained EfficientNetV2-B0
        base_model = models.efficientnet_v2_s(weights='DEFAULT')
        
        # Extract features (without classifier and final pooling)
        features = base_model.features
        
        # Add channel reducer to make output manageable
        # EfficientNetV2-S outputs 1280 channels, reduce to 128
        reducer = nn.Conv2d(1280, 128, kernel_size=1)
        
        # Combine features + reducer + flatten
        backbone = nn.Sequential(
            features,      # EfficientNet feature extraction
            reducer,       # Channel reduction 1280→128  
            nn.Flatten()   # Flatten for linear layers
        )
        
        return backbone
    
    def _get_head(self, input_features):
        """
        Create regression head for GPS coordinate prediction
        
        Args:
            input_features (int): Number of flattened features from backbone
            
        Returns:
            nn.Sequential: Regression head
        """
        head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(input_features, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, 2),
            nn.Sigmoid()  # Scale to [0,1] range
        )
        
        # Initialize final layer
        nn.init.xavier_uniform_(head[-2].weight, gain=0.1)
        nn.init.constant_(head[-2].bias, 0.0)
        
        return head