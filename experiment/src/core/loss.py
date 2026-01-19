import torch
import torch.nn as nn

class HaversineLoss(nn.Module):
    """
    Haversine Loss Function for GPS Coordinate Regression
    
    Computes the great-circle distance between predicted and target GPS coordinates
    using the Haversine formula. This is more accurate than Euclidean distance for
    latitude/longitude coordinates as it accounts for Earth's spherical geometry.
    
    Mathematical Formula:
        a = sin²(Δlat/2) + cos(lat1) × cos(lat2) × sin²(Δlon/2)
        c = 2 × asin(√a)
        distance = R × c
    
    where R = 6,371,000 meters (Earth's mean radius)
    
    Numerical Stability:
        - Clamps 'a' to [1e-6, 1-1e-6] to prevent sqrt(negative) or sqrt(>1)
        - Clamps sqrt(a) before asin to avoid domain errors [-1, 1]
        - Uses double-clamping strategy to handle floating-point precision errors
        - Clamps final output to prevent overflow
    
    Args:
        min_val (torch.Tensor): Minimum GPS values [min_lat, min_lon] for denormalization
        max_val (torch.Tensor): Maximum GPS values [max_lat, max_lon] for denormalization
        device (torch.device): Device to store normalization parameters
    
    Input:
        pred (torch.Tensor): Normalized predicted coordinates, shape (batch_size, 2)
        target (torch.Tensor): Normalized target coordinates, shape (batch_size, 2)
    
    Returns:
        torch.Tensor: Scalar tensor containing mean Haversine distance in meters
    
    Reference:
        https://en.wikipedia.org/wiki/Haversine_formula
    """
    def __init__(self, min_val, max_val, device):
        super().__init__()
        # Store normalization parameters on device for denormalization
        # These are computed from the training set only (see data_loader.py)
        self.min_val = min_val.to(device)
        self.max_val = max_val.to(device)

    def forward(self, pred, target):
        """
        Compute Haversine distance between predictions and targets
        
        Args:
            pred: Normalized predictions in [0, 1] range
            target: Normalized targets in [0, 1] range
            
        Returns:
            Mean distance in meters across the batch
        """
        # Step 1: Denormalize from [0,1] to original GPS coordinate range
        # Formula: x_real = x_norm * (max - min) + min
        pred_real = pred * (self.max_val - self.min_val) + self.min_val
        target_real = target * (self.max_val - self.min_val) + self.min_val
        
        # Step 2: Convert degrees to radians
        # All trigonometric functions in PyTorch expect radians
        rad_pred = torch.deg2rad(pred_real)
        rad_target = torch.deg2rad(target_real)
        
        # Step 3: Extract latitude and longitude components
        # Convention: [:, 0] = latitude, [:, 1] = longitude
        lat1, lon1 = rad_pred[:, 0], rad_pred[:, 1]
        lat2, lon2 = rad_target[:, 0], rad_target[:, 1]
        
        # Step 4: Compute coordinate differences
        dlat = lat2 - lat1  # Δlat in radians
        dlon = lon2 - lon1  # Δlon in radians
        
        # Step 5: Haversine formula computation
        # a = sin²(Δlat/2) + cos(lat1) × cos(lat2) × sin²(Δlon/2)
        a = torch.sin(dlat / 2)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2)**2
        
        # Clamp only to prevent numerical overflow, not to change small values
        # max=1.0 prevents sqrt(a) > 1 which would make asin undefined
        # No lower bound - tiny values (1e-12) are legitimate for small GPS areas
        a = torch.clamp(a, max=1.0)
        
        # Step 6: Compute sqrt(a) safely
        # Add tiny epsilon to prevent sqrt(0) = 0 causing division issues later
        sqrt_a = torch.sqrt(a + 1e-12)
        
        # Clamp sqrt_a to valid asin domain [-1, 1]
        sqrt_a = torch.clamp(sqrt_a, max=1.0)
        
        # Step 7: Compute angular distance 'c' using inverse sine
        # c = 2 × asin(√a)
        c = 2 * torch.asin(sqrt_a)
        
        # Step 8: Convert to physical distance
        # Earth's mean radius in meters (R = 6,371 km)
        R = 6371000
        dist_meters = R * c
        
        # Step 9: Return mean distance across batch
        return dist_meters.mean()