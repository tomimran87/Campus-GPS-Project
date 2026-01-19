import torch
import numpy as np

class GPSMetrics:
    """
    Comprehensive Evaluation Metrics for GPS Localization
    
    Computes various statistics to assess model performance beyond simple
    mean error. Provides a complete picture of:
        - Central tendency: mean, median
        - Spread: standard deviation, min, max
        - Distribution: percentiles (50th, 90th, 95th, 99th)
        - Accuracy thresholds: % predictions within 5m, 10m, 50m
    
    Why multiple metrics?
        - Mean can be misleading with outliers
        - Median is more robust to extreme errors
        - Percentiles show error distribution shape
        - Accuracy thresholds are application-specific requirements
    
    Example:
        - Mean: 20m might seem good
        - But 95th percentile: 200m means 5% of predictions are terrible
        - This matters for safety-critical applications
    
    Usage:
        distances = compute_distances(predictions, targets)
        metrics = GPSMetrics.compute_all_metrics(distances)
        GPSMetrics.print_metrics(metrics, title="Test Set Performance")
    """
    
    @staticmethod
    def compute_all_metrics(distances):
        """
        Compute all metrics from distance errors
        
        Args:
            distances (torch.Tensor or np.ndarray): Distance errors in meters
                Can be 1D array of distances for each prediction
            
        Returns:
            dict: Dictionary of metric name → value
                Keys:
                    - mean_error: Average distance error
                    - median_error: Median distance error (robust to outliers)
                    - std_error: Standard deviation of errors
                    - min_error: Best prediction error
                    - max_error: Worst prediction error
                    - p50_error: 50th percentile (same as median)
                    - p90_error: 90th percentile (90% of errors below this)
                    - p95_error: 95th percentile
                    - p99_error: 99th percentile
                    - accuracy_5m: Percentage of predictions within 5 meters
                    - accuracy_10m: Percentage within 10 meters
                    - accuracy_50m: Percentage within 50 meters
        """
        # Convert PyTorch tensor to numpy if needed
        if isinstance(distances, torch.Tensor):
            distances = distances.cpu().numpy()
        
        # Compute all metrics
        return {
            # Central tendency
            'mean_error': float(np.mean(distances)),
            'median_error': float(np.median(distances)),
            
            # Spread
            'std_error': float(np.std(distances)),
            'min_error': float(np.min(distances)),
            'max_error': float(np.max(distances)),
            
            # Percentiles - show error distribution
            # p50 = median, p90 means 90% of errors are below this value
            'p50_error': float(np.percentile(distances, 50)),
            'p90_error': float(np.percentile(distances, 90)),
            'p95_error': float(np.percentile(distances, 95)),
            'p99_error': float(np.percentile(distances, 99)),
            
            # Accuracy at thresholds - application-specific requirements
            # What percentage of predictions are "good enough"?
            'accuracy_5m': float(np.mean(distances <= 5) * 100),    # % within 5m
            'accuracy_10m': float(np.mean(distances <= 10) * 100),  # % within 10m
            'accuracy_50m': float(np.mean(distances <= 50) * 100),  # % within 50m
        }
    
    @staticmethod
    def print_metrics(metrics, title="Metrics"):
        """
        Pretty-print metrics table with formatting
        
        Creates an ASCII table with:
            - Aligned columns
            - Grouped sections (errors, percentiles, accuracy)
            - Units (meters, percentages)
        
        Args:
            metrics (dict): Metrics from compute_all_metrics()
            title (str): Title for the metrics table
        """
        print(f"\n{'='*60}")
        print(f"{title:^60}")
        print(f"{'='*60}")
        
        # Error statistics
        print(f"Mean Error:      {metrics['mean_error']:>8.2f} m")
        print(f"Median Error:    {metrics['median_error']:>8.2f} m")
        print(f"Std Dev:         {metrics['std_error']:>8.2f} m")
        print(f"Min Error:       {metrics['min_error']:>8.2f} m")
        print(f"Max Error:       {metrics['max_error']:>8.2f} m")
        
        # Percentiles - show distribution shape
        print(f"-" * 60)
        print(f"50th Percentile: {metrics['p50_error']:>8.2f} m")
        print(f"90th Percentile: {metrics['p90_error']:>8.2f} m")
        print(f"95th Percentile: {metrics['p95_error']:>8.2f} m")
        print(f"99th Percentile: {metrics['p99_error']:>8.2f} m")
        
        # Accuracy thresholds
        print(f"-" * 60)
        print(f"Accuracy (≤5m):  {metrics['accuracy_5m']:>7.1f} %")
        print(f"Accuracy (≤10m): {metrics['accuracy_10m']:>7.1f} %")
        print(f"Accuracy (≤50m): {metrics['accuracy_50m']:>7.1f} %")
        print(f"{'='*60}\n")
    
    @staticmethod
    def compute_haversine_distances(pred, target, min_val, max_val):
        """
        Compute Haversine distances for evaluation
        
        Unlike the loss function (which computes mean for optimization),
        this returns individual distances for each prediction.
        Useful for detailed error analysis and metrics computation.
        
        Args:
            pred (torch.Tensor): Normalized predictions (batch_size, 2)
            target (torch.Tensor): Normalized targets (batch_size, 2)
            min_val (torch.Tensor): Min GPS values for denormalization (2,)
            max_val (torch.Tensor): Max GPS values for denormalization (2,)
            
        Returns:
            np.ndarray: Distance in meters for each prediction (batch_size,)
        """
        # Denormalize from [0,1] to real GPS coordinates
        pred_real = pred * (max_val - min_val) + min_val
        target_real = target * (max_val - min_val) + min_val
        
        # Convert to radians
        rad_pred = torch.deg2rad(pred_real)
        rad_target = torch.deg2rad(target_real)
        
        # Extract lat/lon
        lat1, lon1 = rad_pred[:, 0], rad_pred[:, 1]
        lat2, lon2 = rad_target[:, 0], rad_target[:, 1]
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = torch.sin(dlat / 2)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2)**2
        a = torch.clamp(a, min=1e-6, max=1.0 - 1e-6)
        
        sqrt_a = torch.sqrt(a)
        sqrt_a = torch.clamp(sqrt_a, min=0.0, max=1.0 - 1e-7)
        
        c = 2 * torch.asin(sqrt_a)
        
        # Earth's radius in meters
        R = 6371000
        distances = R * c
        
        # Return as numpy array
        return distances.cpu().numpy()
