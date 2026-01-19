import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split #helps to split the data into train, val, test sets

class GPSDataManager:
    """
    GPS Dataset Manager with Proper Train/Val/Test Splitting
    
    Handles loading, preprocessing, and splitting of GPS image localization data.
    Implements proper data hygiene to prevent data leakage:
        1. Load raw data
        2. Split into train/val/test FIRST
        3. Compute normalization parameters ONLY from training set
        4. Apply same normalization to val/test sets
    
    This prevents data leakage where validation/test statistics influence training.
    
    Data Format:
        - X: Images of shape (N, H, W, C) in [0, 255] range
        - y: GPS coordinates of shape (N, 2) as [latitude, longitude]
    
    Normalization:
        - Images: Scaled to [0, 1] via division by 255
        - GPS: Min-max normalized to [0, 1] using training set statistics ONLY
    
    Args:
        x_path (str): Path to numpy file containing images
        y_path (str): Path to numpy file containing GPS coordinates
        batch_size (int): Batch size for DataLoader (default: 32)
        test_size (float): Fraction of data for testing (default: 0.15)
        val_size (float): Fraction of data for validation (default: 0.15)
        random_state (int): Random seed for reproducibility (default: 42)
    """
    def __init__(self, x_path, y_path, batch_size=32, test_size=0.15, val_size=0.15, random_state=42):
        self.x_path = x_path
        self.y_path = y_path
        self.batch_size = batch_size
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        
        # These will store normalization parameters computed from training set ONLY
        self.min_val = None
        self.max_val = None

    def _load_raw_data(self):
        """
        Load raw numpy arrays from disk
        
        Returns:
            tuple: (X, y) where X is images (N, H, W, C) and y is GPS coords (N, 2)
        """
        print("Loading raw data from disk...")
        X = np.load(self.x_path)
        y = np.load(self.y_path)
        print(f"Loaded {len(X)} images with shape {X.shape}")
        print(f"GPS coordinate range: lat [{y[:, 0].min():.6f}, {y[:, 0].max():.6f}], "
              f"lon [{y[:, 1].min():.6f}, {y[:, 1].max():.6f}]")
        return X, y

    def _preprocess_images(self, X):
        """
        Preprocess images for neural network input
        
        Steps:
            1. Transpose from (N, H, W, C) to (N, C, H, W) for PyTorch
            2. Normalize from [0, 255] to [0, 1]
        
        Args:
            X (np.ndarray): Image array, shape (N, H, W, C)
            
        Returns:
            torch.Tensor: Preprocessed images, shape (N, C, H, W)
        """
        # Transpose to PyTorch convention: channels-first (N, C, H, W)
        if X.shape[-1] == 3:
            X = np.transpose(X, (0, 3, 1, 2))
        
        # Normalize pixel values from [0, 255] to [0, 1]
        if X.max() > 1.0:
            X = X / 255.0
        
        return torch.tensor(X, dtype=torch.float32)

    def _normalize_gps(self, y_train, y_val=None, y_test=None):
        """
        Normalize GPS coordinates to [0, 1] range
        
        CRITICAL: Computes min/max ONLY from training set, then applies
        the same transformation to validation and test sets. This prevents
        data leakage where test set statistics influence training.
        
        Formula: y_norm = (y - min) / (max - min)
        
        Args:
            y_train (np.ndarray): Training GPS coordinates
            y_val (np.ndarray, optional): Validation GPS coordinates
            y_test (np.ndarray, optional): Test GPS coordinates
            
        Returns:
            tuple: (y_train_norm, y_val_norm, y_test_norm) as torch.Tensors
        """
        # Compute normalization parameters from training set ONLY
        # This is the critical fix for data leakage
        self.min_val = np.min(y_train, axis=0)
        self.max_val = np.max(y_train, axis=0)
        
        print(f"Normalization parameters (from training set only):")
        print(f"  Min: lat={self.min_val[0]:.6f}, lon={self.min_val[1]:.6f}")
        print(f"  Max: lat={self.max_val[0]:.6f}, lon={self.max_val[1]:.6f}")
        
        # Normalize training set using its own statistics
        y_train_norm = (y_train - self.min_val) / (self.max_val - self.min_val)
        
        # Apply same normalization to validation/test sets (no data leakage!)
        y_val_norm = (y_val - self.min_val) / (self.max_val - self.min_val) if y_val is not None else None
        y_test_norm = (y_test - self.min_val) / (self.max_val - self.min_val) if y_test is not None else None
        
        # Convert to PyTorch tensors
        y_train_norm = torch.tensor(y_train_norm, dtype=torch.float32)
        y_val_norm = torch.tensor(y_val_norm, dtype=torch.float32) if y_val_norm is not None else None
        y_test_norm = torch.tensor(y_test_norm, dtype=torch.float32) if y_test_norm is not None else None
        
        return y_train_norm, y_val_norm, y_test_norm

    def get_loaders(self):
        """
        Create train/validation/test DataLoaders with proper data splitting
        
        Process:
            1. Load raw data
            2. Split into train/temp (temp = val + test)
            3. Split temp into val/test
            4. Preprocess images (same for all sets)
            5. Normalize GPS using training statistics ONLY (prevents leakage)
            6. Create DataLoaders
        
        Returns:
            tuple: (train_loader, val_loader, test_loader)
        """
        # Step 1: Load raw data
        X, y = self._load_raw_data()
        
        # Step 2: First split - separate out training set
        # temp_size = fraction for validation + test combined
        temp_size = self.test_size + self.val_size
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            test_size=temp_size,
            random_state=self.random_state,
            shuffle=True
        )
        
        # Step 3: Second split - divide temp into validation and test
        # Calculate relative validation size within temp set
        val_relative_size = self.val_size / temp_size
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=(1 - val_relative_size),
            random_state=self.random_state
        )
        
        print(f"\nData split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        print(f"Split ratios: Train={len(X_train)/len(X):.1%}, Val={len(X_val)/len(X):.1%}, Test={len(X_test)/len(X):.1%}\n")
        
        # Step 4: Normalize GPS coordinates using ONLY training statistics
        # This is the key fix for data leakage
        # NOTE: We skip image preprocessing here - augmentation handles it
        y_train_norm, y_val_norm, y_test_norm = self._normalize_gps(y_train, y_val, y_test)
        
        # Step 5: Create augmented datasets with proper transforms
        # Import augmentation module
        from augmentation import GPSAugmentation, AugmentedGPSDataset
        
        # Create augmentation pipelines
        augmentor = GPSAugmentation(image_size=224)
        
        # Create datasets with augmentation
        # Training: aggressive augmentation for robustness
        # Val/Test: only normalization (measure true performance)
        train_ds = AugmentedGPSDataset(
            X_train, y_train_norm.detach().cpu().numpy(),
            transform=augmentor.get_train_transforms()
        )
        val_ds = AugmentedGPSDataset(
            X_val, y_val_norm.detach().cpu().numpy(),
            transform=augmentor.get_val_transforms()
        )
        test_ds = AugmentedGPSDataset(
            X_test, y_test_norm.detach().cpu().numpy(),
            transform=augmentor.get_val_transforms()
        )
        
        # Step 6: Create DataLoaders with parallel loading
        # Shuffle training for better generalization
        # num_workers=4: parallel data loading for faster training
        # pin_memory=True: faster GPU transfer
        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader

    def get_scaling_tensors(self):
        """
        Get normalization parameters as tensors for use in loss function
        
        These tensors are used by HaversineLoss to denormalize predictions
        back to real GPS coordinates before computing distance.
        
        Returns:
            tuple: (min_val, max_val) as torch.Tensors, each shape (2,)
        
        Raises:
            ValueError: If called before get_loaders()
        """
        if self.min_val is None or self.max_val is None:
            raise ValueError("Must call get_loaders() first to compute scaling parameters")
        
        return torch.tensor(self.min_val, dtype=torch.float32), torch.tensor(self.max_val, dtype=torch.float32)