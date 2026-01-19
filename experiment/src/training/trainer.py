import torch
import torch.optim as optim
from tqdm import tqdm
# import cv2
import numpy as np

class Trainer:
    """
    Training Orchestrator for GPS Localization Models
    
    Implements a complete training pipeline with:
        - Standard backpropagation training loop
        - Gradient clipping to prevent exploding gradients
        - NaN/Inf detection with batch skipping
        - Learning rate scheduling based on validation loss
        - Early stopping to prevent overfitting
        - Best model checkpointing
        - Progress bar visualization
    
    Args:
        model (nn.Module): Neural network model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        loss_fn (nn.Module): Loss function (typically HaversineLoss)
        device (torch.device): Device for GPU/CPU computation
        lr (float): Initial learning rate (default: 0.0001)
    """
    def __init__(self, model, train_loader, val_loader, loss_fn, device, lr=0.0001, epochs=30):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.device = device
        self.epochs = epochs
        
        # Optimizer: AdamW (Adam with Weight Decay)
        # AdamW separates weight decay from gradient updates, improving regularization
        # weight_decay=1e-4 provides L2 regularization to prevent overfitting
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        
        # Learning Rate Scheduler: ReduceLROnPlateau
        # Monitors validation loss and reduces LR when improvement plateaus
        # mode='min': Minimize validation loss
        # factor=0.5: Reduce LR by half when triggered
        # patience=3: Wait 3 epochs without improvement before reducing
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        # Early Stopping Parameters
        # Prevents overfitting by stopping when validation loss stops improving
        self.best_val_loss = float('inf')
        self.patience = 10  # Increased from 5 to allow more training
        self.patience_counter = 0
        self.best_model_state = None

    def fit(self):
        """
        Train the model for a specified number of epochs
        
        Implements:
            - Standard training loop with backpropagation
            - Gradient clipping to prevent exploding gradients
            - NaN/Inf detection with batch skipping
            - Learning rate scheduling based on validation loss
            - Early stopping with best model restoration
            - Progress bar visualization
        
        Args:
            epochs (int): Number of training epochs
            
        Returns:
            list: Validation loss history for plotting
        """
        epochs = self.epochs
        history = []
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            nan_batches = 0  # Count NaN occurrences
            
            # Progress bar for training with tqdm
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Zero gradients from previous iteration
                self.optimizer.zero_grad()
                
                # Forward pass: compute predictions
                preds = self.model(images)
                
                # Compute loss using Haversine distance
                loss = self.loss_fn(preds, labels)
                
                # CRITICAL: Check for NaN/Inf before backward pass
                # If detected, skip this batch to prevent complete training collapse
                # NaN can occur from numerical instability in loss computation
                if torch.isnan(loss) or torch.isinf(loss):
                    nan_batches += 1
                    if nan_batches <= 3:  # Only print first few warnings
                        print(f"\n Warning: NaN/Inf loss detected in epoch {epoch+1}, skipping batch")
                    continue
                
                # Backward pass: compute gradients
                loss.backward()
                
                # CRITICAL: Clip gradients to prevent explosion
                # Uses L2 norm clipping: scales gradients if total norm > max_norm
                # max_norm=1.0 is conservative but safe for Haversine loss
                # Without this, gradients can explode to infinity around epoch 6-8
                # This is especially important when asin gradient becomes large
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Update weights using optimizer (AdamW)
                self.optimizer.step()
                
                # Accumulate loss for epoch average
                train_loss += loss.item()
                
                # Step the OneCycleLR scheduler after each batch
                # self.scheduler.step()
                
                # Update progress bar with current loss
                pbar.set_postfix({'loss': f"{loss.item():.2f}m"})
            
            # Warning if many NaN batches occurred
            if nan_batches > 0:
                print(f"Epoch {epoch+1}: {nan_batches} batches skipped due to NaN/Inf")

            # Compute epoch averages
            avg_train_loss = train_loss / len(self.train_loader) if len(self.train_loader) > 0 else float('inf')
            avg_val_loss = self.evaluate()
            
            # Early stopping check
            if avg_val_loss < self.best_val_loss:
                # New best model found
                self.best_val_loss = avg_val_loss
                self.patience_counter = 0
                # Save best model state for later restoration
                self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                print(f"✓ New best model (Val Error: {avg_val_loss:.2f}m)")
            else:
                # No improvement
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"\n Early stopping triggered after {epoch+1} epochs (no improvement for {self.patience} epochs)")
                    # Restore best model before stopping
                    if self.best_model_state is not None:
                        self.model.load_state_dict(self.best_model_state)
                        print(f"✓ Restored best model (Val Error: {self.best_val_loss:.2f}m)")
                    break
            
            # Update learning rate based on validation loss
            # ReduceLROnPlateau monitors validation and reduces LR if no improvement
            self.scheduler.step(avg_val_loss)
            
            # Store history for plotting
            history.append(avg_val_loss)
            
            # Print epoch summary
            print(f"Epoch {epoch+1}: Train Error: {avg_train_loss:.2f}m | Val Error: {avg_val_loss:.2f}m")
        
        return history
        
 
    def evaluate_fusion(self):
        """
        Helper for Approach A validation
        """
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for images, sift_vectors, labels in self.val_loader:
                images, sift_vectors, labels = images.to(self.device), sift_vectors.to(self.device), labels.to(self.device)
                preds = self.model(images, sift_vectors)
                total_loss += self.loss_fn(preds, labels).item()
        return total_loss / len(self.val_loader)

    


    def evaluate(self):
        """
        Evaluate model on validation set
        
        Computes average loss across all validation batches without
        updating model weights. Uses torch.no_grad() to disable gradient
        computation for efficiency.
        
        Returns:
            float: Average validation loss in meters
        """
        self.model.eval()  # Set model to evaluation mode (disables dropout, etc.)
        total_loss = 0.0
        
        # Disable gradient computation for validation (saves memory and computation)
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Forward pass only (no backward)
                preds = self.model(images)
                
                # Compute loss and accumulate
                total_loss += self.loss_fn(preds, labels).item()
        
        # Return average loss across all batches
        return total_loss / len(self.val_loader)