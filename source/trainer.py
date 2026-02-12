import torch
import torch.optim as optim
from tqdm import tqdm
import cv2
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
        # weight_decay=1e-3 provides L2 regularization to prevent overfitting
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-3)
        
        # Learning Rate Scheduler: ReduceLROnPlateau
        # Monitors validation loss and reduces LR when improvement plateaus
        # mode='min': Minimize validation loss
        # factor=0.5: Reduce LR by half when triggered
        # patience=3: Wait 3 epochs without improvement before reducing
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
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
                pbar.set_postfix({'loss': f"{loss.item():.2f}"})
            
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
                print(f"✓ New best model (Val Error: {avg_val_loss:.4f})")
            else:
                # No improvement
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"\n Early stopping triggered after {epoch+1} epochs (no improvement for {self.patience} epochs)")
                    # Restore best model before stopping
                    if self.best_model_state is not None:
                        self.model.load_state_dict(self.best_model_state)
                        print(f"✓ Restored best model (Val Error: {self.best_val_loss:.4f})")
                    break
            
            # Update learning rate based on validation loss
            # ReduceLROnPlateau monitors validation and reduces LR if no improvement
            self.scheduler.step(avg_val_loss)
            
            # Store history for plotting
            history.append(avg_val_loss)
            
            # Print epoch summary
            print(f"Epoch {epoch+1}: Train Error: {avg_train_loss:.4f} | Val Error: {avg_val_loss:.4f}")
        
        return history

    def fit_with_SIFT_A(self):
        """
        Approach A: Late Fusion Training Loop.
        
        Requires:
            - Model that accepts (image, sift_vector) in forward()
            - DataLoader that yields (image, sift_vector, label)
        """
        epochs = self.epochs
        history = []
        
        # 
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs} (Fusion)", leave=False)
            
            # UPDATED: Unpack 3 items instead of 2
            for images, sift_vectors, labels in pbar:
                images = images.to(self.device)
                sift_vectors = sift_vectors.to(self.device) # Ensure SIFT vector is a tensor
                labels = labels.to(self.device)
                
                self.optimizer.zero_grad()
                
                # UPDATED: Pass both inputs to the model
                preds = self.model(images, sift_vectors)
                
                loss = self.loss_fn(preds, labels)
                
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                train_loss += loss.item()
                pbar.set_postfix({'loss': f"{loss.item():.2f}m"})
            
            # Validation Step (Must also be updated for Fusion)
            avg_val_loss = self.evaluate_fusion() # You need a separate eval for fusion
            self.scheduler.step(avg_val_loss)
            history.append(avg_val_loss)
            
            # ... (Early stopping logic remains the same) ...
            
            print(f"Epoch {epoch+1}: Train Error: {train_loss/len(self.train_loader):.2f}m | Val Error: {avg_val_loss:.2f}m")
            
        return history

    def evaluate_fusion(self):
        """Helper for Approach A validation"""
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for images, sift_vectors, labels in self.val_loader:
                images, sift_vectors, labels = images.to(self.device), sift_vectors.to(self.device), labels.to(self.device)
                preds = self.model(images, sift_vectors)
                total_loss += self.loss_fn(preds, labels).item()
        return total_loss / len(self.val_loader)

    def fit_with_SIFT_B(self):
        """
        Approach B: Train CNN normally, then 'Fit' a SIFT Retrieval System.
        
        1. Trains the EfficientNet model using standard backprop.
        2. Builds a SIFT Index of the Training Set.
        3. Evaluates on Validation Set using 'CNN + SIFT Refinement'.
        """
        
        # Step 1: Standard CNN Training
        print("Phase 1: Training EfficientNet Backbone...")
        cnn_history = self.fit() # Use the standard fit method you already wrote
        
        # 
        
        # Step 2: Build SIFT Database (Indexing)
        print("\nPhase 2: Building SIFT Landmark Index...")
        self.model.eval()
        sift = cv2.SIFT_create(nfeatures=500) # Limit features for speed
        
        # We need to store SIFT descriptors and their corresponding Ground Truth GPS
        database_descriptors = []
        database_gps = []
        
        # Iterate through training data to build the index
        # Note: We need the raw images for SIFT, so we assume the loader provides them
        # or we inverse-normalize the tensor. Here we assume tensor -> numpy conversion.
        with torch.no_grad():
            for images, labels in tqdm(self.train_loader, desc="Indexing"):
                for i in range(len(images)):
                    # Convert Tensor to Grayscale for OpenCV
                    img_np = images[i].cpu().permute(1, 2, 0).numpy()
                    img_np = (img_np * 255).astype(np.uint8)
                    img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                    
                    # Extract SIFT
                    kp, des = sift.detectAndCompute(img_gray, None)
                    
                    # Only store if descriptors were found
                    if des is not None:
                        database_descriptors.append(des)
                        database_gps.append(labels[i].cpu().numpy())

        # Step 3: Evaluate with Refinement
        print("\nPhase 3: Evaluating with SIFT Refinement...")
        total_refined_loss = 0.0
        
        # FLANN Matcher parameters (Fast Library for Approximate Nearest Neighbors)
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Refining"):
                images = images.to(self.device)
                
                # 3a. Get Coarse Prediction from CNN
                cnn_preds = self.model(images)
                
                for i in range(len(images)):
                    # 3b. Extract SIFT from Validation Image
                    img_np = images[i].cpu().permute(1, 2, 0).numpy()
                    img_np = (img_np * 255).astype(np.uint8)
                    img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                    kp, des = sift.detectAndCompute(img_gray, None)
                    
                    refined_pred = cnn_preds[i].cpu().numpy()
                    
                    # 3c. Find Best Match in Database
                    if des is not None and len(database_descriptors) > 0:
                        best_match_idx = -1
                        max_good_matches = 0
                        
                        # Compare against database (This is slow - brute force loop for demo)
                        # In production, use a global descriptor like VLAD or Bag of Words
                        # to filter candidates first!
                        for db_idx, db_des in enumerate(database_descriptors):
                            try:
                                matches = flann.knnMatch(des, db_des, k=2)
                                # Lowe's Ratio Test
                                good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
                                
                                if len(good_matches) > max_good_matches:
                                    max_good_matches = len(good_matches)
                                    best_match_idx = db_idx
                            except: continue
                        
                        # 3d. Refine Logic
                        # If we found a very strong visual match (>10 keypoints matched)
                        if best_match_idx != -1 and max_good_matches > 5:
                            retrieved_gps = database_gps[best_match_idx]
                            # Weighted average: Trust the retrieved image 30%, CNN 70%
                            refined_pred = (0.7 * refined_pred) + (0.3 * retrieved_gps)
                    
                    # Calculate Loss on refined prediction
                    # (Assuming simple Euclidean for demo; use Haversine in reality)
                    loss = np.sqrt(np.sum((refined_pred - labels[i].cpu().numpy())**2))
                    total_refined_loss += loss

        avg_refined_loss = total_refined_loss / len(self.val_loader.dataset)
        print(f"Final Error with SIFT Refinement: {avg_refined_loss:.2f}m")
        
        return cnn_history

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