import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
import torchvision.models as models

# --- 1. Define the Architecture (ResNet Transfer Learning) ---

class GPSResNet(nn.Module):
    def __init__(self):
        super(GPSResNet, self).__init__()
        # Load pre-trained ResNet18
        # weights='DEFAULT' uses the best available pre-trained weights
        self.resnet = models.resnet18(weights='DEFAULT')
        
        # Replace the final fully connected layer (which usually outputs 1000 classes)
        # We need it to output 2 numbers (Latitude, Longitude)
        num_features = self.resnet.fc.in_features
        
        # We add a small "Head" with Dropout to prevent memorization
        self.resnet.fc = nn.Sequential(
            nn.Dropout(p=0.2),            
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Linear(64, 2)              # Output: Lat, Lon
        )
        
    def forward(self, x):
        return self.resnet(x)

# --- 2. Training Function ---

def train_gps_model(x_path, y_path, epochs=20, batch_size=32, learning_rate=0.001):
    # 1. Load Data
    print("Loading data...")
    X = np.load(x_path)
    y = np.load(y_path)
    
    # --- STEP 1: CALCULATE & SAVE NORMALIZATION VALUES ---
    min_val = np.min(y, axis=0)
    max_val = np.max(y, axis=0)
    
    # Save these for future use!
    np.save("gps_scaling_values.npy", np.array([min_val, max_val]))
    print(f"Scaling values saved.")
    
    # Normalize: (value - min) / (max - min)
    y_norm = (y - min_val) / (max_val - min_val)

    # Transpose if needed (N, H, W, 3) -> (N, 3, H, W)
    if X.shape[-1] == 3: 
        X = np.transpose(X, (0, 3, 1, 2))
    
    # Normalize images to 0-1 if they are 0-255
    if X.max() > 1.0:
        X = X / 255.0
        
    # --- STEP 2: CONVERT TO TENSORS ---
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y_norm, dtype=torch.float32) # Using Normalized Labels
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X_tensor, y_tensor, test_size=0.2, random_state=42
    )
    
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Create DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Setup Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    
    model = GPSResNet().to(device)
    
    # L1Loss (MAE) is better for GPS accuracy than MSE
    criterion = nn.L1Loss() 
    
    # Use standard Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print("Starting training with ResNet18...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.6f}")

    # --- STEP 3: EVALUATE ---
    print("Evaluating...")
    model.eval()
    
    # Validation MAE
    total_mae = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_mae += criterion(outputs, labels).item()
    avg_mae = total_mae / len(test_loader)
    
    # Visual Check
    print("\n------------------------------------------------")
    print("Visual Check: Last Photo (Denormalized)")
    
    last_img, last_label_norm = test_dataset[-1]
    last_img_batch = last_img.unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred_norm = model(last_img_batch)[0].cpu().numpy()
    
    real_norm = last_label_norm.numpy()
    
    # Denormalize
    real_gps = real_norm * (max_val - min_val) + min_val
    pred_gps = pred_norm * (max_val - min_val) + min_val
    
    print(f"Real GPS:      {real_gps[0]:.6f}, {real_gps[1]:.6f}")
    print(f"Predicted GPS: {pred_gps[0]:.6f}, {pred_gps[1]:.6f}")
    
    diff_lat = abs(real_gps[0] - pred_gps[0])
    diff_lon = abs(real_gps[1] - pred_gps[1])
    
    # Approx error in meters
    dist_m = np.sqrt((diff_lat * 111000)**2 + (diff_lon * 95000)**2)
    
    print(f"Difference:    {diff_lat:.6f}, {diff_lon:.6f}")
    print(f"Approx Error:  {dist_m:.2f} meters")
    print("------------------------------------------------\n")

    torch.save(model.state_dict(), "gps_resnet_model.pth")
    print("Model saved to gps_resnet_model.pth")
    
    return model, avg_mae