from model_trainer import train_gps_model

def main():
    # --- 1. Data Paths ---
    # Make sure these files are in the same folder (uploaded via scp)
    X_path = "X_gps.npy"
    y_path = "y_gps.npy"
    
    # --- 2. Hyperparameters (The "Tuning Knobs") ---
    # Since we are using ResNet (Transfer Learning), it learns fast.
    # 20 epochs is usually enough, but you can try 30 if the loss is still dropping.
    EPOCHS = 50
    
    # Standard batch size for 1080 Ti is 32 or 64. 
    BATCH_SIZE = 32
    
    # Pre-trained models like small learning rates. 
    # If the loss jumps around wildly, lower this to 0.0001.
    LEARNING_RATE = 0.001 

    print(f"--- Starting GPS Regression Project ---")
    print(f"Data: {X_path}, {y_path}")
    print(f"Config: {EPOCHS} epochs, LR={LEARNING_RATE}, ResNet18")

    # --- 3. Train ---
    # The function now returns the model and the final validation loss (MAE)
    model, final_mae = train_gps_model(
        X_path, 
        y_path, 
        epochs=EPOCHS, 
        batch_size=BATCH_SIZE, 
        learning_rate=LEARNING_RATE
    )
    
    print(f"\nFinal Test Set MAE (Normalized): {final_mae:.6f}")
    print("SUCCESS: Model saved to 'gps_resnet_model.pth'")
    print("SUCCESS: Scaling values saved to 'gps_scaling_values.npy'")

if __name__ == "__main1__":
    main()