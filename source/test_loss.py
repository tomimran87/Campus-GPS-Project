import torch
import numpy as np
from loss import HaversineLoss

# Simulation
device = torch.device("cuda")

# Real GPS normalization from data
min_val = torch.tensor([31.261283, 34.801083], dtype=torch.float32)
max_val = torch.tensor([31.262683, 34.804469], dtype=torch.float32)

loss_fn = HaversineLoss(min_val, max_val, device)

# Create predictions and targets (normalized [0, 1])
preds = torch.tensor([[0.5, 0.5], [0.6, 0.6], [0.4, 0.4]], device=device, requires_grad=True)
targets = torch.tensor([[0.3, 0.7], [0.8, 0.2], [0.5, 0.5]], device=device)

print(f"Predictions: {preds}")
print(f"Targets: {targets}")

# Compute loss
loss = loss_fn(preds, targets)
print(f"\nLoss: {loss.item():.2f}m")
print(f"Loss requires_grad: {loss.requires_grad}")

# Backward
loss.backward()

print(f"\nPrediction gradients: {preds.grad}")
print(f"Gradient norm: {preds.grad.norm():.6f}")

if preds.grad.norm() == 0:
    print("\n❌ ZERO GRADIENTS - LOSS FUNCTION PROBLEM!")
else:
    print("\n✓ Gradients are flowing")
