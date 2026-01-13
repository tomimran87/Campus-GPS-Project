"""
Simple script to examine y_gps.npy from liran_data
"""
import numpy as np

# Load the GPS coordinates
y_path = "/home/tommimra/GPS_BGU_model/liran_data/y_gps.npy"
y_data = np.load(y_path)

print(f"Shape: {y_data.shape}")
print(f"Data type: {y_data.dtype}")
print(f"Min values: lat={y_data[:, 0].min():.6f}, lon={y_data[:, 1].min():.6f}")
print(f"Max values: lat={y_data[:, 0].max():.6f}, lon={y_data[:, 1].max():.6f}")
print(f"Mean values: lat={y_data[:, 0].mean():.6f}, lon={y_data[:, 1].mean():.6f}")
print(f"Unique values in latitude: {len(np.unique(y_data[:, 0]))}")
print(f"Unique values in longitude: {len(np.unique(y_data[:, 1]))}")

print("\nFirst 10 GPS coordinates:")
for i in range(min(10, len(y_data))):
    print(f"  [{i:3d}]: lat={y_data[i, 0]:.6f}, lon={y_data[i, 1]:.6f}")

print("\nLast 10 GPS coordinates:")
for i in range(max(0, len(y_data)-10), len(y_data)):
    print(f"  [{i:3d}]: lat={y_data[i, 0]:.6f}, lon={y_data[i, 1]:.6f}")

# Check if all values are the same
all_same_lat = np.all(y_data[:, 0] == y_data[0, 0])
all_same_lon = np.all(y_data[:, 1] == y_data[0, 1])

print(f"\nAll latitudes are the same: {all_same_lat}")
print(f"All longitudes are the same: {all_same_lon}")

if all_same_lat and all_same_lon:
    print(f"ALL GPS coordinates are: [{y_data[0, 0]:.6f}, {y_data[0, 1]:.6f}]")