import os
import numpy as np
import torch
from PIL import Image
from pillow_heif import register_heif_opener

# Note: Ensure you have installed the required package in your environment:
# pip install pillow-heif

# Register HEIC opener to handle iPhone photos
register_heif_opener()




def get_gps_from_image(image):
    """
    Extracts GPS coordinates (Latitude, Longitude) from an image's EXIF data.
    """
    # 1. Get the main EXIF object
    exif_data = image.getexif()
    if not exif_data:
        return None

    # 2. Use get_ifd() to extract the GPS sub-directory
    # 0x8825 is the hex code for 34853 (GPSInfo)
    try:
        gps_info = exif_data.get_ifd(0x8825)
    except AttributeError:
        # Fallback for older PIL versions
        gps_info = exif_data.get(34853)

    if not gps_info:
        return None

    # 3. Helper to safe-convert IFDRational objects to float
    def to_float(val):
        # If it's already a number, return it
        if isinstance(val, (int, float)):
            return float(val)
        # If it's an IFDRational object (typical in PIL), convert it
        if hasattr(val, 'numerator') and hasattr(val, 'denominator'):
            return float(val.numerator) / float(val.denominator) if val.denominator != 0 else 0.0
        return float(val)

    try:
        # Extract Lat/Lon using standard integer tags
        # 1: LatRef, 2: Lat, 3: LonRef, 4: Lon
        lat_dms = gps_info[2]
        lat_ref = gps_info[1]
        lon_dms = gps_info[4]
        lon_ref = gps_info[3]

        # dms is usually a tuple of 3 values (deg, min, sec)
        # We ensure each part is converted to float
        lat = (to_float(lat_dms[0]) +
               to_float(lat_dms[1]) / 60.0 +
               to_float(lat_dms[2]) / 3600.0)

        lon = (to_float(lon_dms[0]) +
               to_float(lon_dms[1]) / 60.0 +
               to_float(lon_dms[2]) / 3600.0)

        # Handle Hemisphere
        if lat_ref == 'S': lat = -lat
        if lon_ref == 'W': lon = -lon

        return (lat, lon)

    except (KeyError, IndexError, TypeError, ZeroDivisionError) as e:
        # This catches "int object not subscriptable" if data is still malformed
        # print(f"Debug: GPS Parse Error: {e}")
        return None





def numpy_to_tensors(X, y, device=None, task_type='regression'):
    """
    Converts numpy X and y arrays into PyTorch tensors with correct dtypes.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.
        device (torch.device, optional): Device to move tensors to (e.g., 'cuda', 'cpu').
        task_type (str): 'regression' or 'classification'.
    """
    # 1. Auto-detect device if not specified
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. Convert X (Features usually need to be float32)
    X_tensor = torch.from_numpy(X).float().to(device)

    # 3. Convert y (Target depends on the task)
    y_tensor = torch.from_numpy(y).to(device)

    if task_type == 'regression':
        y_tensor = y_tensor.float()  # MSELoss expects float
        # Ensure y has the shape (N, 1) if it's currently (N,)
        if y_tensor.dim() == 1:
            y_tensor = y_tensor.unsqueeze(1)

    elif task_type == 'classification':
        y_tensor = y_tensor.long()  # CrossEntropyLoss expects long (int64)

    return X_tensor, y_tensor

