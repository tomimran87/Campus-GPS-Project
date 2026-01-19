import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import pillow_heif
import os
import csv
from tqdm import tqdm


# Register HEIC opener
pillow_heif.register_heif_opener()


# --- CONFIGURATION ---
IMG_SIZE = (255, 255)
BASE_DIR = '.'
PHOTOS_DIR = os.path.join(BASE_DIR, 'dataset_root', 'images')
OUTPUT_DIR = os.path.join(BASE_DIR, 'latest_data')

# --- HELPER FUNCTIONS ---

def get_decimal_from_dms(dms, ref):
    # Sanity check: ensure dms is a list/tuple with 3 values
    if not isinstance(dms, (list, tuple)) or len(dms) < 3:
        return 0.0

    degrees = dms[0]
    minutes = dms[1]
    seconds = dms[2]
    decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
    if ref in ['S', 'W']:
        decimal = -decimal
    return decimal

def get_coordinates(image_path):
    """
    Robust GPS extraction that handles HEIC pointers and standard JPEG dicts.
    """
    try:
        image = Image.open(image_path)
        gps_info = {}

        # METHOD 1: Modern Pillow "get_ifd" (Best for HEIC)
        # 0x8825 is the standard ID for the GPS IFD (Image File Directory)
        try:
            exif = image.getexif()
            if exif:
                # This automatically follows the pointer if it exists
                gps_data = exif.get_ifd(0x8825)
                if gps_data:
                    for tag, value in gps_data.items():
                        decoded = GPSTAGS.get(tag, tag)
                        gps_info[decoded] = value
        except Exception:
            pass

        # METHOD 2: Fallback to _getexif (Best for older JPGs)
        # Only run if Method 1 found nothing
        if not gps_info:
             exif_data = image._getexif()
             if exif_data:
                for tag, value in exif_data.items():
                    decoded = TAGS.get(tag, tag)
                    if decoded == "GPSInfo":
                        # CRITICAL FIX: Check if value is actually a dict before looping
                        if isinstance(value, dict):
                            for t in value:
                                sub_decoded = GPSTAGS.get(t, t)
                                gps_info[sub_decoded] = value[t]

        # Extract Lat/Lon if we found the keys
        if 'GPSLatitude' in gps_info and 'GPSLongitude' in gps_info:
            lat = get_decimal_from_dms(gps_info['GPSLatitude'], gps_info['GPSLatitudeRef'])
            lon = get_decimal_from_dms(gps_info['GPSLongitude'], gps_info['GPSLongitudeRef'])
            return [lat, lon]
        else:
            return None

    except Exception as e:
        print(f"Error reading metadata for {os.path.basename(image_path)}: {e}")
        return None

def process_folder(folder_path, output_csv=None):
    """
    Process images in a folder and extract GPS coordinates.
    
    Optionally creates a ground truth CSV file with image names and GPS coordinates.
    
    Args:
        folder_path (str): Path to the folder containing images
        output_csv (str, optional): Path to save ground truth CSV file.
                                   CSV will contain columns: [image_name, latitude, longitude]
                                   If None, no CSV is created.
    
    Returns:
        tuple: (images, coordinates) where:
            - images: List of numpy arrays (H, W, 3) with dtype uint8
            - coordinates: List of [latitude, longitude] pairs
    """
    images = []
    coordinates = []
    filenames = []
    print(f"Processing folder: {folder_path}...")

    if not os.path.exists(folder_path):
        print(f"Warning: {folder_path} does not exist. Skipping.")
        return [], []

    valid_extensions = ('.jpg', '.jpeg', '.png', '.tiff', '.heic')
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)]

    for filename in tqdm(files, desc="Processing images"):
        filepath = os.path.join(folder_path, filename)

        coords = get_coordinates(filepath)

        if coords is not None:
            try:
                img = Image.open(filepath).convert('RGB')
                img = img.resize(IMG_SIZE)
                images.append(np.array(img))
                coordinates.append(coords)
                filenames.append(filename)
            except Exception as e:
                print(f"Failed to process image {filename}: {e}")

    print(f"Found {len(images)} valid images in {folder_path}")
    
    # Save ground truth CSV if output_csv path is provided
    if output_csv is not None and len(images) > 0:
        try:
            # Create directory if it doesn't exist
            output_dir = os.path.dirname(output_csv)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Write CSV file
            with open(output_csv, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                # Write header
                writer.writerow(['image_name', 'latitude', 'longitude'])
                # Write data rows
                for filename, coords in zip(filenames, coordinates):
                    writer.writerow([filename, coords[0], coords[1]])
            
            print(f"✓ Ground truth CSV saved to {output_csv}")
        except Exception as e:
            print(f"✗ Error saving CSV to {output_csv}: {e}")
    
    return images, coordinates


def load_photos():
    # Process standard photos folder with ground truth CSV
    csv_path = os.path.join(BASE_DIR_DIR, 'dataset_root', 'gt.csv')
    X, y = process_folder(PHOTOS_DIR, output_csv=csv_path)
    X = np.array(X)
    y = np.array(y)
    print(f"Data shape before cleaning: {y.shape}")
    return X, y

def save_data(X, y, output_dir=OUTPUT_DIR):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    x_path = os.path.join(output_dir, 'X_photos.npy')
    y_path = os.path.join(output_dir, 'y_photos.npy')
    np.save(x_path, X)
    np.save(y_path, y)
    print(f"Saved images to {x_path} and GPS coords to {y_path}")


if __name__ == "__main__":
    X, y = load_photos()
    save_data(X, y)