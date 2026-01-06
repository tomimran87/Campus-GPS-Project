import os
import numpy as np
from PIL import Image
from pillow_heif import register_heif_opener
# from google.colab import drive

# Register HEIC opener
register_heif_opener()

# Mount Drive
# if not os.path.exists('/content/drive'):
#     drive.mount('/content/drive')

def get_decimal_from_dms(dms, ref):
    degrees = dms[0]
    minutes = dms[1]
    seconds = dms[2]
    decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
    if ref in ['S', 'W']:
        decimal = -decimal
    return decimal

def get_gps_from_image(image):
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
               to_float(lat_dms[1])/60.0 +
               to_float(lat_dms[2])/3600.0)

        lon = (to_float(lon_dms[0]) +
               to_float(lon_dms[1])/60.0 +
               to_float(lon_dms[2])/3600.0)

        # Handle Hemisphere
        if lat_ref == 'S': lat = -lat
        if lon_ref == 'W': lon = -lon

        return (lat, lon)

    except (KeyError, IndexError, TypeError, ZeroDivisionError) as e:
        # This catches "int object not subscriptable" if data is still malformed
        # print(f"Debug: GPS Parse Error: {e}")
        return None

def create_dataset_from_drive(folder_path, image_size=(224, 224)):
    images_list = []
    labels_list = []

    print(f"Scanning {folder_path}...")
    valid_extensions = ('.jpg', '.jpeg', '.png', '.heic', '.HEIC')

    for filename in os.listdir(folder_path):
        # if filename.endswith(valid_extensions):
        file_path = os.path.join(folder_path, filename)
        try:
            with Image.open(file_path) as img:
                gps = get_gps_from_image(img)

                if gps:
                    img_resized = img.resize(image_size)
                    # Normalize now or later? Usually easier to do later as shown previously
                    img_array = np.array(img_resized.convert('RGB'))

                    images_list.append(img_array)
                    labels_list.append(gps)
                else:
                    print(f"Skipping {filename}: No GPS metadata found.")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    return np.array(images_list), np.array(labels_list)

# --- USAGE ---
# Update this path
DRIVE_FOLDER = '/mnt/c/Users/maymi/OneDrive/שולחן העבודה'

X, y = create_dataset_from_drive('/mnt/c/Users/maymi/OneDrive/תמונות/DL_project')

if len(X) > 0:
    print(f"Success! Found {len(X)} images with GPS data.")
    # Normalize and Save
    X_norm = X.astype('float32') / 255.0
    np.save(os.path.join(DRIVE_FOLDER, 'X_gps.npy'), X_norm)
    np.save(os.path.join(DRIVE_FOLDER, 'y_gps.npy'), y)
else:
    print("No images with GPS data were found.")