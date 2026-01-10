import folium
import math
from get_photo_coordinates_to_tensors import get_gps_from_image, numpy_to_tensors
import numpy as np
from PIL import Image

def get_default_map():
    """
    Returns a dark-mode default map for the initial load.
    """
    # We use CartoDB dark_matter for the 'Dark Mode' aesthetic
    m = folium.Map(location=[25, 0], zoom_start=2, tiles='CartoDB dark_matter')
    return m._repr_html_()

# --- 2. The Visualizer (Updated for Dark Mode) ---
def map_distance_visualizer(gps1, gps2):
    m = folium.Map(location=[gps1[0], gps1[1]], tiles='CartoDB dark_matter')

    # True Location (Green Crosshair)
    folium.Marker(
        location=[gps1[0], gps1[1]], 
        popup="True Location", 
        icon=folium.Icon(color="green", icon="crosshairs", prefix='fa')
    ).add_to(m)

    # Predicted Location (Red Crosshair)
    folium.Marker(
        location=[gps2[0], gps2[1]], 
        popup="AI Prediction", 
        icon=folium.Icon(color="red", icon="crosshairs", prefix='fa')
    ).add_to(m)

    # The Line
    folium.PolyLine(
        locations=[gps1, gps2], 
        color="#a855f7", # Neon Purple to match your theme
        weight=4, 
        opacity=0.8, 
        dash_array='10'
    ).add_to(m)

    m.fit_bounds([gps1, gps2], padding=(50,50))
    return m._repr_html_()

# Model prediction function
def predict(image_path, image_size=(224, 224)):
    if image_path is None:
        return 0.0, 0.0
    try:
        input_img = Image.open(image_path)
    except Exception:
        return 0.0, 0.0
    
    final_results = [31.262520, 34.799629]
    img_resized =  input_img.resize(image_size)
    img_array = np.array(img_resized.convert('RGB'))

    X_batch = np.array([img_array])
    # coordinates_tensor = Model(input_img)
    # final_results = coordinates_tensor[0]
    return final_results

def real_Gps_Coordination(image_path):
    if image_path is None:
        return 0.0, 0.0

    try:
        img_obj = Image.open(image_path)
        gps = get_gps_from_image(img_obj)
    except Exception:
        return 0.0, 0.0

    if gps is None:
        return 0.0, 0.0
        
    return gps[0], gps[1]

def calculate_Distance(real_location, predicted_location):
    R = 6371.0

    lat1_rad = math.radians(real_location[0])
    lon1_rad = math.radians(real_location[1])
    lat2_rad = math.radians(predicted_location[0])
    lon2_rad = math.radians(predicted_location[1])

    dlon_rad = lon2_rad - lon1_rad
    dlat_rad = lat2_rad - lat1_rad

    a = math.sin(dlat_rad / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon_rad / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c