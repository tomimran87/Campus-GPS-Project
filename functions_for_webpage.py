import folium
import math
from get_photo_coordinates_to_tensors import get_gps_from_image, numpy_to_tensors
import numpy as np

def map_distance_visualizer(gps1, gps2):
    m = folium.Map(location=[gps1[0], gps1[1]], titles='OpenStreetMap')

    folium.Marker(
        location=[gps1[0], gps1[1]],
        popup="True Location",
        tooltip="Click for details",
        icon=folium.Icon(color="blue", icon="map-marker")
    ).add_to(m)

    folium.Marker(
        location=[gps2[0], gps2[1]],
        popup="Model Prediction Location",
        tooltip="Click for details",
        icon=folium.Icon(color="red", icon="info-sign")
    ).add_to(m)

    folium.PolyLine(
        locations=[[gps1[0], gps1[1]], [gps2[0], gps2[1]]],
        color="purple",
        weight=5,
        opacity=0.8
    ).add_to(m)

    m.fit_bounds([gps1[0], gps1[1]], [gps2[0], gps2[1]], padding=(50,50))
    return m._repr_html_()

# Model prediction function
def predict(input_img, image_size=(224, 224)):
    final_results = [31.262520, 34.799629]
    img_resized = input_img.resize(image_size)
    img_array = np.array(img_resized.convert('RGB'))

    X_batch = np.array([img_array])
    # coordinates_tensor = Model(input_img)
    # final_results = coordinates_tensor[0]
    return final_results

def real_Gps_Coordination(input_img):
    gps = get_gps_from_image(input_img)
    return 31.262319, 34.802650

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