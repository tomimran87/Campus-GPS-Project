from predict import predict_gps
import numpy as np
from PIL import Image

# Put the path to the image here:
img_path = "../dataset_root/images/IMG_2322.JPG"
image = np.array(Image.open(img_path).convert('RGB'))

gps_prediction = predict_gps(image)

print(f"predicted gps: {gps_prediction}")