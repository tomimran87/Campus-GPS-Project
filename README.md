# 🌍 Campus Image-to-GPS Regression

### 🔗 [Click Here to Launch the Live App](https://huggingface.co/spaces/liranatt/GPU_Modell_Liran_and_Tom)

### Project Overview
This project implements a **Deep Learning-based Visual Localization System** designed to predict precise GPS coordinates (Latitude, Longitude) from a single ground-level image. Unlike traditional retrieval-based methods, this system utilizes a **Regression-based approach** powered by an **EfficientNet-B7** backbone, allowing for continuous coordinate prediction rather than discrete classification.

The model was trained on a custom dataset collected within the **Ben-Gurion University** campus, demonstrating the feasibility of autonomous navigation and localization in GPS-denied environments using purely visual data.

### 🚀 Key Technologies
* **Architecture:** EfficientNet-B7 (Pre-trained on ImageNet, Fine-tuned for Regression)
* **Framework:** PyTorch
* **Interface:** Gradio (Custom CSS/HTML Design)
* **Visualization:** Folium & Leaflet.js for interactive mapping
* **Data Processing:** Pillow-HEIF for raw mobile sensor data

### 📊 Performance Metrics
The system evaluates performance using the **Haversine Formula** to calculate the Great Circle distance between the predicted coordinates and the ground truth EXIF metadata. 

### 👥 Authors
Developed by **Liran Attar** and **Tom Mimran** as part of the Computer Science Department research track.
