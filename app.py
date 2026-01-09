import gradio as gr
from functions_for_webpage import *


# --- The Logic Pipeline ---
def run_full_pipeline(image):
    """
    This function orchestrates the entire flow:
    Image -> Prediction -> Real GPS -> Distance -> Map
    """
    if image is None:
        return None, "No image uploaded", "", ""

    # 1. Get Model Prediction (Currently Dummy)
    pred_N, pred_E = predict(image)

    # 2. Get Real Location (Currently Dummy)
    real_N, real_E = real_Gps_Coordination(image)

    # 3. Calculate Distance
    # Note: Ensure calculate_Distance in your functions file handles the math correctly!
    dist_km = calculate_Distance((real_N, real_E), (pred_N, pred_E))

    # 4. Generate Map
    # The function needs to return the HTML string for Gradio to render it
    map_html = map_distance_visualizer((real_N, real_E), (pred_N, pred_E))

    return map_html, f"{dist_km:.2f} km", f"{real_N}, {real_E}", f"{pred_N}, {pred_E}"


# --- The Interface ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    # 1. Header & Intro
    gr.Markdown("# 🌍 GPS Location Predictor")
    gr.Markdown("""
    ### Project Introduction
    [Write your project description here. Explain that this model takes an image and predicts its exact GPS coordinates using EfficientNet.]
    """)

    gr.HTML("<hr>")  # Visual separator

    # 2. Main Content Area
    with gr.Row():
        # Left Column: Input
        with gr.Column(scale=1):
            gr.Markdown("### 1. Upload Image")
            # type="pil" ensures it works with your EXIF extraction code later
            input_img = gr.Image(label="Input Photo", type="pil", height=300)

            run_btn = gr.Button("🚀 Predict Location", variant="primary", size="lg")

            # Stats Group
            with gr.Group():
                gr.Markdown("### Statistics")
                dist_output = gr.Textbox(label="Error Distance (Haversine)", value="0 km")
                real_coords = gr.Textbox(label="True Coordinates (From EXIF)", value="Waiting...")
                pred_coords = gr.Textbox(label="Model Prediction", value="Waiting...")

        # Right Column: Map Output
        with gr.Column(scale=2):
            gr.Markdown("### 2. Location Visualization")
            # We use HTML component because Folium maps render as HTML
            map_output = gr.HTML(label="Interactive Map")

    # 3. Wiring
    run_btn.click(
        fn=run_full_pipeline,
        inputs=[input_img],
        outputs=[map_output, dist_output, real_coords, pred_coords]
    )

# Launch
if __name__ == "__main__":
    demo.launch()