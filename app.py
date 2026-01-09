import gradio as gr
from functions_for_webpage import *

# --- 1. Custom CSS for "Wow" Factor ---
# This hides the footer, centers the title, and gives the map a card-like effect
custom_css = """
.container {max-width: 1200px; margin: auto; padding-top: 20px;}
.header-text {text-align: center; font-family: 'Helvetica Neue', sans-serif;}
.header-text h1 {font-weight: 800; font-size: 3rem; background: linear-gradient(90deg, #4F46E5, #9333EA); -webkit-background-clip: text; -webkit-text-fill-color: transparent;}
.map-container {border-radius: 15px; border: 2px solid #e5e7eb; overflow: hidden; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);}
.stat-box {border: 1px solid #e5e7eb; border-radius: 10px; padding: 10px; background-color: #f9fafb;}
"""

# --- 2. The Logic Pipeline (Unchanged) ---
def run_full_pipeline(image):
    if image is None:
        return None, "No image uploaded", "", ""
    
    pred_N, pred_E = predict(image)
    real_N, real_E = real_Gps_Coordination(image)
    dist_km = calculate_Distance((real_N, real_E), (pred_N, pred_E))
    map_html = map_distance_visualizer((real_N, real_E), (pred_N, pred_E))
    
    return map_html, f"{dist_km:.2f} km", f"{real_N}, {real_E}", f"{pred_N}, {pred_E}"


# --- 3. The Designer Interface ---
# We use a curated theme for a cleaner look
theme = gr.themes.Default(
    primary_hue="indigo",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui"]
).set(
    button_primary_background_fill="linear-gradient(90deg, #4F46E5, #9333EA)",
    button_primary_background_fill_hover="linear-gradient(90deg, #4338ca, #7e22ce)",
    button_primary_text_color="white",
    block_shadow="0 4px 6px -1px rgba(0,0,0,0.1)",
    block_title_text_weight="700"
)

with gr.Blocks(theme=theme, css=custom_css, title="GeoLocator AI") as demo:
    
    # Header Section
    with gr.Row(elem_classes=["container"]):
        with gr.Column():
            gr.HTML("""
                <div class="header-text">
                    <h1>📍 GeoLocator AI</h1>
                    <p style="font-size: 1.2rem; color: #6b7280;">High-Precision GPS Regression via EfficientNet</p>
                </div>
            """)
    
    # Main Dashboard
    with gr.Row(elem_classes=["container"]):
        
        # --- LEFT PANEL: CONTROLS ---
        with gr.Column(scale=1, min_width=350):
            with gr.Group():
                gr.Markdown("### 📸 Image Input", elem_id="input-header")
                # Removed 'height' to let it adapt, added interactive=True explicit
                input_img = gr.Image(type="pil", label="Upload Street View", sources=["upload", "clipboard"], interactive=True)
                
                run_btn = gr.Button("Find Location 🌍", variant="primary", size="lg")

            # Statistics Section - Styled as cards
            gr.Markdown("### 📊 Inference Stats")
            with gr.Group():
                with gr.Row():
                    dist_output = gr.Textbox(label="Error Margin (Haversine)", value="0 km", elem_classes=["stat-box"])
                with gr.Row():
                    real_coords = gr.Textbox(label="True GPS (EXIF)", value="-", elem_classes=["stat-box"])
                    pred_coords = gr.Textbox(label="Predicted GPS", value="-", elem_classes=["stat-box"])

        # --- RIGHT PANEL: VISUALIZATION ---
        with gr.Column(scale=2):
             # We wrap the map in a Group to apply the custom CSS shadow class
            with gr.Group(elem_classes=["map-container"]):
                gr.Markdown("### 🗺️ Live Map Visualization")
                # Increased height for better visibility
                map_output = gr.HTML(label="Interactive Map", min_height=600)

    # Wiring
    run_btn.click(
        fn=run_full_pipeline, 
        inputs=[input_img], 
        outputs=[map_output, dist_output, real_coords, pred_coords]
    )

if __name__ == "__main__":
    demo.launch()