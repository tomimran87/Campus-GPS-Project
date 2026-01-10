import gradio as gr
from functions_for_webpage import *
import design_assets as design

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
    
    # 1. Prediction & Real Data
    pred_N, pred_E = predict(image)
    real_N, real_E = real_Gps_Coordination(image)
    
    # 2. Math & Vis
    dist_km = calculate_Distance((real_N, real_E), (pred_N, pred_E))
    map_html = map_distance_visualizer((real_N, real_E), (pred_N, pred_E))
    
    # Formatting the output for the "HUD" look
    return map_html, f"{dist_km:.3f} km", f"{real_N:.4f}, {real_E:.4f}", f"{pred_N:.4f}, {pred_E:.4f}"

    

with gr.Blocks(theme=design.get_theme(), css=design.get_css(), title="Orbital Vision AI") as demo:
    
    # 1. Custom HTML Header (The "Hero" Section)
    gr.HTML(design.get_header_html())
    
    with gr.Row():
        # --- LEFT: COMMAND CENTER ---
        with gr.Column(scale=4, min_width=350):
            
            # Input Area
            gr.Markdown("### 📡 Uplink", elem_classes=["section-header"])
            input_img = gr.Image(
                type="pil", 
                label="Sensor Data (Image)", 
                height=320,
                elem_classes=["image-container"]
            )
            
            # Action Button
            run_btn = gr.Button("INITIALIZE SCAN 🚀", variant="primary", elem_id="predict-btn", size="lg")
            
            # HUD Stats (Heavily styled via CSS)
            gr.HTML("<br>") # Spacer
            gr.Markdown("### 📊 Telemetry", elem_classes=["section-header"])
            
            with gr.Group():
                # We use the 'stat-card' CSS class defined in design_assets.py
                dist_output = gr.Textbox(label="Deviation (Haversine)", value="0.000 km", elem_classes=["stat-card"])
                
                with gr.Row():
                    real_coords = gr.Textbox(label="Ground Truth", value="--", elem_classes=["stat-card"])
                    pred_coords = gr.Textbox(label="AI Est.", value="--", elem_classes=["stat-card"])

        # --- RIGHT: VISUALIZATION ---
        with gr.Column(scale=7):
            # The Map
            map_output = gr.HTML(
                value=get_default_map(), # Ensure this function is in functions_for_webpage.py
                label="Geospatial Lock",
                elem_classes=["map-container"],
                min_height=750 # Taller map for "Command Center" feel
            )

    # Wiring
    run_btn.click(
        fn=run_full_pipeline, 
        inputs=[input_img], 
        outputs=[map_output, dist_output, real_coords, pred_coords]
    )

if __name__ == "__main__":
    demo.launch()