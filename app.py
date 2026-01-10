import gradio as gr
from functions_for_webpage import *
import design_assets as design 

def run_full_pipeline(image):
    if image is None:
        return None, "No image uploaded", "", ""
    
    pred_N, pred_E = predict(image)
    real_N, real_E = real_Gps_Coordination(image)
    dist_km = calculate_Distance((real_N, real_E), (pred_N, pred_E))
    map_html = map_distance_visualizer((real_N, real_E), (pred_N, pred_E))
    
    # Return formatted strings
    return map_html, f"{dist_km:.3f} km", f"{real_N:.4f}, {real_E:.4f}", f"{pred_N:.4f}, {pred_E:.4f}"

with gr.Blocks(theme=design.get_theme(), css=design.get_css(), title="GeoSpatial Core") as demo:
    
    gr.HTML(design.get_header_html())
    
    with gr.Row():
        # --- LEFT PANEL ---
        with gr.Column(scale=4, min_width=350):
            
            gr.Markdown("### 📡 Visual Input", elem_classes=["section-header"])
            input_img = gr.Image(
                type="filepath", 
                label="Sensor Feed", 
                height=320,
                elem_classes=["image-container"]
            )
            
            run_btn = gr.Button("RUN INFERENCE 🚀", variant="primary", elem_id="predict-btn", size="lg")
            
            gr.HTML("<br>") 
            
            # UPDATED TERMINOLOGY HERE
            gr.Markdown("### 📊 Inference Metrics", elem_classes=["section-header"])
            
            with gr.Group():
                # "Spatial Offset" sounds very tech-forward
                dist_output = gr.Textbox(label="Spatial Offset (Error)", value="0.000 km", elem_classes=["stat-card"])
                
                with gr.Row():
                    real_coords = gr.Textbox(label="Ground Truth", value="--", elem_classes=["stat-card"])
                    pred_coords = gr.Textbox(label="Model Estimate", value="--", elem_classes=["stat-card"])

        # --- RIGHT PANEL ---
        with gr.Column(scale=7):
            map_output = gr.HTML(
                value=get_default_map(), 
                label="Geospatial Lock",
                elem_classes=["map-container"],
                min_height=900 
            )

    run_btn.click(
        fn=run_full_pipeline, 
        inputs=[input_img], 
        outputs=[map_output, dist_output, real_coords, pred_coords]
    )

if __name__ == "__main__":
    demo.launch()