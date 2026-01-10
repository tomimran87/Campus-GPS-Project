import gradio as gr

def get_theme():
    """
    Creates a custom 2026-style theme based on the 'Base' theme
    but overridden with deep-space colors and neon accents.
    """
    return gr.themes.Base(
        primary_hue="violet",
        secondary_hue="indigo",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui"],
    ).set(
        body_background_fill="#0f111a",  # Very deep slate/black
        block_background_fill="#1e2230", # Slightly lighter for cards
        block_border_width="1px",
        block_border_color="rgba(255, 255, 255, 0.1)",
        block_shadow="0 10px 30px rgba(0,0,0,0.5)",
        input_background_fill="rgba(0,0,0,0.3)", # Semi-transparent inputs
        button_primary_background_fill="linear-gradient(135deg, #6366f1 0%, #a855f7 100%)",
        button_primary_text_color="white",
        button_primary_shadow="0 0 15px rgba(168, 85, 247, 0.5)", # Neon glow
    )

def get_css():
    """
    Injects aggressive CSS for animations, glassmorphism, and layout.
    """
    return """
    /* --- ANIMATED BACKGROUND --- */
    body {
        background: radial-gradient(circle at 50% 0%, #2e1065 0%, #0f111a 60%);
        background-attachment: fixed;
    }

    /* --- TYPOGRAPHY --- */
    .hero-container {
        text-align: left;
        padding: 40px 0 20px 0;
        border-bottom: 1px solid rgba(255,255,255,0.05);
        margin-bottom: 30px;
    }
    .hero-title {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        font-size: 4rem;
        background: linear-gradient(to right, #c4b5fd, #818cf8, #2dd4bf);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -2px;
        line-height: 1.1;
    }
    .hero-subtitle {
        font-size: 1.25rem;
        color: #94a3b8;
        font-weight: 300;
        max-width: 600px;
        margin-top: 10px;
    }

    /* --- GLASS CARDS (STATS) --- */
    .stat-card {
        background: rgba(255, 255, 255, 0.03) !important;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 16px !important;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .stat-card:hover {
        border-color: rgba(139, 92, 246, 0.5) !important;
        transform: translateY(-2px);
    }
    .stat-card textarea {
        background: transparent !important;
        border: none !important;
        color: #e2e8f0 !important;
        font-size: 1.4rem !important;
        font-weight: 700 !important;
        box-shadow: none !important;
        padding-top: 5px !important;
    }
    .stat-card label span {
        color: #64748b !important;
        text-transform: uppercase;
        font-size: 0.75rem !important;
        letter-spacing: 1.5px;
        font-weight: 600;
    }

    /* --- MAP CONTAINER --- */
    .map-container {
        border-radius: 20px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        background: #1e2230;
    }

    /* --- BUTTONS --- */
    #predict-btn {
        font-weight: 700;
        letter-spacing: 0.5px;
        border: none;
        transition: all 0.3s ease;
    }
    #predict-btn:hover {
        transform: scale(1.02);
        box-shadow: 0 0 25px rgba(168, 85, 247, 0.7);
    }
    
    /* --- IMAGE UPLOAD --- */
    .image-container button {
        border-radius: 16px;
    }
    """

def get_header_html():
    return """
    <div class="hero-container">
        <div class="hero-title">Orbital Vision</div>
        <div class="hero-subtitle">
            Next-generation geospatial regression engine powered by EfficientNet-B7. 
            Pinpointing locations from ground-level imagery with sub-15m accuracy.
        </div>
    </div>
    """