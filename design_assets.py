import gradio as gr

def get_theme():
    return gr.themes.Base(
        primary_hue="violet",
        secondary_hue="indigo",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui"],
    ).set(
        body_background_fill="#0f111a", 
        block_background_fill="#1e2230", 
        block_border_width="1px",
        block_border_color="rgba(255, 255, 255, 0.1)",
        block_shadow="0 10px 30px rgba(0,0,0,0.5)",
        input_background_fill="rgba(0,0,0,0.3)", 
        button_primary_background_fill="linear-gradient(135deg, #6366f1 0%, #a855f7 100%)",
        button_primary_text_color="white",
        button_primary_shadow="0 0 15px rgba(168, 85, 247, 0.5)", 
    )

def get_css():
    return """
    body {
        background: radial-gradient(circle at 50% 0%, #2e1065 0%, #0f111a 60%);
        background-attachment: fixed;
    }

    /* --- HERO HEADER --- */
    .hero-container {
        text-align: center;
        padding: 40px 0 20px 0;
        border-bottom: 1px solid rgba(255,255,255,0.05);
        margin-bottom: 30px;
        display: flex;
        flex-direction: column;
        align-items: center;     
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
        max-width: 800px;
        margin-top: 15px;
        line-height: 1.6;
    }

    /* --- THE LAYOUT MAGIC (Flexbox Linking) --- */
    .dashboard-row {
        display: flex !important;
        align-items: stretch !important; /* This forces both columns to be equal height */
        gap: 20px;
    }
    /* Make the inner columns fill the height */
    .dashboard-row > div {
        display: flex;
        flex-direction: column;
        height: auto !important; 
    }

    /* --- LEFT SIDE: AUTO-GROWING TEXT --- */
    .stat-card {
        background: rgba(255, 255, 255, 0.03) !important;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 16px !important;
        padding: 15px !important;
        display: flex;
        flex-direction: column;
        justify-content: center;
        flex-grow: 0; /* Don't stretch the card itself unnecessarily */
    }
    .stat-card textarea {
        background: transparent !important;
        border: none !important;
        color: #e2e8f0 !important;
        font-size: 1.25rem !important; 
        font-weight: 700 !important;
        box-shadow: none !important;
        
        /* Auto-fit magic */
        height: auto !important; 
        min-height: 60px !important; /* Minimum space */
        white-space: pre-wrap !important; /* Wrap text naturally */
        overflow: visible !important;
        resize: vertical !important;
    }
    .stat-card label span {
        color: #94a3b8 !important;
        text-transform: uppercase;
        font-size: 0.75rem !important;
        letter-spacing: 1.5px;
        font-weight: 600;
        margin-bottom: 5px;
        display: block;
    }

    /* --- RIGHT SIDE: LIQUID MAP --- */
    .map-container {
        /* This is crucial: fill the available space provided by the neighbor column */
        height: 100% !important; 
        width: 100% !important;
        min-height: 600px; /* Fallback: don't get smaller than this */
        
        display: flex;
        flex-direction: column;
        border-radius: 20px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
    }
    
    /* Force the inner iframe to fill the container */
    .map-container > div, 
    .map-container iframe {
        height: 100% !important;
        width: 100% !important;
        flex-grow: 1;
        border: none;
    }

    /* --- BUTTONS & HEADERS --- */
    #predict-btn {
        font-weight: 700;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
    }
    #predict-btn:hover {
        transform: scale(1.02);
        box-shadow: 0 0 25px rgba(168, 85, 247, 0.7);
    }
    .section-header p {
        font-size: 1.1rem;
        color: #cbd5e1;
        font-weight: 600;
        margin-bottom: 10px;
    }
    """

def get_header_html():
    return """
    <div class="hero-container">
        <div class="hero-title">Campus Image-to-GPS Regression for
            Localization and Navigation</div>
        <div class="hero-subtitle">
            Utilizing an EfficientNet regression model to identify specific GPS locations from images taken within Ben-Gurion University
        </div>
    </div>
    """