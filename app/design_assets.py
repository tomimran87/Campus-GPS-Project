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

    /* --- LAYOUT LINKING --- */
    .dashboard-row {
        display: flex !important;
        align-items: stretch !important;
        gap: 20px;
    }
    .dashboard-row > div {
        display: flex;
        flex-direction: column;
        height: auto !important; 
    }

    /* --- SECTION HEADERS (CENTERED) --- */
    /* This forces your headlines to be centered */
    .section-header p, .section-header h3 {
        text-align: center !important; 
        color: #e0e7ff !important;
        font-weight: 700;
        margin-bottom: 10px;
        font-size: 1.2rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        width: 100%;
    }

    /* --- STAT CARDS --- */
    .stat-card {
        background: rgba(255, 255, 255, 0.03) !important;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 16px !important;
        padding: 15px !important;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .stat-card textarea {
        background: transparent !important;
        border: none !important;
        color: #e2e8f0 !important;
        font-size: 1.25rem !important; 
        font-weight: 700 !important;
        box-shadow: none !important;
        height: auto !important; 
        min-height: 60px !important; 
        white-space: pre-wrap !important; 
        overflow: hidden !important;
        resize: none !important;
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

    /* --- MAP CONTAINER (FIXED) --- */
    .map-container {
        flex-grow: 1;
        display: flex;
        flex-direction: column;
        border-radius: 20px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
    }
    
    /* FIX: Force the map inside to have a minimum height so it never disappears */
    .map-container iframe {
        min-height: 800px !important; /* <--- THIS IS THE FIX FOR THE MISSING MAP */
        width: 100% !important;
        height: 100% !important;
        border: none !important;
        display: block !important;
    }

    #predict-btn {
        font-weight: 700;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
    }
    #predict-btn:hover {
        transform: scale(1.02);
        box-shadow: 0 0 25px rgba(168, 85, 247, 0.7);
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