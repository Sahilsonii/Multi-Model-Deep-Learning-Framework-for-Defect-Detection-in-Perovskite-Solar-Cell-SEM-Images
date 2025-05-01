import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
from streamlit.components.v1 import html
import base64
from io import BytesIO
import os

# Load YOLOv8 model
model = YOLO(r"C:\Users\ASUS\Desktop\New folder (2)\model training with Ultralytics Yolo v8\best_model.pt")

class_names = [
    "3D perovskite",
    "3D perovskite with PbI2 excess",
    "3D perovskite with pinholes",
    "3D-2D mixed perovskite",
    "3D-2D mixed perovskite with pinholes"
]

class_descriptions = {
    "3D perovskite": "🎯 Pure 3D crystalline structures used in photovoltaic layers. Known for high efficiency and good stability.",
    "3D perovskite with PbI2 excess": "⚠️ Contains unreacted PbI₂, indicating incomplete conversion—may impact performance.",
    "3D perovskite with pinholes": "🔍 Features microscopic pinholes that can cause leakage currents and lowered efficiency.",
    "3D-2D mixed perovskite": "💧 A blend of 3D and layered 2D perovskites, improving moisture resistance and enhancing film quality. They balance performance with durability.",
    "3D-2D mixed perovskite with pinholes": "🕳 Mixed dimensional structures with visible pinhole defects. While structurally robust, these defects can still reduce functional surface area and efficiency."
}

def get_prediction(img: Image.Image) -> int:
    arr = np.array(img)
    res = model.predict(arr, verbose=False)
    return int(res[0].probs.top1)

def image_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# Read HTML content from file
def load_html_content(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        return "<div style='color: red;'>Error: HTML file not found at specified path.</div>"
    except Exception as e:
        return f"<div style='color: red;'>Error loading HTML file: {str(e)}</div>"

html_file_path = r"C:\Users\ASUS\Desktop\New folder (2)\model training with Ultralytics Yolo v8\classess_info.html"
html_content = load_html_content(html_file_path)

# Streamlit page config
st.set_page_config(page_title="Perovskite Classifier", layout="wide")

# — Custom CSS —
st.markdown("""
<style>
body { font-family: 'Segoe UI', sans-serif; background: #0f1117; }
/* Title styling */
.title-box {
    text-align: center;
    background: #202531;
    border: 2px solid #256D85;
    border-radius: 30px;
    padding: 1rem 2.5rem;
    margin-top: 1rem;
    margin-bottom: 1.5rem;
    color: #00BFFF;
    font-size: 2.5rem;
    font-weight: 700;
    font-family: 'Montserrat', 'Roboto', sans-serif;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
}
/* Floating banner effect */
.banner-box {
    animation: floatBanner 3s ease-in-out infinite;
    transition: transform 0.3s;
}
.banner-box:hover {
    transform: scale(1.01);
}
@keyframes floatBanner {
    0% { transform: translateY(0); }
    50% { transform: translateY(-5px); }
    100% { transform: translateY(0); }
}
/* Upload box */
.upload-box {
    border: 2px dashed #256D85;
    padding: 1.2rem;
    border-radius: 10px;
    background: #151a28;
    color: #eee;
    margin-bottom: 2rem;
}
/* Prediction and description */
.result-card, .description-card, .info-card {
    background: #ffffff;
    padding: 1.2rem;
    border-radius: 12px;
    box-shadow: 0 3px 10px rgba(0,0,0,0.08);
    margin-top: 1.5rem;
}
.result-card h2 {
    font-size: 1.2rem;
    color: #1e78e6;
}
.description-card h3, .info-card h3 {
    font-size: 1.05rem;
    color: #256D85;
    margin-bottom: 0.5rem;
}
.description-card p {
    font-size: 0.95rem;
    color: #333;
}
img.pred-img {
    width: 100%;
    height: auto;
    border-radius: 10px;
    display: block;
    margin: 1rem auto;
}
.footer {
    text-align: center;
    font-size: 0.85rem;
    color: #aaa;
    padding: 1rem;
}
</style>
""", unsafe_allow_html=True)

# — HEADER —
st.markdown('<div class="title-box">🧪 Perovskite Classification</div>', unsafe_allow_html=True)

# — Banner image —
st.markdown('<div class="banner-box">', unsafe_allow_html=True)
st.image("sample images.png", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# — Upload section —
# st.markdown('<div class="upload-box">', unsafe_allow_html=True)
uploaded = st.file_uploader("📎 Upload a microscope image...", type=["jpg", "jpeg", "png", "tif", "bmp"])
st.markdown('</div>', unsafe_allow_html=True)

# — Prediction section —
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    if st.button("🎯 Predict"):
        with st.spinner("Analyzing..."):
            idx = get_prediction(img)
            label = class_names[idx]
            desc = class_descriptions[label]
            img_b64 = image_to_base64(img)

        st.markdown(f"""
        <div class="result-card">
            <h2>🔬 Predicted Class: {label}</h2>
            <img class="pred-img" src="data:image/png;base64,{img_b64}" style="width: 100%; height: auto;">
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="description-card">
            <h3>📘 About “{label}”</h3>
            <p>{desc}</p>
        </div>
        """, unsafe_allow_html=True)

        # Embed HTML content
        st.markdown(f"""
        <div class="info-card">
            <h3>📚 Additional Class Information</h3>
        </div>
        """, unsafe_allow_html=True)
        html(html_content, height=500, scrolling=True)

else:
    st.info("Upload an image to classify it.")

# — Footer —
st.markdown('<div class="footer">© 2025 Perovskite Research Lab • Built with 💖 & Streamlit</div>', unsafe_allow_html=True)