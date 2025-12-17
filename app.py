import sys
import os
import time

# -------------------- PATH SETUP --------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import torch
from PIL import Image
import numpy as np

from model_loader import load_model
from utils.preprocess import preprocess_image
from gradcam import GradCAM
from utils.visualization import overlay_cam
from config import CLASS_NAMES

# -------------------- STREAMLIT CONFIG --------------------
st.set_page_config(
    page_title="Brain Tumor Detection",
    layout="centered"
)

# -------------------- CUSTOM CSS (CLOUD SAFE) --------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(
        rgba(10, 25, 47, 0.92),
        rgba(10, 25, 47, 0.92)
    ),
    url("https://images.unsplash.com/photo-1582719478250-c89cae4dc85b");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

h1 {
    color: #E6F1FF;
    text-align: center;
    font-weight: 700;
}

h2, h3 {
    color: #64FFDA;
}

.block-container {
    padding-top: 2rem;
}

[data-testid="stFileUploader"] {
    background-color: rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 1rem;
}

.pred-box {
    background-color: rgba(255,255,255,0.12);
    padding: 1.2rem;
    border-radius: 14px;
    border-left: 6px solid #64FFDA;
    margin-top: 1.5rem;
}

.footer {
    text-align: center;
    color: #8892B0;
    font-size: 0.85rem;
    margin-top: 3rem;
}
</style>
""", unsafe_allow_html=True)

# -------------------- HERO SECTION --------------------
st.markdown("<h1>🧠 Brain Tumor Detection System</h1>", unsafe_allow_html=True)
st.markdown("""
<p style="text-align:center; color:#CCD6F6; font-size:1.1rem;">
AI-powered MRI analysis using <b>EfficientNet-B0</b> with <b>Explainable Grad-CAM</b><br>
Accurate • Fast • Interpretable • Cloud Deployed
</p>
""", unsafe_allow_html=True)

st.markdown("---")

# -------------------- DEVICE --------------------
DEVICE = torch.device("cpu")

# -------------------- MODEL PATH --------------------
MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "brain_tumor_efficientnet_b0_final.pth")

if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file not found on server.")
    st.code(MODEL_PATH)
    st.stop()

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_cached_model():
    model = load_model(MODEL_PATH)
    model.to(DEVICE)
    model.eval()
    return model

model = load_cached_model()

# -------------------- GRADCAM SETUP --------------------
try:
    target_layer = model.conv_head
except AttributeError:
    st.error("❌ Grad-CAM target layer not found in model.")
    st.stop()

gradcam = GradCAM(model, target_layer)

# -------------------- FILE UPLOAD --------------------
st.subheader("📤 Upload MRI Scan")

uploaded = st.file_uploader(
    "Supported formats: JPG, PNG, JPEG",
    type=["jpg", "png", "jpeg"]
)

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded MRI Image", width=320)

    input_tensor = preprocess_image(image).to(DEVICE)

    # -------------------- INFERENCE --------------------
    start_time = time.time()

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()

    inference_time = time.time() - start_time

    # -------------------- RESULTS --------------------
    st.markdown("<div class='pred-box'>", unsafe_allow_html=True)
    st.subheader("🧪 Prediction Result")

    st.write(f"**Tumor Type:** `{CLASS_NAMES[pred_idx]}`")
    st.write(f"**Confidence:** `{probs[0][pred_idx] * 100:.2f}%`")
    st.progress(float(probs[0][pred_idx]))
    st.write(f"⏱ **Inference Time:** `{inference_time:.3f} seconds`")

    st.markdown("</div>", unsafe_allow_html=True)

    # -------------------- CLASS PROBABILITIES --------------------
    st.subheader("📊 Class-wise Prediction Probabilities")

    prob_dict = {
        CLASS_NAMES[i]: float(probs[0][i]) * 100
        for i in range(len(CLASS_NAMES))
    }

    st.bar_chart(prob_dict)

    # -------------------- GRAD-CAM --------------------
    st.subheader("🔍 Model Explainability (Grad-CAM)")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Original MRI", use_container_width=True)

    with col2:
        cam = gradcam.generate(input_tensor, pred_idx)
        heatmap = overlay_cam(image, cam)
        st.image(heatmap, caption="Grad-CAM Heatmap", use_container_width=True)

    # -------------------- CLINICAL INTERPRETATION --------------------
    st.subheader("🧠 Clinical Interpretation")

    st.write(f"""
The AI system predicts **{CLASS_NAMES[pred_idx]}** with a confidence of 
**{probs[0][pred_idx] * 100:.2f}%**.

The Grad-CAM visualization highlights the MRI regions that most influenced 
the model’s decision, improving transparency and trust.

⚠️ *This system is intended for clinical decision support only and must not 
replace professional medical diagnosis.*
""")

# -------------------- FOOTER --------------------
st.markdown("""
<div class="footer">
© 2025 | Brain Tumor Detection System<br>
Department of Artificial Intelligence<br>
The Islamia University of Bahawalpur
</div>
""", unsafe_allow_html=True)
