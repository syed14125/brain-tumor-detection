import sys
import os

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

st.title("Brain Tumor Detection System")
st.caption("EfficientNet-B0 with Explainable Grad-CAM")

# -------------------- DEVICE (CPU ONLY) --------------------
DEVICE = torch.device("cpu")

# -------------------- MODEL PATH (RELATIVE) --------------------
MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "brain_tumor_efficientnet_b0_final.pth")

# -------------------- MODEL EXISTENCE CHECK --------------------
if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file not found.")
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
    st.error("❌ Grad-CAM target layer not found (model.conv_head).")
    st.stop()

gradcam = GradCAM(model, target_layer)

# -------------------- FILE UPLOAD --------------------
uploaded = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded MRI", width=300)

    input_tensor = preprocess_image(image).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()

    st.subheader("Prediction Result")
    st.write(f"**Tumor Type:** {CLASS_NAMES[pred_idx]}")
    st.write(f"**Confidence:** {probs[0][pred_idx] * 100:.2f}%")

    # -------------------- GRAD-CAM --------------------
    cam = gradcam.generate(input_tensor, pred_idx)
    heatmap = overlay_cam(image, cam)

    st.subheader("Grad-CAM Explainability")
    st.image(heatmap, caption="Model Focus Area", width=300)
