import sys
import os

# Add project root to PYTHONPATH
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
from config import CLASS_NAMES, DEVICE

st.set_page_config(page_title="Brain Tumor Detection", layout="centered")

st.title("🧠 Brain Tumor Detection System")
st.caption("EfficientNet-B0 with Explainable Grad-CAM")

# Load model
@st.cache_resource
def load_cached_model():
    return load_model(r"C:\Users\SONIC LAPTOPS\Desktop\brain tumor\model\brain_tumor_efficientnet_b0_final.pth")

model = load_cached_model()
gradcam = GradCAM(model, model.conv_head)

uploaded = st.file_uploader("Upload MRI Image", type=["jpg","png","jpeg"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded MRI", width=300)

    input_tensor = preprocess_image(image).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs).item()

    st.subheader("Prediction Result")
    st.write(f"**Tumor Type:** {CLASS_NAMES[pred_idx]}")
    st.write(f"**Confidence:** {probs[0][pred_idx]*100:.2f}%")

    cam = gradcam.generate(input_tensor, pred_idx)
    heatmap = overlay_cam(image, cam)

    st.subheader("Grad-CAM Explainability")
    st.image(heatmap, caption="Model Focus Area", width=300)
