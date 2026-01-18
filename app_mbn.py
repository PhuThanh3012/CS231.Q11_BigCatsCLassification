import streamlit as st
import numpy as np
import pandas as pd
import joblib
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

# --------------------------------------------------------
# CONFIG
# --------------------------------------------------------
st.set_page_config(page_title="Big Cats Classifier", layout="wide")
st.title("🐯🦁 Big Cats Classification")

CLASSES = ["cheetah", "leopard", "lion", "tiger"]

# --------------------------------------------------------
# LOAD ONLY MOBILENET + RANDOM FOREST MODEL
# --------------------------------------------------------
MODEL_PATH = "models/MobileNetV3_Small_rf_model.pkl"

try:
    mobile_rf = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"❌ Cannot load model: {e}")
    st.stop()

# --------------------------------------------------------
# FEATURE EXTRACTOR — MobileNetV3
# --------------------------------------------------------
mobilenet = models.mobilenet_v3_small(pretrained=True)
mobilenet.classifier = nn.Identity()  # remove final FC -> output feature vector
mobilenet.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize( mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225] )
])

def extract_features(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    x = transform(pil).unsqueeze(0)

    with torch.no_grad():
        feat = mobilenet(x).cpu().numpy().flatten()

    return feat

# --------------------------------------------------------
# PREDICT
# --------------------------------------------------------
def predict(img):
    feat = extract_features(img)
    pred = mobile_rf.predict([feat])[0]
    prob = mobile_rf.predict_proba([feat])[0]
    return pred, prob

# --------------------------------------------------------
# UI — UPLOAD + OUTPUT
# --------------------------------------------------------
uploaded = st.file_uploader("📤 Upload big cat image...", type=["jpg", "jpeg", "png"])

if uploaded:
    pil = Image.open(uploaded).convert("RGB")
    img = np.array(pil)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    st.image(pil, caption="Input Image", width=380)

    pred, prob = predict(img_bgr)

    st.subheader("🔍 Result")
    st.success(f"Predicted Class: **{CLASSES[pred].upper()}**")

    st.subheader("📊 Confidence Table")
    df = pd.DataFrame({
        "Class": CLASSES,
        "Confidence": np.round(prob, 4)
    }).sort_values("Confidence", ascending=False)

    st.dataframe(df, height=200)

st.markdown("---")
st.caption("Big Cats Classifier 🐾")
