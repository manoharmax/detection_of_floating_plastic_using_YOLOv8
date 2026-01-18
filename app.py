import streamlit as st
from ultralytics import YOLO
import os
import cv2
import numpy as np
from PIL import Image
import tempfile
import pandas as pd

# Load YOLOv8 model
@st.cache_resource
def load_model():
    return YOLO("runs/detect/train_yolov8x_balanced_50ep_resized3/weights/best.pt")

model = load_model()

# Streamlit UI config
st.set_page_config(page_title="Plastic Detection", layout="wide")
st.title("🧼 Detection and Monitoring of Floating Plastic Debris")
st.markdown("Upload image(s), adjust detection settings, and run the plastic detection model.")

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Settings")
    conf = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
    iou = st.slider("IoU Threshold", 0.0, 1.0, 0.45, 0.05)
    zoom_option = st.checkbox("🔍 Enable Zoom-based Tiling (512x512 patches)")
    show_metrics = st.checkbox("📊 Show Detection Metrics Table")
    run = st.button("🔎 Run Detection")

# Image uploader
uploaded_files = st.file_uploader("📁 Upload Image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# Begin detection
if run and uploaded_files:
    results_data = []

    for file in uploaded_files:
        img = Image.open(file).convert("RGB")
        img_array = np.array(img)

        detections = []

        if zoom_option:
            # Tile image into 512x512 patches
            h, w = img_array.shape[:2]
            for y in range(0, h, 512):
                for x in range(0, w, 512):
                    tile = img_array[y:y+512, x:x+512]
                    if tile.shape[0] == 512 and tile.shape[1] == 512:
                        tile_result = model.predict(source=tile, conf=conf, iou=iou, show=False, verbose=False)
                        detections.extend(tile_result)
        else:
            result = model.predict(source=img_array, conf=conf, iou=iou, show=False, verbose=False)
            detections = result

        # Display results
        for result in detections:
            # Force all labels to 'plastic'
            result.names = {i: "plastic" for i in result.names}

            # Plot image with modified labels
            img_out = result.plot(labels=True)
            st.image(img_out, caption="🧠 Detection Output", use_column_width=True)

            # Detection stats
            stats = {
                "File": file.name,
                "Detections": len(result.boxes),
                "Avg Confidence": round(result.boxes.conf.mean().item(), 2) if len(result.boxes) else 0
            }
            results_data.append(stats)

    # Show metrics table
    if show_metrics and results_data:
        df = pd.DataFrame(results_data)
        st.markdown("### 📈 Detection Summary")
        st.dataframe(df)

else:
    st.info("Upload image(s) and click 'Run Detection' to begin.")
