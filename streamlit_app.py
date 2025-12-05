# Import required libraries
import PIL
import streamlit as st
from ultralytics import YOLO
import time
import pandas as pd
import datetime

# Replace with your model path
model_path = 'weights/t29.pt'  # T29 weight file

# Streamlit page config
st.set_page_config(
    page_title="Microplastic Detection using YOLOv8",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar
with st.sidebar:
    st.header("Upload Image")
    source_img = st.file_uploader("Choose an image...",
                                  type=("jpg", "jpeg", "png", "bmp", "webp"))

    confidence = float(st.slider("Select Model Confidence", 25, 100, 40)) / 100

# Main Title
st.title("🌊 Microplastic Detection using YOLOv8")
st.write("This application detects and classifies different types of microplastics using a YOLOv8 model.")

# GitHub Link
st.markdown("""
### 🔗 GitHub Repository  
<a href='https://github.com/InnocentPerson/microplastic-detection' target='_blank'>
Click here to view the project on GitHub
</a>
""", unsafe_allow_html=True)

# Columns for input/output
col1, col2 = st.columns(2)

# Display uploaded image
with col1:
    if source_img:
        uploaded_image = PIL.Image.open(source_img)
        st.image(uploaded_image, caption="Uploaded Image", width=600)

# Load YOLO model
try:
    model = YOLO(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check path: {model_path}")
    st.error(ex)

# Detect button
if st.sidebar.button("Detect Objects"):

    if not source_img:
        st.error("Please upload an image first!")
    else:
        start_time = time.time()

        # Prediction
        res = model.predict(uploaded_image, conf=confidence)

        end_time = time.time()
        prediction_time = end_time - start_time

        boxes = res[0].boxes
        class_idx = res[0].boxes.cls.cpu().numpy().astype(int)

        # Label mapping
        label_names = {
            0: 'Fibers',
            1: 'Films',
            2: 'Fragments',
            3: 'Pallets'
        }

        # Counting detections
        label_counts = {}
        for label in class_idx:
            label_name = label_names.get(label, 'Unknown')
            label_counts[label_name] = label_counts.get(label_name, 0) + 1

        # Plot detections
        res_plotted = res[0].plot()[:, :, ::-1]

        # Display detected image & results
        with col2:
            st.image(res_plotted, caption='Detected Image', width=600)

            st.subheader("📌 Microplastic Count Summary")
            for idx, (label, count) in enumerate(label_counts.items()):
                st.write(f"**{label}: {count}**")

            # Total microplastics
            total_microplastics = sum(label_counts.values())

            # Pollution level logic
            if total_microplastics <= 10:
                pollution = "🟢 Low contamination"
                color = "green"
            elif total_microplastics <= 30:
                pollution = "🟡 Medium contamination"
                color = "orange"
            else:
                pollution = "🔴 High contamination"
                color = "red"

            st.subheader("🌡 Pollution Level")
            st.markdown(
                f"<h4 style='color:{color}'>Total Microplastics: {total_microplastics} → {pollution}</h4>",
                unsafe_allow_html=True,
            )

            # Prediction time
            st.write(f"⏱ Prediction Time: {prediction_time:.2f} seconds")

            # Detection box details
            with st.expander("Detection Results (Raw YOLO Output)"):
                for box in boxes:
                    st.write(box.xywh)

            # Save results to CSV
            df = pd.DataFrame([label_counts])
            df["filename"] = source_img.name
            df["timestamp"] = datetime.datetime.now()

            df.to_csv("microplastic_detection_log.csv", mode="a",
                      index=False, header=False)

            st.success("Results saved to microplastic_detection_log.csv")
