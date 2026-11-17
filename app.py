import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="Prahari helmet detection", page_icon="🪖")
st.title("Prahari: helmet detection")
st.caption("YOLO11n trained on Kaggle andrewmvd/helmet-detection, test mAP50 0.831, mAP50-95 0.481")


@st.cache_resource
def load():
    return YOLO("yolo11n_best.pt")


# ponytail: 0.4 default because 0.25 boxes phantom helmets in dark frame corners (seen up to 0.39 on an all-black image)
conf = st.slider("Confidence threshold", 0.05, 0.95, 0.4, 0.05)
for f in st.file_uploader("Upload road photos", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True) or []:
    r = load().predict(Image.open(f).convert("RGB"), conf=conf, imgsz=640, verbose=False)[0]
    st.image(r.plot(), channels="BGR", caption=f"{f.name}: {r.verbose().rstrip(', ')}")
