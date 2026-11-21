import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="Prahari helmet detection", page_icon="🪖")
st.title("Prahari: helmet detection")
st.caption("YOLO11s at 960 px, trained on Kaggle andrewmvd/helmet-detection + EdgeVision Indian phone photos. "
           "Test mAP50: andrewmvd 0.835, EdgeVision 0.921")


@st.cache_resource
def load():
    return YOLO("yolo11s_960_best.pt")


# ponytail: 0.25 default finds 46/54 no-helmet riders on the andrewmvd test vs 41 at 0.4, for 45 vs 26 false alarms
# (training/results/better/metrics.csv); no boxes even at 0.05 on black or dark-noise frames
conf = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
for f in st.file_uploader("Upload road photos", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True) or []:
    r = load().predict(Image.open(f).convert("RGB"), conf=conf, imgsz=960, verbose=False)[0]
    st.image(r.plot(), channels="BGR", caption=f"{f.name}: {r.verbose().rstrip(', ')}")
