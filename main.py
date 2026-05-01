import streamlit as st
from PIL import Image
from fetch_satellite import get_fire_data
from model_utils import load_model, predict_image

st.set_page_config(layout="wide")
st.title("🔥 Forest Fire Detection (Satellite + AI)")

# ---------------- METRICS ----------------
try:
    with open("metrics.txt", "r") as f:
        acc, prec, rec = map(float, f.read().split(","))

    st.subheader("📊 Model Performance")
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{acc*100:.2f}%")
    c2.metric("Precision", f"{prec*100:.2f}%")
    c3.metric("Recall", f"{rec*100:.2f}%")
except:
    st.warning("⚠ Train model to see metrics")

# ---------------- SATELLITE MAP (FAST) ----------------
st.header("🌍 Live Fire Map")

data = get_fire_data()

if data is not None:
    st.success(f"🔥 {len(data)} Active Fires Detected")

    # Fast map
    st.map(data[['latitude', 'longitude']])

    # Simple table
    st.subheader("Top Fire Points")
    st.dataframe(data[['latitude', 'longitude', 'confidence']].head(20))

else:
    st.error("⚠ API issue")

# ---------------- MODEL ----------------
model = None
try:
    model = load_model()
except:
    st.warning("⚠ Model not loaded")

# ---------------- PREDICTION ----------------
st.header("🧠 Image Detection")

uploaded = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded and model:
    img = Image.open(uploaded)
    st.image(img)

    label, conf = predict_image(model, img)

    st.success(f"Prediction: {label}")
    st.metric("Confidence", f"{conf*100:.2f}%")