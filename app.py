import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import time
import os

# --------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------
st.set_page_config(
    page_title="Deepfake Image Detector",
    page_icon="🧠",
    layout="wide",
)

# --------------------------------------------------------
# FIXED CSS (DARK PREMIUM UI)
# --------------------------------------------------------
st.markdown("""
<style>

.stApp {
    background-color: #101010;
}

/* HEADINGS */
h1, h2, h3 {
    text-align: center;
    font-weight: 700;
    color: white;
    animation: fadeIn 1.3s ease-in;
}

/* SUBTEXT */
p, span, label {
    color: #ddd !important;
}

/* GRADIENT TITLE */
.title-box {
    padding: 20px;
    border-radius: 12px;
    background: linear-gradient(135deg, #503C3C, #2e2222);
    box-shadow: 0 0 20px rgba(0,0,0,0.5);
    animation: fadeInDrop 1s ease-out;
}

/* CARDS */
.metric-card {
    background-color: #1b1b1b;
    padding: 18px;
    border-radius: 12px;
    box-shadow: 0 0 15px rgba(0,0,0,0.4);
    transition: 0.2s;
}
.metric-card:hover {
    transform: translateY(-5px);
}

/* BUTTON */
.stButton>button {
    background: linear-gradient(135deg, #503C3C, #382a2a);
    color: white;
    border-radius: 8px;
    padding: 10px 20px;
    border: none;
}
.stButton>button:hover {
    background: linear-gradient(135deg, #705050, #4b3838);
}

/* ANIMATIONS */
@keyframes fadeIn {
    from {opacity: 0;} to {opacity: 1;}
}
@keyframes fadeInDrop {
    0% {opacity: 0; transform: translateY(-20px);}
    100% {opacity:1; transform: translateY(0);}
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------
# SIDEBAR
# --------------------------------------------------------
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["🏠 Home", "📊 Evaluation Metrics", "🔥 Grad-CAM++", "ℹ About"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("Made by **Anushka • ML Engineer**")

# --------------------------------------------------------
# MODEL LOADING
# --------------------------------------------------------
@st.cache_resource()
def load_model():
    return tf.keras.models.load_model("/mnt/c/Deepfake Image Detection/saved_model/final_finetuned_model.h5")  # CHANGE THIS

model = load_model()

# --------------------------------------------------------
# PAGE 1 — HOME
# --------------------------------------------------------
if page == "🏠 Home":

    st.markdown("<div class='title-box'><h1>Deepfake Image Detection</h1></div>", unsafe_allow_html=True)
    st.write("### ⚠ Upload an image to analyze whether it is Real or Fake")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        with st.spinner("Analyzing image..."):
            time.sleep(1.2)

            img = image.resize((128,128))
            img = np.array(img) / 255.0
            img = np.expand_dims(img, axis=0)

            pred = model.predict(img)[0][0]
            label = "FAKE" if pred >= 0.5 else "REAL"
            conf = pred if pred >= 0.5 else 1 - pred
            conf *= 100

        st.markdown(f"""
            <div class="metric-card">
                <h2>Prediction: <span style="color:#ff5050;">{label}</span></h2>
                <h3>Confidence: {conf:.2f}%</h3>
            </div>
        """, unsafe_allow_html=True)

# --------------------------------------------------------
# Page 2 — Evaluation Metrics
# --------------------------------------------------------
elif page == "📊 Evaluation Metrics":

    st.markdown("<div class='title-box'><h1>Evaluation Metrics</h1></div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### ROC Curve")
        st.image("/mnt/c/Deepfake Image Detection/Results/roc_curve.png")

    with col2:
        st.markdown("#### PR Curve")
        st.image("/mnt/c/Deepfake Image Detection/Results/pr_curve.png")

    with col3:
        st.markdown("#### Confusion Matrix")
        st.image("/mnt/c/Deepfake Image Detection/Results/conf_matrix.png")

# --------------------------------------------------------
# Page 3 — Grad-CAM++
# --------------------------------------------------------
elif page == "🔥 Grad-CAM++":

    st.markdown("<div class='title-box'><h1>Grad-CAM++</h1></div>", unsafe_allow_html=True)
    st.info("Will be added soon!")

# --------------------------------------------------------
# Page 4 — About
# --------------------------------------------------------
elif page == "ℹ About":

    st.markdown("<div class='title-box'><h1>About This Project</h1></div>", unsafe_allow_html=True)
    st.write("""
    A deep-learning based **Deepfake Image Detector** built by **Anushka**  
    using ResNet50, CNNs, GPU acceleration, and Streamlit UI.
    """)





