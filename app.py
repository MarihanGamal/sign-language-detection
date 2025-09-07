import streamlit as st
import cv2
import numpy as np
from PIL import Image
import json
import tensorflow as tf
import time

# ================== Load Model & Labels ==================
model = tf.keras.models.load_model("sign_mnist_finetuned.h5", compile=False)
with open("label_map.json", "r") as f:
    label_map = json.load(f)
label_map = {int(k): v for k, v in label_map.items()}

# ================== Streamlit Config ==================
st.set_page_config(page_title="Sign Language Detection App", layout="wide")

# ================== THEME (Bubble Font) ==================
st.markdown(
    """
    <style>
        /* استيراد خط Bubblegum Sans */
        @import url('https://fonts.googleapis.com/css2?family=Bubblegum+Sans&display=swap');

        :root {
            --pink-1: #ff1a8c;
            --pink-2: #ff3385;
            --pink-3: #ff66a3;
            --pink-4: #ff99c2;
        }

        .stApp {
            background: #000;
            color: var(--pink-4);
            font-family: 'Bubblegum Sans', cursive;
        }

        /* العنوان بالخط البابل */
        .app-title {
            text-align:center;
            font-size:75px;
            font-family:'Bubblegum Sans', cursive;
            font-weight:400;
            text-transform:none;
            letter-spacing: 1px;
            color: var(--pink-3);
            margin: 20px 0 30px 0;
        }

        /* الكروت */
        .card {
            background: rgba(255, 51, 133, 0.06);
            border: 1px solid rgba(255, 102, 163, .3);
            border-radius: 24px;
            box-shadow: 0 6px 22px rgba(255, 51, 133, .25);
            padding: 18px;
        }
        .result-card {
            background: linear-gradient(135deg, rgba(255,51,133,.2), rgba(255,153,194,.12));
            border: 1px solid rgba(255, 153, 194, .35);
            border-radius: 26px;
            padding: 14px 20px;
            text-align:center;
        }

        /* الأزرار */
        .stButton > button {
            background: linear-gradient(135deg, var(--pink-2), var(--pink-1));
            color: #fff;
            border-radius: 16px;
            font-weight: 600;
            font-family:'Bubblegum Sans', cursive;
            padding: 12px 22px;
            transition: transform .1s ease, box-shadow .2s ease;
            border: 1px solid rgba(255,255,255,.18);
            font-size: 18px;
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 24px rgba(255, 51, 133, .35);
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ================== Title ==================
st.markdown("<div class='app-title'> 🤟 Sign Language Detection </div>", unsafe_allow_html=True)

# ================== Helper Functions ==================
def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (28, 28))
    img = img / 255.0
    img = img.reshape(1, 28, 28, 1)
    return img

def predict_letter(img):
    processed = preprocess_image(img)
    preds = model.predict(processed)
    class_id = np.argmax(preds, axis=1)[0]
    return label_map[class_id]

def result_card(letter: str) -> str:
    return f"""
    <div class="result-card">
        <div style="
            font-size: 110px;
            line-height: 1;
            font-weight: 900;
            color: #ff3385;
            font-family: 'Montserrat', sans-serif;">
            {letter}
        </div>
        <div style="font-size: 20px; letter-spacing: 3px; color:#ff99c2; margin-top:6px; font-family:'Montserrat', sans-serif;">
            PREDICTION
        </div>
    </div>
    """

# ================== Mode Selection ==================
option = st.radio("Choose Input Method:", ["📹 Live Camera", "📸 Camera Snapshot", "🖼 Upload Image"], horizontal=True)

# ================== LIVE CAMERA ==================
if option == "📹 Live Camera":
    col1, col2 = st.columns([2,1])
    with col2:
        duration = st.slider("⏱ Capture duration (seconds)", 3, 20, 8)
        fps_hint = st.select_slider("Frame rate hint", options=[10, 15, 20, 25, 30], value=15)

    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        frame_placeholder = st.empty()
        st.markdown("</div>", unsafe_allow_html=True)

    res_placeholder = st.empty()

    start = st.button("📸 Start Live", type="primary")
    if start:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("⚠ Failed to open camera.")
        else:
            start_time = time.time()
            frame_interval = 1.0 / float(fps_hint)
            last_frame_t = 0.0

            while True:
                ok, frame = cap.read()
                now = time.time()
                if not ok:
                    st.error("⚠ Failed to grab frame.")
                    break

                if (now - last_frame_t) < frame_interval:
                    continue
                last_frame_t = now

                pred = predict_letter(frame)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(rgb, channels="RGB", use_column_width=True, caption="Live Camera")
                res_placeholder.markdown(result_card(pred), unsafe_allow_html=True)

                if (now - start_time) > duration:
                    break

            cap.release()
            st.success("⛔ Live session ended.")

# ================== BROWSER CAMERA SNAPSHOT ==================
elif option == "📸 Camera Snapshot":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    img_file = st.camera_input("Take a picture")
    st.markdown("</div>", unsafe_allow_html=True)

    if img_file is not None:
        image = Image.open(img_file)
        img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        st.image(image, caption="Captured Image", use_column_width=True)
        pred = predict_letter(img_bgr)
        st.markdown(result_card(pred), unsafe_allow_html=True)

# ================== UPLOAD IMAGE ==================
elif option == "🖼 Upload Image":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        pred = predict_letter(img_bgr)
        st.markdown(result_card(pred), unsafe_allow_html=True)
