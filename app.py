import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import os
import pickle
import requests
from tensorflow.keras.models import load_model

from config import Config
from utils import FeatureExtractor, draw_styled_landmarks

st.set_page_config(page_title="Sign Language AI", layout="wide")

mp_holistic = mp.solutions.holistic


# ===============================
# Helper
# ===============================

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# ===============================
# Gemini AI
# ===============================

def get_gemini_sentence(keywords):

    api_key = Config.GEMINI_API_KEY

    if not api_key or "YOUR_GEMINI" in api_key:
        return " ".join(keywords)

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"

    prompt = (
        "Convert these keywords into a natural English sentence: "
        + " ".join(keywords)
    )

    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    try:
        r = requests.post(url, json=payload, timeout=5)
        if r.status_code == 200:
            return r.json()["candidates"][0]["content"]["parts"][0]["text"]
    except:
        pass

    return " ".join(keywords)


# ===============================
# DATA COLLECTION PAGE
# ===============================

def data_collection_page():

    st.header("📷 Data Collection")

    gesture_name = st.text_input("Enter Gesture Name")

    st.info(f"Target Sequence Length: {Config.SEQUENCE_LENGTH} frames")

    extractor = FeatureExtractor()

    img_file = st.camera_input("Capture gesture frame")

    if img_file is not None:

        bytes_data = img_file.getvalue()

        np_img = np.frombuffer(bytes_data, np.uint8)

        frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as holistic:

            results = holistic.process(image)

            draw_styled_landmarks(image, results)

            st.image(image)

            if st.button("Save Frame") and gesture_name:

                features = extractor.extract_enhanced_features(results)

                g_path = os.path.join(Config.DATA_PATH, gesture_name)

                ensure_dir(g_path)

                file_id = len(os.listdir(g_path))

                np.save(os.path.join(g_path, f"{file_id}.npy"), features)

                st.success("Sample saved!")


# ===============================
# TRAINING PAGE
# ===============================

def training_page():

    st.header("🧠 Model Training")

    if st.button("Start Training"):

        status = st.empty()

        status.info("Training started...")

        try:

            import train_model

            train_model.main()

            status.success("Training Complete!")

            st.balloons()

        except Exception as e:

            status.error(str(e))


# ===============================
# RECOGNITION PAGE
# ===============================

def recognition_page():

    st.header("🗣 Gesture Recognition")

    try:

        model = load_model(Config.MODEL_PATH)

        with open(Config.LABELS_PATH, "rb") as f:
            label_map = pickle.load(f)

        with open(Config.SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)

    except Exception as e:

        st.error(f"Model loading error: {e}")

        return

    extractor = FeatureExtractor()

    img_file = st.camera_input("Capture gesture")

    if img_file is not None:

        bytes_data = img_file.getvalue()

        np_img = np.frombuffer(bytes_data, np.uint8)

        frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as holistic:

            results = holistic.process(image)

            draw_styled_landmarks(image, results)

            st.image(image)

            features = extractor.extract_enhanced_features(results)

            data = np.array(features).reshape(1, -1)

            data_scaled = scaler.transform(data)

            data_scaled = data_scaled.reshape(
                1,
                Config.SEQUENCE_LENGTH,
                Config.feature_dim
            )

            probs = model.predict(data_scaled)[0]

            best_idx = np.argmax(probs)

            confidence = probs[best_idx]

            inv_map = {v: k for k, v in label_map.items()}

            pred_label = inv_map[best_idx]

            st.success(f"Prediction: {pred_label}")

            st.write(f"Confidence: {confidence:.2f}")


# ===============================
# MAIN APP
# ===============================

def main():

    st.sidebar.title("Hand Gesture AI")

    mode = st.sidebar.radio(
        "Go to:",
        ["Data Collection", "Train Model", "Recognition"]
    )

    if mode == "Data Collection":
        data_collection_page()

    elif mode == "Train Model":
        training_page()

    elif mode == "Recognition":
        recognition_page()


if __name__ == "__main__":
    main()