import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import pickle
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

from config import Config
from utils import FeatureExtractor, draw_styled_landmarks
from tensorflow.keras.models import load_model
import os


mp_holistic = mp.solutions.holistic

st.set_page_config(page_title="Sign Language AI", layout="wide")

st.title("🤟 Sign Language Recognition")

# -------------------------
# Load Model
# -------------------------
if not os.path.exists(Config.MODEL_PATH):
    st.error("Model file not found in models folder")
    st.stop()
@st.cache_resource
def load_ai():

    model = load_model(Config.MODEL_PATH, compile=False)        
    with open(Config.LABELS_PATH, "rb") as f:
        label_map = pickle.load(f)

    with open(Config.SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    return model, label_map, scaler

model, label_map, scaler = load_ai()

inv_map = {v: k for k, v in label_map.items()}

extractor = FeatureExtractor()

sequence_buffer = []


# -------------------------
# Video Processor
# -------------------------

class GestureProcessor(VideoTransformerBase):

    def __init__(self):
        self.sequence = []

    def transform(self, frame):

        img = frame.to_ndarray(format="bgr24")

        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as holistic:

            results = holistic.process(image)

            draw_styled_landmarks(img, results)

            features = extractor.extract_enhanced_features(results)

            self.sequence.append(features)

            if len(self.sequence) > Config.SEQUENCE_LENGTH:
                self.sequence.pop(0)

            if len(self.sequence) == Config.SEQUENCE_LENGTH:

                data = np.array(self.sequence).reshape(-1, 774)

                data_scaled = scaler.transform(data)

                data_scaled = data_scaled.reshape(
                    1,
                    Config.SEQUENCE_LENGTH,
                    774
                )

                probs = model.predict(data_scaled)[0]

                best_idx = np.argmax(probs)

                confidence = probs[best_idx]

                label = inv_map[best_idx]

                cv2.putText(
                    img,
                    f"{label} ({confidence:.2f})",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,0),
                    2
                )

        return img


# -------------------------
# Webcam Stream
# -------------------------

webrtc_streamer(
    key="gesture",
    video_transformer_factory=GestureProcessor
)