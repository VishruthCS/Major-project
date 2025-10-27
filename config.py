import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyCW1l8Cft3nAVGsrqykySSaOGTwWdEzC_Y")
    DATA_PATH = "enhanced_gesture_data"
    MODEL_PATH = "gesture_lstm_enhanced.h5"
    LABELS_PATH = "enhanced_labels.pkl"

    SEQUENCE_LENGTH = 30
    MIN_SAMPLES_PER_GESTURE = 15
    TRAIN_EPOCHS = 60

    CONF_THRESHOLD = 0.6
    MOVEMENT_THRESHOLD = 0.03
    PREDICTION_SMOOTHING = 7
    PAUSE_THRESHOLD = 2.5
    TTS_COOLDOWN = 1.5

    GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
