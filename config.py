#
# Configuration File (FINAL STABLE VERSION)
#
# All settings and parameters for the entire project are stored here.
#

import os
from dotenv import load_dotenv

# Load environment variables from .env file (like your API key)
load_dotenv()

class Config:
    # --- IMPORTANT: API key is loaded from your .env file ---
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")
    
    # --- Project File Paths ---
    # We use a new data path to store our new, enhanced features
    DATA_PATH = "models/gesture_data"
    # --- NEW MODEL PATHS ---
    MODEL_PATH = "models/gesture_lstm_model.keras" # We are using a KNN model
    LABELS_PATH = "models/enhanced_labels.pkl"
    SCALER_PATH = "models/scaler_lstm.pkl"
    feature_dim = 774

    # --- Model & Data Parameters ---
    SEQUENCE_LENGTH = 40
    MIN_SAMPLES_PER_GESTURE = 15 # 15-20 is ideal
    
    # --- Recognition Parameters ---
    CONF_THRESHOLD = 0.75
    MOVEMENT_THRESHOLD = 0.02
    PREDICTION_SMOOTHING = 7
    
    # --- Hybrid AI Parameters ---
    PAUSE_THRESHOLD = 2.5 
    TTS_COOLDOWN = 1.5

