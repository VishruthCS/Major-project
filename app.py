import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import os
import time
import pickle
import threading
import queue
import pyttsx3
import requests
from tensorflow.keras.models import load_model

# --- Import your project modules ---
from config import Config
from utils import FeatureExtractor, draw_styled_landmarks

# --- Page Config ---
st.set_page_config(page_title="Sign Language AI", layout="wide")

# ==========================================
# LOW-LATENCY CAMERA THREAD
# ==========================================
class CameraStream:
    """Runs camera in a background thread to prevent frame buffering lag."""
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.ret, self.frame = self.cap.read()
        self.running = True
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame

    def read(self):
        return self.ret, self.frame

    def release(self):
        self.running = False
        self.thread.join()
        self.cap.release()

# --- Helper Functions ---
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# --- MediaPipe Setup ---
mp_holistic = mp.solutions.holistic

# ==========================================
# BACKGROUND WORKERS (Prevents UI Freezing)
# ==========================================

# ✅ FIX: Store queues in session_state so Streamlit doesn't delete them on re-runs
if 'tts_queue' not in st.session_state:
    st.session_state.tts_queue = queue.Queue()
    st.session_state.gemini_in_q = queue.Queue()
    st.session_state.gemini_out_q = queue.Queue()

tts_queue = st.session_state.tts_queue
gemini_in_q = st.session_state.gemini_in_q
gemini_out_q = st.session_state.gemini_out_q

def tts_loop():
    # Fix for Windows: Background threads must initialize COM before using pyttsx3
    try:
        import pythoncom
        pythoncom.CoInitialize()
    except ImportError:
        pass # If on Linux/Streamlit Cloud, this simply skips and continues normally

    engine = pyttsx3.init()
    while True:
        text = tts_queue.get()
        if text:
            try:
                engine.say(text)
                engine.runAndWait()
            except:
                pass
        tts_queue.task_done()

def get_gemini_sentence(keywords):
    api_key = Config.GEMINI_API_KEY
    if not api_key or "YOUR_GEMINI" in api_key:
        return " ".join(keywords) # Fallback if no key
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    prompt = (
            "You are a helpful assistant for a sign language user. "
            "Convert these keywords into a natural, grammatically correct English sentence dont give multiple sentence give one based on prompt. "
            "Add connecting words if needed and ignore nothing. "
            "Keywords: " + " ".join(keywords)
        )
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    
    try:
        response = requests.post(url, json=payload, timeout=5)
        if response.status_code == 200:
            return response.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
    except:
        pass
    return " ".join(keywords)

def gemini_worker_loop():
    while True:
        keywords = gemini_in_q.get()
        if keywords:
            sentence = get_gemini_sentence(keywords)
            gemini_out_q.put(sentence)
        gemini_in_q.task_done()

# Start background threads only ONCE
if 'threads_started' not in st.session_state:
    threading.Thread(target=tts_loop, daemon=True).start()
    threading.Thread(target=gemini_worker_loop, daemon=True).start()
    st.session_state['threads_started'] = True

# ==========================================
# 1. DATA COLLECTION PAGE (FIXED)
# ==========================================
def data_collection_page():
    st.header("📷 Data Collection")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        gesture_name = st.text_input("Enter Gesture Name", key="gesture_input")
        st.info(f"Target Sequence Length: {Config.SEQUENCE_LENGTH} frames")
        
        # Stats
        if gesture_name:
            path = os.path.join(Config.DATA_PATH, gesture_name)
            if os.path.exists(path):
                count = len([f for f in os.listdir(path) if f.endswith('.npy')])
                st.metric("Existing Samples", count)
            else:
                st.write("New Gesture Folder will be created.")

    with col2:
        run_camera = st.checkbox("Start Camera", key="run_col")
        status_text = st.empty()
        frame_window = st.image([])

    # ✅ Button OUTSIDE the while loop.
    record_btn = st.sidebar.button("Capture Sample", disabled=not gesture_name, key="record_btn")

    if run_camera:
        cap = CameraStream(0)
        extractor = FeatureExtractor()
        recording_buffer = []
        
        # Set recording state based on the button click
        is_recording = record_btn 
        
        if is_recording:
            status_text.warning("🔴 Recording...")

        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while run_camera:
                ret, frame = cap.read()
                if not ret: 
                    st.error("Camera not found")
                    break
                
                # Processing
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(image)
                draw_styled_landmarks(image, results)

                # Recording Logic
                if is_recording:
                    features = extractor.extract_enhanced_features(results)
                    recording_buffer.append(features)
                    
                    # Draw Progress
                    cv2.putText(image, f"REC: {len(recording_buffer)}/{Config.SEQUENCE_LENGTH}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    
                    if len(recording_buffer) == Config.SEQUENCE_LENGTH:
                        is_recording = False # Stop recording
                        
                        # Save Data
                        g_path = os.path.join(Config.DATA_PATH, gesture_name)
                        ensure_dir(g_path)
                        file_id = len([f for f in os.listdir(g_path) if f.endswith('.npy')])
                        np.save(os.path.join(g_path, f"{file_id}.npy"), np.array(recording_buffer))
                        
                        status_text.success(f"✅ Saved sample {file_id}!")
                        
                        # Stop camera gracefully and refresh to reset button state and update counts
                        cap.release()
                        time.sleep(0.5) 
                        st.rerun() 

                # Display
                frame_window.image(image, channels="RGB")

        # Failsafe release if loop breaks
        if cap.running:
            cap.release()

# ==========================================
# 2. TRAINING PAGE
# ==========================================
def training_page():
    st.header("🧠 Model Training")
    if st.button("Start Training"):
        status = st.empty()
        status.info("⏳ Training in progress...")
        try:
            import train_model
            train_model.main()
            status.success("✅ Training Complete! Model saved.")
            st.balloons()
        except Exception as e:
            status.error(f"Error: {e}")

# ==========================================
# 3. RECOGNITION PAGE (OPTIMIZED)
# ==========================================
def recognition_page():
    st.header("🗣️ Gesture Recognition")

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

            st.image(image)

            st.success(f"Prediction: {pred_label}")
            st.write(f"Confidence: {confidence:.2f}")
def main():
    st.sidebar.title("Hand Gesture AI")
    mode = st.sidebar.radio("Go to:", ["Data Collection", "Train Model", "Recognition"])
    
    if mode == "Data Collection": data_collection_page()
    elif mode == "Train Model": training_page()
    elif mode == "Recognition": recognition_page()

if __name__ == "__main__":
    main()