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
from collections import deque
from tensorflow.keras.models import load_model

# --- Import your project modules ---
from config import Config
from utils import FeatureExtractor, draw_styled_landmarks

# --- Page Config ---
st.set_page_config(page_title="Sign Language AI", layout="wide")

# --- Initialize Session State ---
if 'recording' not in st.session_state:
    st.session_state['recording'] = False
if 'sequence_buffer' not in st.session_state:
    st.session_state['sequence_buffer'] = []
if 'sentence' not in st.session_state:
    st.session_state['sentence'] = []

# --- Helper Functions ---
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# --- MediaPipe Setup ---
mp_holistic = mp.solutions.holistic

# --- TTS Worker (Background Thread) ---
# We use a global queue for TTS so it doesn't block the Streamlit UI
tts_queue = queue.Queue()
def tts_loop():
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

# Start TTS thread once
if 'tts_started' not in st.session_state:
    t = threading.Thread(target=tts_loop, daemon=True)
    t.start()
    st.session_state['tts_started'] = True

# --- Gemini API Call ---
def get_gemini_sentence(keywords):
    if not keywords:
        return ""
    api_key = Config.GEMINI_API_KEY
    if not api_key or "YOUR_GEMINI" in api_key:
        return " ".join(keywords) # Fallback

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    prompt = f"Convert these sign language keywords into a natural English sentence: {' '.join(keywords)}"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    
    try:
        response = requests.post(url, json=payload, timeout=5)
        if response.status_code == 200:
            return response.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
    except:
        pass
    return " ".join(keywords)

# ==========================================
# 1. DATA COLLECTION PAGE
# ==========================================
def data_collection_page():
    st.header("📷 Data Collection")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        gesture_name = st.text_input("Enter Gesture Name")
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

    if run_camera:
        cap = cv2.VideoCapture(0)
        # Fix resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) 
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        extractor = FeatureExtractor()
        recording_buffer = []
        is_recording = False

        # Create buttons inside the loop using a trick or placement outside
        # For Streamlit, we use keyboard shortcuts or sidebar buttons usually, 
        # but here let's use a "Record" checkbox in sidebar for simplicity
        
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
                
                # Logic
                record_btn = st.sidebar.button("Capture Sample", disabled=not gesture_name)
                
                if record_btn:
                    is_recording = True
                    recording_buffer = []
                    status_text.warning("🔴 Recording...")

                if is_recording:
                    features = extractor.extract_enhanced_features(results)
                    recording_buffer.append(features)
                    
                    # Draw Progress
                    cv2.putText(image, f"REC: {len(recording_buffer)}/{Config.SEQUENCE_LENGTH}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    
                    if len(recording_buffer) == Config.SEQUENCE_LENGTH:
                        is_recording = False
                        # Save
                        g_path = os.path.join(Config.DATA_PATH, gesture_name)
                        ensure_dir(g_path)
                        file_id = len([f for f in os.listdir(g_path) if f.endswith('.npy')])
                        np.save(os.path.join(g_path, f"{file_id}.npy"), np.array(recording_buffer))
                        status_text.success(f"✅ Saved sample {file_id}!")
                        time.sleep(0.5) # Short pause
                        st.rerun() # Refresh to update counts

                # Display
                frame_window.image(image, channels="RGB")

        cap.release()

# ==========================================
# 2. TRAINING PAGE
# ==========================================
def training_page():
    st.header("🧠 Model Training")
    st.write("Train the LSTM model on your collected data.")
    
    if st.button("Start Training"):
        status_area = st.empty()
        status_area.info("⏳ Loading data and starting training... check terminal for details.")
        
        # Run the existing training script logic
        try:
            import train_model
            # Redirect stdout to a string buffer if you want to show logs in UI
            # For now, we just run the main function
            train_model.main()
            status_area.success("✅ Training Complete! Model saved.")
            st.balloons()
        except Exception as e:
            status_area.error(f"Error during training: {e}")

# ==========================================
# 3. RECOGNITION PAGE
# ==========================================
# ==========================================
# 3. RECOGNITION PAGE (UPDATED)
# ==========================================
# ==========================================
# 3. RECOGNITION PAGE (OPTIMIZED)
# ==========================================
def recognition_page():
    st.header("🗣️ Real-Time Recognition")
    
    # Layout
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.subheader("Settings")
        # ✅ FIX 1: Add Mirror Option
        mirror_mode = st.checkbox("Mirror Camera (Selfie Mode)", value=True)
        
        st.divider()
        st.subheader("Prediction")
        pred_placeholder = st.empty()
        prob_placeholder = st.empty()
        
        st.divider()
        st.subheader("📝 Raw Keywords")
        keywords_placeholder = st.empty()
        
        st.divider()
        st.subheader("🤖 Gemini AI Output")
        gemini_placeholder = st.empty()

        st.write("") 
        if st.button("Clear All"):
            st.session_state['sentence'] = []
            st.rerun()

    with col1:
        run_rec = st.checkbox("Start Recognition", key="run_rec")
        frame_window = st.image([])

    if run_rec:
        try:
            model = load_model(Config.MODEL_PATH)
            with open(Config.LABELS_PATH, "rb") as f:
                label_map = pickle.load(f)
            with open("models/scaler_lstm.pkl", "rb") as f:
                scaler = pickle.load(f)
        except Exception as e:
            st.error(f"Error: {e}. Train the model first!")
            return

        cap = cv2.VideoCapture(0)
        
        # ✅ FIX 2: LOWER RESOLUTION FOR SPEED
        # Streamlit cannot handle 1280x720 smoothly. 640x480 is much faster.
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        extractor = FeatureExtractor()
        sequence_buffer = []
        sentence_words = []
        last_pred_time = 0
        CONF_THRESHOLD = 0.75 
        
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while run_rec:
                ret, frame = cap.read()
                if not ret: 
                    st.error("Camera disconnected")
                    break
                
                # ✅ FIX 3: MIRRORING logic
                if mirror_mode:
                    frame = cv2.flip(frame, 1)

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(image)
                draw_styled_landmarks(image, results)
                
                # Feature Extraction & Prediction Logic (Same as before)
                features = extractor.extract_enhanced_features(results)
                sequence_buffer.append(features)
                if len(sequence_buffer) > Config.SEQUENCE_LENGTH:
                    sequence_buffer.pop(0)
                
                if len(sequence_buffer) == Config.SEQUENCE_LENGTH:
                    motion = np.mean(np.abs(np.array(sequence_buffer)[-1] - np.array(sequence_buffer)[-5]))
                    
                    if motion > 0.015: 
                        data = np.array(sequence_buffer)
                        data_flat = data.reshape(-1, 774)
                        data_scaled = scaler.transform(data_flat).reshape(1, Config.SEQUENCE_LENGTH, 774)
                        
                        probs = model.predict(data_scaled, verbose=0)[0]
                        best_idx = np.argmax(probs)
                        confidence = probs[best_idx]
                        
                        inv_map = {v: k for k, v in label_map.items()}
                        pred_label = inv_map[best_idx]
                        
                        if confidence > CONF_THRESHOLD:
                            pred_placeholder.markdown(f"### ✅ {pred_label}")
                            prob_placeholder.progress(float(confidence))
                        else:
                            pred_placeholder.markdown(f"❓ Unsure ({pred_label})")
                            prob_placeholder.progress(float(confidence))
                        
                        # Sentence Logic
                        if confidence > CONF_THRESHOLD and (time.time() - last_pred_time > 1.5):
                            if not sentence_words or sentence_words[-1] != pred_label:
                                sentence_words.append(pred_label)
                                last_pred_time = time.time()
                                keywords_placeholder.info(" ".join(sentence_words))
                                
                                if len(sentence_words) >= 2:
                                    gemini_placeholder.text("🤔 AI is thinking...")
                                    final_sent = get_gemini_sentence(sentence_words)
                                    gemini_placeholder.success(final_sent)
                                    tts_queue.put(final_sent) 
                                else:
                                    tts_queue.put(pred_label) 
                
                # Display Frame
                frame_window.image(image, channels="RGB")

        cap.release()

# ==========================================
# MAIN ROUTER
# ==========================================
def main():
    st.sidebar.title("Hand Gesture AI")
    mode = st.sidebar.radio("Go to:", ["Data Collection", "Train Model", "Recognition"])
    
    if mode == "Data Collection":
        data_collection_page()
    elif mode == "Train Model":
        training_page()
    elif mode == "Recognition":
        recognition_page()

if __name__ == "__main__":
    main()