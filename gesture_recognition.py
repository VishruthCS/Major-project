#
# Step 3: Gesture Recognition (FINAL DEBUG VERSION)
#
# Changes:
# - Added VERBOSE LOGGING to the Gemini Worker to see exactly why API fails.
# - Removed "gemini-2.5-flash" if it's causing 404s (sticking to stable models).
#

import cv2
import numpy as np
import os
import mediapipe as mp
import pickle
import threading
import queue
import pyttsx3
import time
import logging
from collections import deque
import requests

# Import from our shared files
from config import Config
from utils import FeatureExtractor, draw_styled_landmarks, mp_holistic
from tensorflow.keras.models import load_model

# --- TTS Worker ---
def tts_worker(q, engine):
    while True:
        text = q.get()
        if text is None:
            q.task_done()
            break
        if text:
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                logging.error(f"TTS Error: {e}")
        q.task_done()

# --- Gemini Worker (DEBUG MODE) ---
def gemini_worker(q_in, q_out):
    api_key = (getattr(Config, "GEMINI_API_KEY", "") or "").strip()
    
    # Debug print to confirm key is loaded (masking part of it for safety)
    if api_key:
        print(f"[Gemini Debug] API Key loaded: {api_key[:5]}...{api_key[-4:]}")
    else:
        print("[Gemini Debug] ❌ API Key NOT found!")

    COOLDOWN_SECONDS = 2.0
    last_call_time = 0
    
    # Use standard, stable model names first
    models_to_try = ["gemini-2.5-flash", "gemini-pro"]

    while True:
        keywords = q_in.get()
        if keywords is None:
            break
        if not keywords:
            q_out.put("")
            q_in.task_done()
            continue

        # Enforce cooldown
        now = time.time()
        if now - last_call_time < COOLDOWN_SECONDS:
            time.sleep(COOLDOWN_SECONDS - (now - last_call_time))
        last_call_time = time.time()

        prompt = (
            "You are a helpful assistant for a sign language user. "
            "Convert these keywords into a natural, grammatically correct English sentence. "
            "Add connecting words if needed. "
            "Keywords: " + " ".join(keywords)
        )
        
        headers = {"Content-Type": "application/json"}
        payload = {"contents": [{"parts": [{"text": prompt}]}]}

        sent_ok = False
        print(f"\n[Gemini Debug] Sending keywords: {keywords}")

        for model_name in models_to_try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
            print(f"[Gemini Debug] Trying model: {model_name}...")
            
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=8)
                
                print(f"[Gemini Debug] Status Code: {resp.status_code}")
                
                if resp.status_code == 200:
                    data = resp.json()
                    # Extract text safely
                    try:
                        text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                        print(f"[Gemini Debug] ✅ Success! API returned: '{text}'")
                        if text:
                            q_out.put(text)
                            sent_ok = True
                            break
                    except (KeyError, IndexError) as e:
                         print(f"[Gemini Debug] ❌ JSON Parse Error: {e}")
                         print(f"[Gemini Debug] Raw JSON: {data}")

                else:
                    print(f"[Gemini Debug] ❌ API Error: {resp.text}")
            
            except Exception as e:
                print(f"[Gemini Debug] ❌ Network/Request Error: {e}")
                continue
        
        if not sent_ok:
            print("[Gemini Debug] ⚠️ All models failed. Using fallback (raw keywords).")
            # Fallback if AI fails: just join keywords
            q_out.put(" ".join(keywords))
        
        q_in.task_done()

# --- Main Function ---
def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # 1. Load Model, Labels, AND Scaler
    model, label_map, scaler = None, None, None
    model_loaded = False
    try:
        logging.info(f"Loading model from {Config.MODEL_PATH}...")
        model = load_model(Config.MODEL_PATH)
        
        logging.info(f"Loading labels from {Config.LABELS_PATH}...")
        with open(Config.LABELS_PATH, "rb") as f:
            label_map = pickle.load(f)
            
        # CRITICAL: Load the scaler used in training
        scaler_path = "models/scaler_lstm.pkl"
        # Fallback scaler path check
        if not os.path.exists(scaler_path):
            scaler_path = "models/scaler_lstm.pkl" 
            
        logging.info(f"Loading scaler from {scaler_path}...")
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

        model_loaded = True
        logging.info("✅ All AI components loaded successfully.")
        
    except Exception as e:
        logging.error(f"❌ CRITICAL ERROR: Could not load model/scaler. Run 'python main.py train' first. Details: {e}")
        return

    # 2. Camera setup
    cap = cv2.VideoCapture(0)  # or try cv2.CAP_MSMF
    if not cap.isOpened():
        logging.error("❌ Could not open webcam.")
        return
    
    # Optional: Increase camera resolution for better hand detection
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # 3. Threads
    tts_engine = pyttsx3.init()
    tts_queue = queue.Queue()
    threading.Thread(target=tts_worker, args=(tts_queue, tts_engine), daemon=True).start()

    gemini_q_in, gemini_q_out = queue.Queue(), queue.Queue()
    threading.Thread(target=gemini_worker, args=(gemini_q_in, gemini_q_out), daemon=True).start()

    # 4. State Variables
    feature_extractor = FeatureExtractor()
    sequence_buffer = [] # Store raw frames for the current gesture
    
    # UI State
    sentence_keywords = []
    final_sentence = ""
    last_final_sentence_time = 0
    is_recording = False
    prediction_text = "Waiting..."
    prob_text = ""
    
    # Motion Detection Parameters
    START_THRESHOLD = 0.02  # Sensitivity to start recording
    STOP_THRESHOLD = 0.015  # Sensitivity to stop recording
    motion_cooldown = 0     # Prevent rapid triggers

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            # Image processing
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            draw_styled_landmarks(image, results)

            # Extract features
            enhanced_features = feature_extractor.extract_enhanced_features(results)
            
            # --- Logic: Dynamic Gesture Detection (SMARTER TRIGGER) ---
            # Velocity Indices derived from features.py:
            # Full Velocity Block: indices 258 to 516 (258 total features)
            #   - Pose Vel (132 dims): 258 -> 390
            #   - LH Vel   (63 dims):  390 -> 453
            #   - RH Vel   (63 dims):  453 -> 516

            vel_lh = enhanced_features[390 : 453]
            vel_rh = enhanced_features[453 : 516]

            # Calculate intensity for each hand separately
            motion_lh = np.mean(np.abs(vel_lh))
            motion_rh = np.mean(np.abs(vel_rh))
            
            # Trigger if EITHER hand is moving significantly
            motion = max(motion_lh, motion_rh)

            # 1. Start Recording
            if not is_recording and motion > START_THRESHOLD and time.time() > motion_cooldown:
                is_recording = True
                sequence_buffer = []
                prediction_text = "🔴 Recording..."
                prob_text = ""

            # 2. Record Frames
            if is_recording:
                sequence_buffer.append(enhanced_features)
                
                # 3. Stop Recording (if motion drops or buffer too full)
                if (motion < STOP_THRESHOLD and len(sequence_buffer) > 10) or len(sequence_buffer) > 60:
                    is_recording = False
                    motion_cooldown = time.time() + 1.0 # Pause for 1 second after gesture
                    
                    # PROCESS GESTURE
                    if len(sequence_buffer) >= 10: # Ignore tiny twitches
                        # Prepare data (Pad/Trim to 40 frames)
                        seq_arr = np.array(sequence_buffer)
                        if len(seq_arr) < Config.SEQUENCE_LENGTH:
                            pad = np.tile(seq_arr[-1:], (Config.SEQUENCE_LENGTH - len(seq_arr), 1))
                            seq_arr = np.concatenate([seq_arr, pad], axis=0)
                        else:
                            seq_arr = seq_arr[:Config.SEQUENCE_LENGTH]
                        
                        # Scale Data
                        flat = seq_arr.reshape(-1, 774)
                        scaled = scaler.transform(flat)
                        final_input = scaled.reshape(1, Config.SEQUENCE_LENGTH, 774)

                        # Predict
                        probs = model.predict(final_input, verbose=0)[0]
                        best_idx = np.argmax(probs)
                        best_prob = probs[best_idx]
                        
                        # Get Label
                        inv_map = {v: k for k, v in label_map.items()}
                        pred_label = inv_map.get(best_idx, "Unknown")
                        
                        # Threshold Check
                        threshold = getattr(Config, 'CONF_THRESHOLD', 0.65)
                        
                        if best_prob > threshold: # Only accept confident predictions
                            prediction_text = f"✅ {pred_label}"
                            prob_text = f"({best_prob:.0%})"
                            
                            # Add to sentence if different from last word
                            if not sentence_keywords or sentence_keywords[-1] != pred_label:
                                sentence_keywords.append(pred_label)
                        else:
                            prediction_text = "❓ Unsure"
                            prob_text = f"({best_prob:.0%})"
                    else:
                        prediction_text = "❌ Too Short"

            # --- Gemini Logic (Pause Based) ---
            # If user stops gesturing for 2.5 seconds, send sentence to AI
            if not is_recording and sentence_keywords and (time.time() - motion_cooldown > 2.5):
                gemini_q_in.put(list(sentence_keywords))
                sentence_keywords = [] # Clear buffer
                final_sentence = "..."
                last_final_sentence_time = time.time()

            # Check for Gemini Result
            try:
                final_sentence = gemini_q_out.get_nowait()
                tts_queue.put(final_sentence)
                last_final_sentence_time = time.time()
            except queue.Empty:
                pass

            # Clear old sentence from screen after 10s
            if time.time() - last_final_sentence_time > 10:
                final_sentence = ""

            # --- UI Overlay ---
            # Bottom Panel
            cv2.rectangle(image, (0, image.shape[0] - 80), (image.shape[1], image.shape[0]), (0, 0, 0), -1)
            
            # Current Words
            cv2.putText(image, " ".join(sentence_keywords), (20, image.shape[0] - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # AI Sentence
            cv2.putText(image, final_sentence, (20, image.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 100), 2)

            # Top Status
            status_color = (0, 0, 255) if is_recording else (0, 255, 0)
            cv2.putText(image, f"{prediction_text} {prob_text}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
            
            cv2.imshow("Gesture Recognition", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    tts_queue.put(None)
    gemini_q_in.put(None)

if __name__ == "__main__":
    main()