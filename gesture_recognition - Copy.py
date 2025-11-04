#
# Step 3: Gesture Recognition (FINAL STABLE VERSION)
#
# This script is fast, stable, and avoids the TensorFlow conflict.
# It loads the simple KNN model and runs the full Hybrid AI system.
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
import multiprocessing as mp_process

# --- NO TENSORFLOW IMPORTED ---

# Import from our shared files
from config import Config
from utils import FeatureExtractor, draw_styled_landmarks, mp_holistic

# --- Worker Functions (These run in separate processes/threads) ---
def tts_worker(q, engine):
    while True:
        text = q.get()
        if text is None: break
        if text:
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                logging.error(f"TTS Error: {e}")
        q.task_done()

def gemini_worker(q_in, q_out):
    # --- FIX: Hardcoded API key and URL from your working version ---
    API_KEY = "AIzaSyCW1l8Cft3nAVGsrqykySSaOGTwWdEzC_Y"
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={API_KEY}"
    # --- END FIX ---

    while True:
        keywords = q_in.get()
        if keywords is None: break
        
        # This check is no longer needed, but we check for empty keywords
        if not keywords:
            q_out.put("")
            q_in.task_done()
            continue

        prompt = f"The following are keywords from a signed sentence. Form a complete, grammatically correct English sentence. Do not add any extra information. Keywords: {' '.join(keywords)}"
        headers = {'Content-Type': 'application/json'}
        payload = {"contents": [{"parts": [{"text": prompt}]}]}

        try:
            response = requests.post(api_url, json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()
            completed_sentence = result['candidates'][0]['content']['parts'][0]['text']
            q_out.put(completed_sentence.strip())
        except Exception:
            q_out.put("API Error.")
        q_in.task_done()

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    # --- Load our new, simple model ---
    model = None
    label_map = None
    scaler = None
    model_loaded = False
    try:
        with open(Config.MODEL_PATH, "rb") as f:
            model_data = pickle.load(f)
        model = model_data['model']
        label_map = model_data['label_map']
        scaler = model_data['scaler']
        model_loaded = True
        logging.info(f"✅ Existing model & {len(label_map)} labels loaded.")
    except Exception as e:
        logging.error(f"Could not load model: {e}. Run 'python main.py train' first.")

    # --- Camera can now start without conflict ---
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("❌ Could not open webcam.")
        return
    logging.info("✅ Camera initialized successfully.")
    time.sleep(1.0)
    
    cv2.namedWindow("Gesture Recognition", cv2.WINDOW_NORMAL)

    # Setup workers
    tts_engine = pyttsx3.init()
    tts_queue = queue.Queue()
    # --- FIX: Assign threads to variables ---
    tts_thread = threading.Thread(target=tts_worker, args=(tts_queue, tts_engine), daemon=True)
    tts_thread.start()
    
    gemini_q_in = queue.Queue()
    gemini_q_out = queue.Queue()
    gemini_thread = threading.Thread(target=gemini_worker, args=(gemini_q_in, gemini_q_out), daemon=True)
    gemini_thread.start()
    # --- END FIX ---
    
    sequence_buffer = deque(maxlen=Config.SEQUENCE_LENGTH)
    prediction_history = deque(maxlen=ConfigOpening)
    sentence_keywords = []
    last_gesture_time = time.time()
    final_sentence, last_final_sentence_time = "", 0
    stable_prediction = "None"
    failed_frame_count = 0
    
    feature_extractor = FeatureExtractor()

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                failed_frame_count += 1
                if failed_frame_count > 10: logging.error("❌ Lost camera connection."); break
                continue
            failed_frame_count = 0

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            draw_styled_landmarks(image, results)
            
            enhanced_features = feature_extractor.extract_enhanced_features(results)
            sequence_buffer.append(enhanced_features)

            current_prediction = "None"
            if model_loaded and len(sequence_buffer) == Config.SEQUENCE_LENGTH:
                movement = np.sum(np.abs(np.diff(np.array(sequence_buffer), axis=0)))
                norm_movement = movement / (Config.SEQUENCE_LENGTH * len(enhanced_features)) if len(enhanced_features) > 0 else 0
                
                if norm_movement >= Config.MOVEMENT_THRESHOLD:
                    # Flatten the sequence for our KNN model
                    flat_features = np.array(sequence_buffer).flatten().reshape(1, -1)
                    # Scale the features just like we did in training
                    scaled_features = scaler.transform(flat_features)
                    
                    probs = model.predict_proba(scaled_features)[0]
                    best_idx, best_prob = np.argmax(probs), np.max(probs)
                    
                    if best_prob > Config.CONF_THRESHOLD:
                        # --- FIX: Get gesture NAME, not number ---
                        predicted_index = model.classes_[best_idx]
                        prediction_name = label_map[predicted_index] 
                        prediction_history.append(prediction_name)
                        # --- END FIX ---
                        
                        if len(prediction_history) == Config.PREDICTION_SMOOTHING:
                            # --- FIX: Fixed IndentationError ---
                            current_prediction = max(set(prediction_history), key=list(prediction_history).count)
                            # --- END FIX ---

            if current_prediction != "None":
                last_gesture_time = time.time()
                if not sentence_keywords or sentence_keywords[-1] != current_prediction:
                    sentence_keywords.append(current_prediction)
            
            if time.time() - last_gesture_time > Config.PAUSE_THRESHOLD and sentence_keywords:
                gemini_q_in.put(list(sentence_keywords))
                sentence_keywords.clear()
                last_gesture_time, final_sentence, last_final_sentence_time = time.time(), "...", time.time()
            
            try:
                new_sentence = gemini_q_out.get_nowait()
                final_sentence, last_final_sentence_time = new_sentence, time.time()
                tts_queue.put(final_sentence)
            except queue.Empty:
                if time.time() - last_final_sentence_time > 10: final_sentence = ""

            # UI Display
            cv2.rectangle(image, (0, image.shape[0] - 80), (image.shape[1], image.shape[0]), (0,0,0), -1)
            # This line will now work because sentence_keywords contains strings
            cv2.putText(image, " ".join(sentence_keywords), (10, image.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(image, final_sentence, (10, image.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 100), 2)
            
            if not model_loaded:
                cv2.putText(image, "No model loaded. Please run 'python main.py train'", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                if current_prediction != "None": stable_prediction = current_prediction
                cv2.putText(image, f"Prediction: {stable_prediction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("Gesture Recognition", image)
            if cv2.waitKey(10) & 0xFF == ord('q'): break

    # --- Cleanup ---
    cap.release()
    cv2.destroyAllWindows()
    # --- FIX: Join threads correctly ---
    tts_queue.put(None); tts_thread.join()
    gemini_q_in.put(None); gemini_thread.join()
    # --- END FIX ---

if __name__ == "__main__":
    main()