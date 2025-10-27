#
# Hybrid AI Gesture Recognizer - FINAL ALL-IN-ONE VERSION
#
# This is our final, complete, and most robust script. It combines all features
# into one file and uses advanced techniques like lazy loading and multi-threading
# to ensure stability and performance.
#
# This is the script for your deadline, partner. Let's win.
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
import sys
from dotenv import load_dotenv
# --- TensorFlow/Keras will be imported later ("lazy loading") ---

# --- Step 1: Configuration ---
class Config:
    # --- IMPORTANT: ADD YOUR GEMINI API KEY HERE ---
    GEMINI_API_KEY = "AIzaSyCW1l8Cft3nAVGsrqykySSaOGTwWdEzC_Y"
    
    # --- Project file paths ---
    DATA_PATH = "enhanced_gesture_data"
    MODEL_PATH = "gesture_lstm_enhanced.h5"
    LABELS_PATH = "enhanced_labels.pkl"

    # --- Model & Data Parameters ---
    SEQUENCE_LENGTH = 30
    MIN_SAMPLES_PER_GESTURE = 15 # 15-20 is ideal
    TRAIN_EPOCHS = 60
    
    # --- Recognition Parameters ---
    # Reverted to your stable, tested value
    CONF_THRESHOLD = 0.5 
    MOVEMENT_THRESHOLD = 0.02 
    PREDICTION_SMOOTHING = 5
    
    # --- Hybrid AI Parameters ---
    PAUSE_THRESHOLD = 2.5 
    TTS_COOLDOWN = 1.5

# --- Step 2: MediaPipe Setup ---
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyCW1l8Cft3nAVGsrqykySSaOGTwWdEzC_Y")
MODEL_PATH = os.getenv("MODEL_PATH", "gesture_lstm_enhanced.h5")
LABELS_PATH = os.getenv("LABELS_PATH", "enhanced_labels.pkl")
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
# Optional: check that API key is loaded
if not GEMINI_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY not found in .env file!")
# --- Step 3: Utility Functions ---
class FeatureExtractor:
    """
    Manages state for extracting enhanced features (raw, velocity, acceleration).
    This is our "smart textbook" creator.
    """
    def __init__(self):
        self.raw_keypoints_buffer = deque(maxlen=3)

    def extract_raw_keypoints(self, results):
        """Extracts the base keypoints for a single frame."""
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
        return np.concatenate([pose, lh, rh])

    def extract_enhanced_features(self, results):
        """Extracts a rich set of features (raw + velocity + acceleration)."""
        base_features = self.extract_raw_keypoints(results)
        self.raw_keypoints_buffer.append(base_features)

        if len(self.raw_keypoints_buffer) < 3:
            # On the first few frames, return a zero vector of the correct final shape
            return np.zeros(len(base_features) * 3, dtype=np.float32)

        velocity = self.raw_keypoints_buffer[2] - self.raw_keypoints_buffer[1]
        previous_velocity = self.raw_keypoints_buffer[1] - self.raw_keypoints_buffer[0]
        acceleration = velocity - previous_velocity

        return np.concatenate([base_features, velocity, acceleration]).astype(np.float32)

def draw_styled_landmarks(image, results):
    """Draws styled landmarks on the image."""
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(80,22,10), thickness=1, circle_radius=1), 
                                 mp_drawing.DrawingSpec(color=(80,44,121), thickness=1, circle_radius=1)) 
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                                 mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)) 
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

# --- Step 4: Worker Threads (TTS & Gemini) ---
def tts_worker(q, engine):
    """Worker thread for text-to-speech."""
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
    """Worker thread to process keywords with the Gemini API."""
    if not GEMINI_API_KEY:
        print("❌ GEMINI_API_KEY not found in .env")
        return

    while True:
        keywords = q_in.get()
        if keywords is None:
            break

        if not keywords:
            q_out.put("No keywords provided.")
            q_in.task_done()
            continue

        # Create a clear natural language prompt
        prompt = (
            f"The following are keywords from a sign language sentence. "
            f"Form a grammatically correct English sentence without adding new information.\n\n"
            f"Keywords: {' '.join(keywords)}"
        )

        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        try:
            response = requests.post(GEMINI_URL, json=payload, timeout=15)
            response.raise_for_status()
            data = response.json()
            sentence = data["candidates"][0]["content"]["parts"][0]["text"].strip()
            q_out.put(sentence)
        except requests.exceptions.Timeout:
            q_out.put("Gemini API timeout — please try again.")
        except Exception as e:
            q_out.put(f"API Error: {e}")
        finally:
            q_in.task_done()
# --- Step 5: Data Collection Mode ---
def run_data_collection():
    """Main function to run the enhanced data collection UI."""
    os.makedirs(Config.DATA_PATH, exist_ok=True)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
    input_mode, input_text_gesture = False, ""
    is_recording, record_buffer = False, []
    notification_text, last_notification_time = "", 0
    feature_extractor = FeatureExtractor()

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            draw_styled_landmarks(image, results)
            
            enhanced_features = feature_extractor.extract_enhanced_features(results)
            if is_recording:
                record_buffer.append(enhanced_features)

            if time.time() - last_notification_time > 5:
                notification_text = ""

            key = cv2.waitKey(10) & 0xFF
            
            if input_mode:
                cv2.rectangle(image, (100, 180), (image.shape[1] - 100, 280), (0, 0, 0), -1)
                cv2.putText(image, "Enter Gesture Name:", (120, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(image, input_text_gesture, (120, 255), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                if 32 <= key <= 126: input_text_gesture += chr(key)
                elif key == 8: input_text_gesture = input_text_gesture[:-1]
                elif key == 13:
                    if input_text_gesture:
                        os.makedirs(os.path.join(Config.DATA_PATH, input_text_gesture), exist_ok=True)
                        notification_text = f"Press 'v' to REC for '{input_text_gesture}'"
                        last_notification_time = time.time()
                    input_mode = False
            else:
                if key == ord('q'): break
                elif key == ord('c'):
                    input_mode, input_text_gesture = True, ""
                elif key == ord('v'):
                    is_recording = not is_recording
                    if is_recording:
                        record_buffer = []
                        notification_text = "🔴 Recording..."
                    else:
                        notification_text = "⏸ Recording stopped. Press 's' to save."
                    last_notification_time = time.time()
                elif key == ord('s'):
                    if input_text_gesture and record_buffer:
                        if len(record_buffer) >= Config.SEQUENCE_LENGTH:
                            to_save = np.array(record_buffer[-Config.SEQUENCE_LENGTH:])
                        else:
                            notification_text = f"⚠️ Rec too short. Need {Config.SEQUENCE_LENGTH} frames."
                            last_notification_time = time.time()
                            continue
                        
                        gesture_path = os.path.join(Config.DATA_PATH, input_text_gesture)
                        sample_count = len([f for f in os.listdir(gesture_path) if f.endswith('.npy')])
                        np.save(os.path.join(gesture_path, f"{sample_count}.npy"), to_save)
                        
                        notification_text = f"✅ Saved sample {sample_count} for '{input_text_gesture}'"
                        last_notification_time = time.time()
                        record_buffer = []
                    else:
                        notification_text = "⚠️ No recording to save."
                        last_notification_time = time.time()

            if is_recording: cv2.putText(image, "REC", (image.shape[1] - 80, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if not input_mode:
                cv2.putText(image, f"Gesture: '{input_text_gesture}' | Press 'v' to REC/STOP", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if notification_text:
                cv2.putText(image, notification_text, (10, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.imshow('Step 1: Data Collection', image)

    cap.release()
    cv2.destroyAllWindows()

# --- Step 6: Model Training Mode ---
def run_model_training():
    """Loads the collected enhanced data, trains the advanced model, and saves the best version."""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report

    logging.info("--- Starting Enhanced Model Training ---")
    
    if not os.path.exists(Config.DATA_PATH):
        logging.error(f"Data path '{Config.DATA_PATH}' not found. Please run 'collect' first.")
        return
        
    gestures = sorted([d for d in os.listdir(Config.DATA_PATH) if os.path.isdir(os.path.join(Config.DATA_PATH, d))])
    
    if len(gestures) < 2:
        logging.warning("Please collect at least 2 different gestures before training.")
        return

    label_map = {label: idx for idx, label in enumerate(gestures)}
    sequences, labels = [], []

    logging.info("Loading collected enhanced gesture data...")
    for gesture, label in label_map.items():
        gpath = os.path.join(Config.DATA_PATH, gesture)
        sample_files = [f for f in os.listdir(gpath) if f.lower().endswith(".npy")]
        if len(sample_files) < Config.MIN_SAMPLES_PER_GESTURE:
            logging.warning(f"Gesture '{gesture}' needs {Config.MIN_SAMPLES_PER_GESTURE} samples. Aborting.")
            return
        for fname in sample_files:
            sequences.append(np.load(os.path.join(gpath, fname)))
            labels.append(label)

    X = np.array(sequences, dtype=np.float32)
    labels_arr = np.array(labels)
    y = to_categorical(labels_arr).astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=labels_arr)
    
    logging.info(f"📊 Training on {len(X_train)} samples for {len(label_map)} gestures...")
    
    model = Sequential([
        LSTM(64, return_sequences=True, activation='relu', input_shape=(Config.SEQUENCE_LENGTH, X.shape[2])),
        LSTM(128, return_sequences=True, activation='relu'),
        Dropout(0.2),
        LSTM(64, return_sequences=False, activation='relu'),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(len(label_map), activation='softmax')
    ])
    
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.summary()
    
    early_stopping_callback = EarlyStopping(
        monitor='val_categorical_accuracy', patience=10, verbose=1, mode='max', restore_best_weights=True
    )
    
    history = model.fit(
        X_train, y_train, epochs=Config.TRAIN_EPOCHS, validation_data=(X_test, y_test), 
        batch_size=16, verbose=1, callbacks=[early_stopping_callback]
    )

    model.save(Config.MODEL_PATH)
    with open(Config.LABELS_PATH, "wb") as f: pickle.dump(label_map, f)

    logging.info("\n--- TRAINING COMPLETE ---")
    
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    final_accuracy = accuracy_score(y_true, y_pred)
    logging.info(f"Final Best Validation Accuracy: {final_accuracy * 100:.2f}%")
    
    print("\nClassification Report (from best model):")
    target_names = ["" for _ in range(len(label_map))]
    for name, index in label_map.items(): target_names[index] = name
    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))
    
    logging.info(f"Model saved to '{Config.MODEL_PATH}'")

# --- Step 7: Recognition Mode ---
# load_dotenv()

# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# MODEL_PATH = os.getenv("MODEL_PATH", "gesture_model.h5")
# LABELS_PATH = os.getenv("LABELS_PATH", "label_map.pkl")

# Optional: check that API key is loaded
if not GEMINI_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY not found in .env file!")

def run_gesture_recognition():
    """Main function for the real-time recognition UI."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    # --- Try multiple camera sources if 0 fails ---
    cap = None
    for i in range(3):
        test_cap = cv2.VideoCapture(i)
        if test_cap.isOpened():
            cap = test_cap
            logging.info(f"✅ Camera initialized successfully at index {i}.")
            break
        test_cap.release()

    if cap is None:
        logging.error("❌ Could not open any webcam. Try connecting your camera or using DroidCam.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768) 
    time.sleep(1.0)

    model, label_map, model_loaded = None, None, False
    sequence_buffer = deque(maxlen=Config.SEQUENCE_LENGTH)
    prediction_history = deque(maxlen=Config.PREDICTION_SMOOTHING)
    sentence_keywords, last_gesture_time = [], time.time()
    final_sentence, last_final_sentence_time = "", 0
    stable_prediction, failed_frame_count = "None", 0

    feature_extractor = FeatureExtractor()

    # --- LAZY LOADING: Start TTS & Gemini workers ---
    tts_engine = pyttsx3.init()
    tts_queue = queue.Queue()
    threading.Thread(target=tts_worker, args=(tts_queue, tts_engine), daemon=True).start()

    gemini_q_in, gemini_q_out = queue.Queue(), queue.Queue()
    threading.Thread(target=gemini_worker, args=(gemini_q_in, gemini_q_out), daemon=True).start()

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                failed_frame_count += 1
                logging.warning(f"[Frame Warning] No frame received ({failed_frame_count} times).")
                if failed_frame_count > 20:
                    logging.error("❌ Lost camera connection permanently.")
                    break
                time.sleep(0.2)
                continue
            failed_frame_count = 0

            # --- Load model once lazily ---
            if not model_loaded:
                try:
                    from tensorflow.keras.models import load_model
                    logging.info("Loading TensorFlow model...")
                    model = load_model(Config.MODEL_PATH)
                    with open(Config.LABELS_PATH, "rb") as f:
                        label_map = pickle.load(f)
                    logging.info(f"✅ Model loaded with {len(label_map)} gestures.")
                    model_loaded = True
                except Exception as e:
                    logging.error(f"Could not load model: {e}")
                    break

            # --- Process frame ---
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image_rgb)
            draw_styled_landmarks(frame, results)

            enhanced_features = feature_extractor.extract_enhanced_features(results)
            sequence_buffer.append(enhanced_features)

            current_prediction = "None"
            if model_loaded and len(sequence_buffer) == Config.SEQUENCE_LENGTH:
                movement = np.sum(np.abs(np.diff(np.array(sequence_buffer), axis=0)))
                norm_movement = movement / (Config.SEQUENCE_LENGTH * len(enhanced_features))
                if norm_movement >= Config.MOVEMENT_THRESHOLD:
                    inp = np.expand_dims(sequence_buffer, axis=0)
                    probs = model.predict(inp, verbose=0)[0]
                    best_idx, best_prob = np.argmax(probs), np.max(probs)
                    if best_prob > Config.CONF_THRESHOLD:
                        inv_map = {v: k for k, v in label_map.items()}
                        prediction_history.append(inv_map.get(best_idx, "Unknown"))
                        if len(prediction_history) == Config.PREDICTION_SMOOTHING:
                            current_prediction = max(set(prediction_history), key=prediction_history.count)

            if current_prediction != "None":
                last_gesture_time = time.time()
                if not sentence_keywords or sentence_keywords[-1] != current_prediction:
                    sentence_keywords.append(current_prediction)

            if time.time() - last_gesture_time > Config.PAUSE_THRESHOLD and sentence_keywords:
                gemini_q_in.put(list(sentence_keywords))
                sentence_keywords.clear()
                final_sentence, last_final_sentence_time = "...", time.time()

            try:
                new_sentence = gemini_q_out.get_nowait()
                final_sentence, last_final_sentence_time = new_sentence, time.time()
                tts_queue.put(final_sentence)
            except queue.Empty:
                if time.time() - last_final_sentence_time > 10:
                    final_sentence = ""

            # --- UI Overlay ---
            cv2.rectangle(frame, (0, frame.shape[0] - 80), (frame.shape[1], frame.shape[0]), (0,0,0), -1)
            cv2.putText(frame, " ".join(sentence_keywords), (10, frame.shape[0] - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, final_sentence, (10, frame.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 100), 2)
            stable_prediction = current_prediction if current_prediction != "None" else stable_prediction
            cv2.putText(frame, f"Prediction: {stable_prediction}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("Step 3: Real-Time Recognition", frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    tts_queue.put(None)
    gemini_q_in.put(None)


    
# --- Step 8: Main Control Panel ---
def main_control():
    """Parses command-line arguments to run the correct module."""
    if len(sys.argv) < 2:
        print("\n" + "="*50)
        print("Hybrid AI Gesture Recognizer - Control Panel")
        print("="*50)
        print("Usage: python final_project.py [collect | train | recognize]")
        print("\n  collect   - Step 1: Start the enhanced data collection script.")
        print("  train     - Step 2: Train the model on the collected enhanced data.")
        print("  recognize - Step 3: Run the final real-time gesture recognizer.")
        print("="*50)
        return

    command = sys.argv[1].lower()

    if command == "collect":
        print("\n--- Starting Step 1: Enhanced Data Collection ---")
        run_data_collection()
    elif command == "train":
        print("\n--- Starting Step 2: Enhanced Model Training ---")
        run_model_training()
    elif command == "recognize":
        print("\n--- Starting Step 3: Final Hybrid AI Recognizer ---")
        run_gesture_recognition()
    else:
        print(f"Unknown command: '{command}'")
        print("Please use 'collect', 'train', or 'recognize'.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    main_control()

