#
# Step 3: Gesture Recognition (FINAL IMPROVED VERSION)
#
# Now includes:
# ✅ Predict after gesture completion (no mid-gesture guesses)
# ✅ Probability smoothing
# ✅ Gesture cooldown
# ✅ Movement threshold tuning
# ✅ Thread-safe Gemini + TTS
# ✅ UI feedback polish
# ✅ Robust label mapping and safe prob handling
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


# --- Gemini Worker (with retry, cooldown, fallback) ---
def gemini_worker(q_in, q_out):
    import time
    import requests

    api_key = (getattr(Config, "GEMINI_API_KEY", "") or "").strip()
    if not api_key:
        logging.error("❌ Gemini worker: API key missing in Config.")
    COOLDOWN_SECONDS = 3
    last_call_time = 0

    models_to_try = ["gemini-2.5-pro", "gemini-2.5-flash"]

    while True:
        keywords = q_in.get()
        if keywords is None:
            break
        if not api_key:
            q_out.put("API Key Not Set")
            q_in.task_done()
            continue
        if not keywords:
            q_out.put("")
            q_in.task_done()
            continue

        now = time.time()
        if now - last_call_time < COOLDOWN_SECONDS:
            time.sleep(COOLDOWN_SECONDS - (now - last_call_time))
        last_call_time = time.time()

        prompt = (
            "The following are keywords from a signed sentence. "
            "Make a short, grammatically correct English sentence using ONLY these keywords. "
            "Do NOT add extra words. Keywords: " + " ".join(keywords)
        )
        headers = {"Content-Type": "application/json"}
        payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}

        sent_ok = False
        for model_name in models_to_try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
            for attempt in range(3):
                try:
                    resp = requests.post(url, headers=headers, json=payload, timeout=8)
                    if resp.status_code == 429:
                        wait = (attempt + 1) * 2
                        logging.warning(f"⚠️ Rate limited on {model_name}, retrying in {wait}s...")
                        time.sleep(wait)
                        continue
                    resp.raise_for_status()
                    data = resp.json()
                    if "candidates" in data and data["candidates"]:
                        text = (
                            data["candidates"][0]
                            .get("content", {})
                            .get("parts", [{}])[0]
                            .get("text", "")
                            .strip()
                        )
                        if not text:
                            text = "..."
                        q_out.put(text)
                        sent_ok = True
                        break
                    else:
                        q_out.put("...")
                        sent_ok = True
                        break
                except requests.exceptions.ReadTimeout:
                    wait = (attempt + 1) * 2
                    logging.error(f"[Gemini Error: {model_name}] Timeout. Retrying in {wait}s...")
                    time.sleep(wait)
                    continue
                except Exception as e:
                    wait = (attempt + 1) * 2
                    logging.error(f"[Gemini Error: {model_name}] {e}")
                    if attempt < 2:
                        time.sleep(wait)
                        continue
                    else:
                        break
            if sent_ok:
                break
        if not sent_ok:
            q_out.put("API Error.")
        q_in.task_done()


# --- Main Function ---
def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # --- Load model ---
    model, label_map, scaler = None, None, None
    model_loaded = False
    try:
        with open(Config.MODEL_PATH, "rb") as f:
            model_data = pickle.load(f)
        model = model_data.get("model")
        label_map = model_data.get("label_map", {}) or {}
        scaler = model_data.get("scaler")
        model_loaded = model is not None
        logging.info(f"✅ Model & {len(label_map)} labels loaded.")
        logging.info(f"Loaded label map: {label_map}")
    except Exception as e:
        logging.error(f"❌ Could not load model: {e}. Run 'python main.py train' first.")

    # Build inverse label map safely
    inv_label_map = {}
    try:
        inv_label_map = {v: k for k, v in label_map.items()}
    except Exception:
        inv_label_map = {}

    # --- Camera setup ---
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("❌ Could not open webcam.")
        return
    logging.info("✅ Camera initialized successfully.")
    time.sleep(1.0)
    cv2.namedWindow("Gesture Recognition", cv2.WINDOW_NORMAL)

    # --- Threads ---
    tts_engine = pyttsx3.init()
    tts_queue = queue.Queue()
    threading.Thread(target=tts_worker, args=(tts_queue, tts_engine), daemon=True).start()

    gemini_q_in, gemini_q_out = queue.Queue(), queue.Queue()
    threading.Thread(target=gemini_worker, args=(gemini_q_in, gemini_q_out), daemon=True).start()

    # --- Buffers & State ---
    feature_extractor = FeatureExtractor()
    sequence_buffer = deque(maxlen=Config.SEQUENCE_LENGTH)
    sentence_keywords, prediction_history, prob_history = [], deque(maxlen=Config.PREDICTION_SMOOTHING), deque(maxlen=Config.PREDICTION_SMOOTHING)
    last_gesture_time, final_sentence, last_final_sentence_time = time.time(), "", 0
    stable_prediction, failed_frame_count, warmup_start = "None", 0, time.time()
    waiting_animation, anim_index = [".", "..", "..."], 0
    PAUSE_THRESHOLD = 5.0
    last_sent_keywords = None

    # --- Gesture motion state ---
    gesture_active = False
    gesture_frames = []

    # --- Thresholds ---
    start_threshold = 0.02   # Motion must exceed this to start gesture
    stop_threshold  = 0.03   # Motion <= this considered still (end)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        norm_movement = 0.0  # initialize to avoid UnboundLocalError

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                failed_frame_count += 1
                if failed_frame_count > 10:
                    logging.error("❌ Lost camera connection.")
                    break
                continue
            failed_frame_count = 0

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            draw_styled_landmarks(image, results)

            # --- Warm-up (stabilize mediapipe) ---
            if time.time() - warmup_start < 2:
                cv2.putText(image, "Warming up...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.imshow("Gesture Recognition", image)
                if cv2.waitKey(10) & 0xFF == ord("q"):
                    break
                continue

            enhanced_features = feature_extractor.extract_enhanced_features(results)
            sequence_buffer.append(enhanced_features)

            # --- Calculate movement ---
            if model_loaded and len(sequence_buffer) == Config.SEQUENCE_LENGTH:
                movement = np.sum(np.abs(np.diff(np.array(sequence_buffer), axis=0)))
                norm_movement = movement / (Config.SEQUENCE_LENGTH * len(enhanced_features)) if len(enhanced_features) > 0 else 0
                print(f"Movement: {norm_movement:.4f}, Buffer size: {len(sequence_buffer)}")

                # --- Motion classification ---
                is_moving = norm_movement > start_threshold
                is_still  = norm_movement <= stop_threshold

                # --- Gesture Start ---
                if is_moving and not gesture_active:
                    gesture_active = True
                    gesture_frames = []
                    print("🟢 Gesture started")

                # --- Gesture Recording ---
                if gesture_active:
                    gesture_frames.append(enhanced_features)

                # --- Gesture End ---
                
                # --- Gesture End (trigger prediction once motion stops) ---
                if gesture_active and is_still and len(gesture_frames) > 10:
                    print(f"🔵 Gesture ended, total frames: {len(gesture_frames)}")

    # Convert to numpy array
                    gesture_seq = np.array(gesture_frames)

    # Pad or crop to match training sequence length
                    if len(gesture_seq) < Config.SEQUENCE_LENGTH:
                        pad_len = Config.SEQUENCE_LENGTH - len(gesture_seq)
                        gesture_seq = np.concatenate(
                        [gesture_seq, np.zeros((pad_len, gesture_seq.shape[1]))],
                        axis=0
            )
                    elif len(gesture_seq) > Config.SEQUENCE_LENGTH:
                        gesture_seq = gesture_seq[-Config.SEQUENCE_LENGTH:]  # take last N frames

                    flat_features = gesture_seq.flatten().reshape(1, -1)

    # Scale features (same as training)
                    if scaler is not None:
                        scaled_features = scaler.transform(flat_features)
                        scaled_features = np.nan_to_num(scaled_features)
                    else:
                        scaled_features = np.nan_to_num(flat_features)

    # Predict
                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba(scaled_features)[0]
                    else:
                        scores = model.decision_function(scaled_features)[0]
                        exp_scores = np.exp(scores - np.max(scores))
                        probs = exp_scores / np.sum(exp_scores)

                    best_idx, best_prob = int(np.argmax(probs)), float(np.max(probs))

    # --- Robust label mapping ---
                    prediction_name = None
                    if best_idx in label_map:
                        prediction_name = label_map[best_idx]
                    elif isinstance(label_map, dict):
                        inv = {v: k for k, v in label_map.items()}
                        prediction_name = inv.get(best_idx, None)
                    elif hasattr(model, "classes_"):
                        prediction_name = str(model.classes_[best_idx])
                    if prediction_name is None:
                        prediction_name = f"Class_{best_idx}"

                    prediction_name = str(prediction_name).strip().capitalize()
                    print(f"✅ Final Gesture Recognized: {prediction_name} ({best_prob:.2f})")

    # Update UI and sentence
                    if best_prob > Config.CONF_THRESHOLD:
                        if not sentence_keywords or sentence_keywords[-1] != prediction_name:
                            sentence_keywords.append(prediction_name)
                            last_gesture_time = time.time()

                    gesture_active = False
                    gesture_frames = []
            # --- Send to Gemini (pause-based) ---
            if time.time() - last_gesture_time > PAUSE_THRESHOLD and sentence_keywords:
                if last_sent_keywords != sentence_keywords:
                    logging.info(f"🧠 Sending signed sentence to Gemini: {sentence_keywords}")
                    gemini_q_in.put(list(sentence_keywords))
                    last_sent_keywords = list(sentence_keywords)
                    final_sentence = waiting_animation[anim_index % len(waiting_animation)]
                    anim_index += 1
                    last_final_sentence_time = time.time()

            # --- Handle Gemini response ---
            try:
                new_sentence = gemini_q_out.get_nowait()
                final_sentence, last_final_sentence_time = new_sentence, time.time()
                tts_queue.put(final_sentence)
                sentence_keywords.clear()
            except queue.Empty:
                if time.time() - last_final_sentence_time > 10:
                    final_sentence = ""

            # --- UI ---
            cv2.rectangle(image, (0, image.shape[0] - 80), (image.shape[1], image.shape[0]), (0, 0, 0), -1)
            cv2.putText(image, " ".join(map(str, sentence_keywords)), (10, image.shape[0] - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(image, final_sentence, (10, image.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 100), 2)

            # Movement status color feedback
            color = (0, 165, 255) if gesture_active else (0, 255, 0)
            cv2.putText(image, f"Movement: {norm_movement:.3f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.imshow("Gesture Recognition", image)
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

    # --- Cleanup ---
    cap.release()
    cv2.destroyAllWindows()
    tts_queue.put(None)
    gemini_q_in.put(None)
