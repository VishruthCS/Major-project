#
# Step 3: Gesture Recognition (FINAL IMPROVED VERSION)
#
# Now includes:
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


def gemini_worker(q_in, q_out):
    """
    Gemini worker (final):
      - cooldown to avoid 429
      - retries on timeout / network fail
      - fallback to flash if pro is slow
      - always sends something back to q_out
    """
    import time
    import requests

    api_key = (getattr(Config, "GEMINI_API_KEY", "") or "").strip()
    if not api_key:
        logging.error("❌ Gemini worker: API key missing in Config.")
    COOLDOWN_SECONDS = 3  # don't hit API too often
    last_call_time = 0

    # models to try, in order
    models_to_try = [
        "gemini-2.5-pro",
        "gemini-2.5-flash",
    ]

    while True:
        keywords = q_in.get()
        if keywords is None:
            break  # shutdown

        if not api_key:
            q_out.put("API Key Not Set")
            q_in.task_done()
            continue

        # if user didn't actually sign anything, don't spam api
        if not keywords:
            q_out.put("")
            q_in.task_done()
            continue

        # Cooldown so we don't get 429
        now = time.time()
        if now - last_call_time < COOLDOWN_SECONDS:
            sleep_for = COOLDOWN_SECONDS - (now - last_call_time)
            time.sleep(sleep_for)
        last_call_time = time.time()

        prompt = (
            "The following are keywords from a signed sentence. "
            "Make a short, grammatically correct English sentence using ONLY these keywords. "
            "Do NOT add extra words. Keywords: " + " ".join(keywords)
        )

        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ]
        }

        sent_ok = False

        for model_name in models_to_try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"

            # try up to 3 times per model
            for attempt in range(3):
                try:
                    # shorter timeout first, slow nets will trigger retry/backoff
                    resp = requests.post(
                        url,
                        headers=headers,
                        json=payload,
                        timeout=8  # ⬅️ shortened timeout
                    )
                    # rate limit / temporary errors
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
                        logging.warning(f"⚠️ Empty Gemini response from {model_name}")
                        q_out.put("...")
                        sent_ok = True
                        break

                except requests.exceptions.ReadTimeout:
                    # this is exactly your error
                    wait = (attempt + 1) * 2
                    logging.error(
                        f"[Gemini Error: {model_name}] Read timed out. Retrying in {wait}s..."
                    )
                    time.sleep(wait)
                    continue
                except Exception as e:
                    wait = (attempt + 1) * 2
                    logging.error(f"[Gemini Error: {model_name}] {e}")
                    # if it's not the last attempt, backoff and retry
                    if attempt < 2:
                        time.sleep(wait)
                        continue
                    else:
                        # give up on this model
                        break

            if sent_ok:
                break  # no need to try fallback

        if not sent_ok:
            # all models failed
            q_out.put("API Error.")
        q_in.task_done()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # --- Load model ---
    model = None
    label_map = None
    scaler = None
    model_loaded = False
    try:
        with open(Config.MODEL_PATH, "rb") as f:
            model_data = pickle.load(f)
        model = model_data.get("model", None)
        label_map = model_data.get("label_map", {}) or {}
        scaler = model_data.get("scaler", None)
        model_loaded = model is not None
        logging.info(f"✅ Existing model & {len(label_map)} labels loaded.")
    except Exception as e:
        logging.error(f"❌ Could not load model: {e}. Run 'python main.py train' first.")

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
    tts_thread = threading.Thread(target=tts_worker, args=(tts_queue, tts_engine), daemon=True)
    tts_thread.start()

    gemini_q_in = queue.Queue()
    gemini_q_out = queue.Queue()
    gemini_thread = threading.Thread(target=gemini_worker, args=(gemini_q_in, gemini_q_out), daemon=True)
    gemini_thread.start()

    # --- Buffers and State ---
    feature_extractor = FeatureExtractor()
    sequence_buffer = deque(maxlen=Config.SEQUENCE_LENGTH)
    prediction_history = deque(maxlen=Config.PREDICTION_SMOOTHING)
    prob_history = deque(maxlen=Config.PREDICTION_SMOOTHING)

    sentence_keywords = []
    last_gesture_time = time.time()
    final_sentence, last_final_sentence_time = "", 0
    stable_prediction = "None"
    failed_frame_count = 0
    waiting_animation = [".", "..", "..."]
    anim_index = 0
    warmup_start = time.time()

    # --- Constants ---
    PAUSE_THRESHOLD = 5.0  # 5s pause = end of sentence
    last_sent_keywords = None  # prevent duplicate Gemini sends

    # --- Main Loop ---
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
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

            # --- Warm-up ---
            if time.time() - warmup_start < 2:
                cv2.putText(image, "Warming up...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.imshow("Gesture Recognition", image)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                continue

            enhanced_features = feature_extractor.extract_enhanced_features(results)
            sequence_buffer.append(enhanced_features)

            current_prediction = "None"

            # --- Prediction block (unchanged) ---
            if model_loaded and len(sequence_buffer) == Config.SEQUENCE_LENGTH:
                movement = np.sum(np.abs(np.diff(np.array(sequence_buffer), axis=0)))
                norm_movement = movement / (Config.SEQUENCE_LENGTH * len(enhanced_features)) if len(enhanced_features) > 0 else 0
                print(f"Movement: {norm_movement:.4f}, Buffer size: {len(sequence_buffer)}")
                if norm_movement >= Config.MOVEMENT_THRESHOLD:
                    flat_features = np.array(sequence_buffer).flatten().reshape(1, -1)
                    if scaler is not None:
                        scaled_features = scaler.transform(flat_features)
                        scaled_features = np.nan_to_num(scaled_features)
                    else:
                        scaled_features = np.nan_to_num(flat_features)

                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba(scaled_features)[0]
                    else:
                        scores = model.decision_function(scaled_features)[0]
                        exp_scores = np.exp(scores - np.max(scores))
                        probs = exp_scores / np.sum(exp_scores)

                    best_idx, best_prob = int(np.argmax(probs)), float(np.max(probs))
                    predicted_class = model.classes_[best_idx]
                    prediction_name = None
                    if predicted_class in label_map:
                        prediction_name = label_map[predicted_class]
                    elif predicted_class in inv_label_map:
                        prediction_name = str(predicted_class)
                    else:
                        try:
                            pc_int = int(predicted_class)
                            if pc_int in label_map:
                                prediction_name = label_map[pc_int]
                            elif pc_int in inv_label_map:
                                prediction_name = inv_label_map[pc_int]
                        except Exception:
                            pass
                    if prediction_name is None:
                        try:
                            inv = {v: k for k, v in label_map.items()}
                            if predicted_class in inv:
                                prediction_name = inv[predicted_class]
                        except Exception:
                            pass
                    if prediction_name is None:
                        prediction_name = str(predicted_class)
                    prediction_history.append(prediction_name)
                    prob_history.append(best_prob)
                    if len(prediction_history) >= Config.PREDICTION_SMOOTHING:
                        most_common_pred = max(set(prediction_history), key=prediction_history.count)
                        avg_prob = np.mean(prob_history) if len(prob_history) > 0 else 0.0
                        print(f"Pred: {most_common_pred}, Avg Prob: {avg_prob:.2f}, Movement: {norm_movement:.4f}")
                        recent_probs = list(prob_history)[-5:] if len(prob_history) > 0 else []
                        recent_mean = np.mean(recent_probs) if len(recent_probs) > 0 else Config.CONF_THRESHOLD
                        dynamic_thresh = max(0.45, recent_mean - 0.05)
                        if avg_prob > dynamic_thresh:
                            current_prediction = most_common_pred
                    print(f"[Predict] {inv_label_map.get(best_idx)} ({best_prob:.2f})")

            # --- Cooldown ---
            if current_prediction != "None":
                if time.time() - last_gesture_time > 0.8:
                    if not sentence_keywords or sentence_keywords[-1] != current_prediction:
                        sentence_keywords.append(current_prediction)
                        last_gesture_time = time.time()

            # --- Improved pause-based sentence sending ---
            if time.time() - last_gesture_time > PAUSE_THRESHOLD and sentence_keywords:
                if last_sent_keywords != sentence_keywords:
                    logging.info(f"🧠 Sending signed sentence to Gemini: {sentence_keywords}")
                    gemini_q_in.put(list(sentence_keywords))
                    last_sent_keywords = list(sentence_keywords)
                    final_sentence = waiting_animation[anim_index % len(waiting_animation)]
                    anim_index += 1
                    last_final_sentence_time = time.time()

            # --- Gemini result handling ---
            try:
                new_sentence = gemini_q_out.get_nowait()
                final_sentence, last_final_sentence_time = new_sentence, time.time()
                tts_queue.put(final_sentence)
                prediction_history.clear()
                prob_history.clear()
                sentence_keywords.clear()  # clear after Gemini response
            except queue.Empty:
                if time.time() - last_final_sentence_time > 10:
                    final_sentence = ""

            # --- UI ---
            cv2.rectangle(image, (0, image.shape[0] - 80), (image.shape[1], image.shape[0]), (0, 0, 0), -1)
            cv2.putText(image, " ".join(sentence_keywords), (10, image.shape[0] - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(image, final_sentence, (10, image.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 100), 2)
            if not model_loaded:
                cv2.putText(image, "No model loaded. Run 'python main.py train'", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                if current_prediction != "None":
                    stable_prediction = current_prediction
                cv2.putText(image, f"Prediction: {stable_prediction}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            if sentence_keywords:
                cv2.putText(image, "Recording sign sequence...", (10, image.shape[0] - 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.imshow("Gesture Recognition", image)
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

    # --- Cleanup ---
    cap.release()
    cv2.destroyAllWindows()
    tts_queue.put(None)
    tts_thread.join()
    gemini_q_in.put(None)
    gemini_thread.join()

if __name__ == "__main__":
    main()
