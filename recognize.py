from config import Config
from features import FeatureExtractor, draw_styled_landmarks
from workers import tts_worker, gemini_worker
import cv2, numpy as np, time, queue, threading, pickle, logging
from collections import deque 
import pyttsx3
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
def run_gesture_recognition():
    """Main function for the real-time recognition UI."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    # --- LAZY LOADING: Camera first ---
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("❌ Could not open webcam.")
        return
    logging.info("✅ Camera initialized successfully.")
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
        while cap.isOpened():
            ret, frame = cap.read()
            time.sleep(0.03) 
            if not ret:
                failed_frame_count += 1
                if failed_frame_count > 10: logging.error("❌ Lost camera connection."); break
                continue
            failed_frame_count = 0

            if not model_loaded:
                try:
                    from tensorflow.keras.models import load_model
                    logging.info("Loading TensorFlow model...")
                    model = load_model(Config.MODEL_PATH)
                    with open(Config.LABELS_PATH, "rb") as f:
                        label_map = pickle.load(f)
                    logging.info(f"✅ Model & {len(label_map)} labels loaded.")
                    model_loaded = True
                except Exception as e:
                    logging.error(f"Could not load model: {e}. Run 'python final_project.py train' first.")
                    break

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
                    inp = np.expand_dims(sequence_buffer, axis=0)
                    probs = model.predict(inp, verbose=0)[0]
                    best_idx, best_prob = np.argmax(probs), np.max(probs)
                    if best_prob > Config.CONF_THRESHOLD:
                        inv_label_map = {v: k for k, v in label_map.items()}
                        prediction_history.append(inv_label_map.get(best_idx, "Unknown"))
                        if len(prediction_history) == Config.PREDICTION_SMOOTHING:
                            current_prediction = max(set(prediction_history), key=list(prediction_history).count)
            if model_loaded and len(sequence_buffer) == Config.SEQUENCE_LENGTH:
                movement = np.sum(np.abs(np.diff(np.array(sequence_buffer), axis=0)))
                norm_movement = movement / (Config.SEQUENCE_LENGTH * len(enhanced_features)) if len(enhanced_features) > 0 else 0

                if norm_movement >= Config.MOVEMENT_THRESHOLD:
                    inp = np.expand_dims(sequence_buffer, axis=0)
                    probs = model.predict(inp, verbose=0)[0]
                    best_idx, best_prob = np.argmax(probs), np.max(probs)

                    inv_label_map = {v: k for k, v in label_map.items()}
                    predicted_label = inv_label_map.get(best_idx, "Unknown")

        # ✅ Confidence threshold (ignore weak predictions)
                    if best_prob < 0.75:
                        predicted_label = "Unknown"

        # ✅ Smooth predictions
                    prediction_history.append(predicted_label)
                    if len(prediction_history) > Config.PREDICTION_SMOOTHING:
                        prediction_history.popleft()

                    current_prediction = max(set(prediction_history), key=list(prediction_history).count)

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
            cv2.putText(image, " ".join(sentence_keywords), (10, image.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(image, final_sentence, (10, image.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 100), 2)
            if current_prediction != "None": stable_prediction = current_prediction
            cv2.putText(image, f"Prediction: {stable_prediction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            if cv2.waitKey(10) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
    # Cleanup threads
    tts_queue.put(None)
    gemini_q_in.put(None)
