#
# data_collection_lstm_final_demo.py
#
# ✅ Bright, demo-ready interface (looks like your KNN demo)
# ✅ Uses Enhanced FeatureExtractor (feature_dim features = base + velocity + acceleration)
# ✅ Press 'v' to start/stop recording
# ✅ If >30 frames, saves in chunks (pads final chunk)
# ✅ Clean notifications + gesture info + auto numbering
#

import os
import time
import cv2
import numpy as np
from config import Config
from features import FeatureExtractor, draw_styled_landmarks
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def ensure_gesture_dir(name):
    path = os.path.join(Config.DATA_PATH, name)
    os.makedirs(path, exist_ok=True)
    return path

def next_index(path):
    files = [f for f in os.listdir(path) if f.endswith(".npy")]
    return len(files)

def save_chunks(sequence, gesture_path, gesture_name="unknown", backup_path=None):
    """
    Splits and saves recordings into 30-frame sequences (pads/trims as needed).
    Also creates a backup copy if backup_path is provided.
    """
    seq_len = Config.SEQUENCE_LENGTH
    saved = 0
    n = len(sequence)

    for i in range(0, n, seq_len):
        chunk = sequence[i:i + seq_len]

        # --- Pad or trim to exact sequence length ---
        if len(chunk) < seq_len:
            pad = np.tile(chunk[-1:], (seq_len - len(chunk), 1))
            chunk = np.concatenate([chunk, pad], axis=0)
        elif len(chunk) > seq_len:
            chunk = chunk[:seq_len]

        # --- Define main and backup save paths ---
        idx = next_index(gesture_path)
        save_path = os.path.join(gesture_path, f"{idx}.npy")

        np.save(save_path, chunk)  # save main file

        if backup_path:
            os.makedirs(backup_path, exist_ok=True)
            backup_file = os.path.join(backup_path, f"{gesture_name}_{idx}.npy")
            np.save(backup_file, chunk)

        saved += 1

    return saved


def run_data_collection():
    os.makedirs(Config.DATA_PATH, exist_ok=True)
    backup_path = os.path.join(Config.DATA_PATH, "_backup")
    os.makedirs(backup_path, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)

    feature_extractor = FeatureExtractor()
    gesture_name = None
    is_recording = False
    buffer = []
    notification = ""
    note_time = 0

    print("\n🎮 CONTROLS:")
    print("  C → Set gesture name")
    print("  V → Start/Stop recording")
    print("  S → Save recording")
    print("  Q → Quit\n")

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                draw_styled_landmarks(image, results)
            except Exception:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            features = feature_extractor.extract_enhanced_features(results)
            # Auto-detect feature dimension on first valid frame
            if 'feature_dim' not in locals() and features is not None:
                feature_dim = len(features)

            if is_recording:
                buffer.append(features)
                cv2.putText(image, f"Recording: {len(buffer)} frames", (10, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # --- Overlays ---
            h, w = image.shape[:2]
            cv2.rectangle(image, (0, 0), (w, 40), (0, 0, 0), -1)
            cv2.putText(image, "Data Collection (Enhanced LSTM)", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            if gesture_name:
                cv2.putText(image, f"Gesture: {gesture_name}", (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            if is_recording:
                cv2.putText(image, "🔴 REC", (w - 120, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            if notification and time.time() - note_time < 4:
                cv2.putText(image, notification, (10, h - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.putText(image, "C=Set Gesture | V=Start/Stop | S=Save | Q=Quit",
                        (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow("Enhanced Data Collection", image)
            key = cv2.waitKey(10) & 0xFF

            # --- Keyboard controls ---
            if key == ord('q'):
                break
            elif key == ord('c'):
                gesture_name = input("Enter gesture name: ").strip()
                if gesture_name:
                    ensure_gesture_dir(gesture_name)
                    notification = f"Ready for '{gesture_name}'. Press V to record."
                    note_time = time.time()
                    buffer = []
            elif key == ord('v') and gesture_name:
                if time.time() - note_time < 0.3:
                    continue  # prevent double-tap trigger

                is_recording = not is_recording
                if is_recording:
                    buffer = []
                    notification = "🔴 Recording started..."
                else:
                    notification = f"⏸ Recording stopped at {len(buffer)} frames."
                note_time = time.time()
            elif key == ord('s') and gesture_name:
                if len(buffer) == 0:
                    notification = "⚠️ No frames recorded!"
                    note_time = time.time()
                    continue
                gesture_path = ensure_gesture_dir(gesture_name)
                arr = np.array(buffer, dtype=np.float32)
                if arr.ndim != 2:
                    notification = f"⚠️ Invalid array shape {arr.shape}"
                    note_time = time.time()
                    continue
# Auto-trim/pad to feature_dim if slightly off
                if arr.shape[1] != feature_dim:
                    fixed = np.zeros((arr.shape[0], feature_dim), dtype=np.float32)
                    trim = min(arr.shape[1], feature_dim)
                    fixed[:, :trim] = arr[:, :trim]
                    arr = fixed

                saved = save_chunks(arr, gesture_path)
                notification = f"✅ Saved {saved} samples to '{gesture_name}'"
                note_time = time.time()
                buffer = []
                is_recording = False

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Data collection finished.")

if __name__ == "__main__":
    run_data_collection()
