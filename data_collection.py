# data_collection.py
import os
import time
import cv2
import numpy as np
from config import Config
from features import FeatureExtractor, draw_styled_landmarks
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def next_index(path):
    files = [f for f in os.listdir(path) if f.endswith('.npy')]
    return len(files)

def save_chunks(sequence, gesture_path, gesture_name=None, seq_len=None):
    """Split sequence into seq_len chunks, pad last chunk if needed."""
    if seq_len is None:
        seq_len = Config.SEQUENCE_LENGTH
    n = len(sequence)
    saved = 0
    for i in range(0, n, seq_len):
        chunk = sequence[i:i+seq_len]
        # pad last if needed
        if len(chunk) < seq_len:
            pad = np.tile(chunk[-1:], (seq_len - len(chunk), 1))
            chunk = np.concatenate([chunk, pad], axis=0)
        elif len(chunk) > seq_len:
            chunk = chunk[:seq_len]
        idx = next_index(gesture_path)
        save_path = os.path.join(gesture_path, f"{idx}.npy")
        np.save(save_path, chunk.astype(np.float32))
        saved += 1
    return saved

def run_data_collection():
    ensure_dir(Config.DATA_PATH)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera open failed.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)

    fe = FeatureExtractor()
    gesture_name = None
    recording = False
    buffer = []
    notification = ""
    note_time = 0

    print("Controls: C=Set gesture | V=Start/Stop record | S=Save | Q=Quit")

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
                pass

            feats = fe.extract_enhanced_features(results)

            if recording:
                buffer.append(feats)
                cv2.putText(image, f"Recording: {len(buffer)} frames", (10, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

            h, w = image.shape[:2]
            cv2.rectangle(image, (0,0),(w,40),(0,0,0),-1)
            cv2.putText(image, "Enhanced Data Collection", (10,28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
            if gesture_name:
                cv2.putText(image, f"Gesture: {gesture_name}", (10,80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
            if recording:
                cv2.putText(image, "🔴 REC", (w-120,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
            if notification and time.time() - note_time < 4:
                cv2.putText(image, notification, (10,h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

            cv2.putText(image, "C=Set Gesture | V=Start/Stop | S=Save | Q=Quit", (10, h-50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            cv2.imshow("Data Collection", image)

            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                gesture_name = input("Gesture name: ").strip()
                if gesture_name:
                    ensure_dir(os.path.join(Config.DATA_PATH, gesture_name))
                    notification = f"Ready for '{gesture_name}'. Press V to record."
                    note_time = time.time()
                    buffer = []
            elif key == ord('v') and gesture_name:
                # toggle recording
                # small debounce
                if time.time() - note_time < 0.25:
                    continue
                recording = not recording
                if recording:
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
                gpath = os.path.join(Config.DATA_PATH, gesture_name)
                ensure_dir(gpath)
                arr = np.array(buffer, dtype=np.float32)
                # shape corrections just in case
                if arr.ndim != 2 or arr.shape[1] != 774:
                    # try to repair: trim/pad per-frame
                    if arr.ndim == 2:
                        fixed = np.zeros((arr.shape[0], 774), dtype=np.float32)
                        trim = min(arr.shape[1], 774)
                        fixed[:, :trim] = arr[:, :trim]
                        arr = fixed
                    else:
                        notification = f"⚠️ Invalid shape {arr.shape}"
                        note_time = time.time()
                        continue
                saved = save_chunks(arr, gpath, gesture_name, seq_len=Config.SEQUENCE_LENGTH)
                notification = f"✅ Saved {saved} samples to '{gesture_name}'"
                note_time = time.time()
                buffer = []
                recording = False

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Collection finished.")

if __name__ == "__main__":
    run_data_collection()
