from config import Config
from features import FeatureExtractor, draw_styled_landmarks
import cv2, os, numpy as np, time
import mediapipe as mp


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
