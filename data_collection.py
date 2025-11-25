import os
import time
import cv2
import numpy as np
import mediapipe as mp
from config import Config
from features import FeatureExtractor, draw_styled_landmarks

# --- Setup MediaPipe ---
mp_holistic = mp.solutions.holistic

def ensure_dir(path):
    """Creates directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)

def get_next_sample_id(gesture_path):
    """Finds the next available file number (e.g., 0.npy, 1.npy)."""
    files = [f for f in os.listdir(gesture_path) if f.endswith('.npy')]
    if not files:
        return 0
    # Extract numbers from filenames like "12.npy"
    ids = []
    for f in files:
        try:
            ids.append(int(f.split('.')[0]))
        except ValueError:
            pass
    return max(ids) + 1 if ids else 0

def draw_ui(image, gesture_name, recording, frame_count, status_msg, countdown_val=None):
    """Draws a professional HUD overlay."""
    h, w = image.shape[:2]
    
    # Top Info Bar
    cv2.rectangle(image, (0, 0), (w, 60), (20, 20, 20), -1)
    
    # Gesture Name
    g_text = f"Target: {gesture_name}" if gesture_name else "Target: [NOT SET] (Press 'C')"
    cv2.putText(image, g_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Recording Status / Countdown
    if countdown_val is not None:
        # Big Countdown Center Screen
        cv2.putText(image, str(countdown_val), (w//2 - 20, h//2), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 255), 5)
        status_text = "Get Ready..."
        color = (0, 255, 255)
    elif recording:
        status_text = f"🔴 REC: {frame_count} frames"
        color = (0, 0, 255)
        
        # Progress Bar for Sequence Length
        target = Config.SEQUENCE_LENGTH
        bar_width = 300
        progress = min(frame_count / target, 1.0)
        
        # Bar Background
        cv2.rectangle(image, (w - 320, 20), (w - 20, 40), (50, 50, 50), -1)
        # Bar Fill
        cv2.rectangle(image, (w - 320, 20), (w - 320 + int(bar_width * progress), 40), (0, 255, 0), -1)
        
        # Auto-save hint
        if frame_count >= target:
             cv2.putText(image, "Sample Complete! (Press 'V' to stop)", (w - 350, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    else:
        status_text = "STANDBY"
        color = (0, 255, 0)

    cv2.putText(image, status_text, (w - 250, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Bottom Instructions
    cv2.rectangle(image, (0, h - 40), (w, h), (20, 20, 20), -1)
    cv2.putText(image, "Controls: [C] Set Name | [V] Record (Hold) | [S] Save Last | [Q] Quit", (20, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    
    # Status Message (e.g., "Saved!")
    if status_msg:
        cv2.putText(image, status_msg, (20, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

def run_data_collection():
    # Initialize
    ensure_dir(Config.DATA_PATH)
    cap = cv2.VideoCapture(0)
    
    # Increase Camera Resolution for better hand tracking
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    feature_extractor = FeatureExtractor()
    
    gesture_name = ""
    is_recording = False
    countdown_start = 0
    countdown_active = False
    
    current_frames = []
    status_message = ""
    status_time = 0
    
    print("--- 📷 Enhanced Data Collection Started ---")
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            # Clear status message after 3 seconds
            if time.time() - status_time > 3:
                status_message = ""

            # 1. Process Frame
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # 2. Draw Skeleton
            draw_styled_landmarks(image, results)
            
            # 3. Check for Hands (Quality Control)
            hands_visible = (results.left_hand_landmarks or results.right_hand_landmarks)
            if not hands_visible and not is_recording:
                cv2.putText(image, "⚠️ No Hands Detected", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # 4. Extract Features
            features = feature_extractor.extract_enhanced_features(results)

            # 5. Handle Countdown Logic
            countdown_val = None
            if countdown_active:
                elapsed = time.time() - countdown_start
                if elapsed < 3.0:
                    countdown_val = 3 - int(elapsed)
                else:
                    # Countdown finished, start recording
                    countdown_active = False
                    is_recording = True
                    current_frames = []
                    status_message = "🔴 Recording Started!"
                    status_time = time.time()

            # 6. Recording Logic
            if is_recording:
                current_frames.append(features)

            # 7. Draw Interface
            draw_ui(image, gesture_name, is_recording, len(current_frames), status_message, countdown_val)

            cv2.imshow('Data Collection', image)

            # 8. Controls
            key = cv2.waitKey(10) & 0xFF

            if key == ord('q'): # Quit
                break
            
            elif key == ord('c'): # Change Gesture Name
                is_recording = False
                countdown_active = False
                # Simple console input (cv2 input is complex)
                print("\nEnter new gesture name in terminal:")
                gesture_name = input(">> ").strip()
                print(f"Target set to: {gesture_name}")
                status_message = f"Target: {gesture_name}"
                status_time = time.time()
                
            elif key == ord('v'): # Start/Stop Recording
                if not gesture_name:
                    status_message = "⚠️ Set Name (C) first!"
                    status_time = time.time()
                elif is_recording:
                    # Stop Recording
                    is_recording = False
                    status_message = f"Stopped. {len(current_frames)} frames captured. Press 'S' to save."
                    status_time = time.time()
                elif not countdown_active:
                    # Start Countdown
                    countdown_active = True
                    countdown_start = time.time()

            elif key == ord('s'): # Save
                if not current_frames:
                    status_message = "⚠️ Nothing to save!"
                elif len(current_frames) < 10:
                    status_message = "⚠️ Too short (<10 frames)!"
                else:
                    # Prepare save path
                    g_path = os.path.join(Config.DATA_PATH, gesture_name)
                    ensure_dir(g_path)
                    
                    # Convert to numpy
                    data = np.array(current_frames, dtype=np.float32)
                    
                    # Pad to Sequence Length (Smart Padding)
                    target_len = Config.SEQUENCE_LENGTH
                    if len(data) < target_len:
                        pad = np.tile(data[-1:], (target_len - len(data), 1))
                        data = np.concatenate([data, pad], axis=0)
                    elif len(data) > target_len:
                        # If extremely long, save multiple chunks? 
                        # For now, let's just trim to keep it simple and consistent for training
                        data = data[:target_len]

                    # Save
                    file_id = get_next_sample_id(g_path)
                    save_name = os.path.join(g_path, f"{file_id}.npy")
                    np.save(save_name, data)
                    
                    status_message = f"✅ Saved {gesture_name}/{file_id}.npy"
                    print(f"Saved {save_name} shape={data.shape}")
                    
                    # Reset
                    current_frames = []
                    status_time = time.time()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_data_collection()