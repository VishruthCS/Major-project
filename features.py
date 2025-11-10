#
# features.py (Final Professional Version)
#
# This file contains helper functions, including our new, brilliant
# enhanced feature extraction logic.
#

import numpy as np
import mediapipe as mp
from collections import deque

# --- MediaPipe Setup ---
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# --- ENHANCED FEATURE EXTRACTION ---
# We use a class to maintain state (like previous frames for velocity)
class FeatureExtractor:
    def __init__(self):
        self.raw_keypoints_buffer = deque(maxlen=3) # Need 3 frames for acceleration

    def extract_raw_keypoints(self, results):
        """Extracts the base keypoints for a single frame."""
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
        return np.concatenate([pose, lh, rh]) # Total 258 features

    def extract_enhanced_features(self, results):
        """
    Extracts enhanced features = base + velocity + acceleration.
    Ensures exact vector length (774).
    """
        base_features = self.extract_raw_keypoints(results)
        self.raw_keypoints_buffer.append(base_features)

    # Keep only last 3 frames in buffer
        if len(self.raw_keypoints_buffer) > 3:
            self.raw_keypoints_buffer.pop(0)

    # Not enough history yet → zero vector
        if len(self.raw_keypoints_buffer) < 3:
            return np.zeros(774, dtype=np.float32)

    # Compute velocity & acceleration
        velocity = self.raw_keypoints_buffer[-1] - self.raw_keypoints_buffer[-2]
        prev_velocity = self.raw_keypoints_buffer[-2] - self.raw_keypoints_buffer[-3]
        acceleration = velocity - prev_velocity

    # Combine and trim/pad exactly to 774
        combined = np.concatenate([base_features, velocity, acceleration])
        if combined.shape[0] > 774:
            combined = combined[:774]
        elif combined.shape[0] < 774:
            combined = np.pad(combined, (0, 774 - combined.shape[0]))

        return combined.astype(np.float32)




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
