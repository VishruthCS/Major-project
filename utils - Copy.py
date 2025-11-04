#
# Utility Functions (Final Professional Version)
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
        Extracts a rich set of features including raw keypoints, velocity,
        and acceleration. This is the "smarter textbook" for our model.
        """
        base_features = self.extract_raw_keypoints(results)
        
        # Store the raw keypoints for calculating derivatives
        self.raw_keypoints_buffer.append(base_features)

        # Ensure we have enough history to calculate features
        if len(self.raw_keypoints_buffer) < 3:
            # If not enough history, return a zero vector of the final expected shape
            # The final shape is base + velocity + acceleration = 258 * 3 = 774
            return np.zeros(len(base_features) * 3, dtype=np.float32)

        # 2. Velocity (current frame - previous frame)
        velocity = self.raw_keypoints_buffer[2] - self.raw_keypoints_buffer[1]

        # 3. Acceleration (current velocity - previous velocity)
        previous_velocity = self.raw_keypoints_buffer[1] - self.raw_keypoints_buffer[0]
        acceleration = velocity - previous_velocity

        # 4. Combine all features into one vector
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

