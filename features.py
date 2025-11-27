# features.py
import numpy as np
import mediapipe as mp
from collections import deque

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

class FeatureExtractor:
    def __init__(self):
        # keep last 3 raw keypoint frames for velocity/accel
        self.raw_keypoints_buffer = deque(maxlen=3)

    def extract_raw_keypoints(self, results):
        """Extract base keypoints for a single frame (258 features)."""
        pose = (np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten()
                if results.pose_landmarks else np.zeros(33 * 4, dtype=np.float32))
        lh = (np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()
              if results.left_hand_landmarks else np.zeros(21 * 3, dtype=np.float32))
        rh = (np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()
              if results.right_hand_landmarks else np.zeros(21 * 3, dtype=np.float32))
        return np.concatenate([pose, lh, rh]).astype(np.float32)  # 258

    def extract_enhanced_features(self, results):
        """
        Returns base + velocity + acceleration = 774 features (float32).
        If buffer doesn't have 3 frames yet, returns zeros(774).
        """
        base = self.extract_raw_keypoints(results)
        # append base to buffer (deque will keep maxlen=3)
        self.raw_keypoints_buffer.append(base)

        if len(self.raw_keypoints_buffer) < 3:
            return np.zeros(774, dtype=np.float32)

        # get last three frames
        f0 = self.raw_keypoints_buffer[-3]
        f1 = self.raw_keypoints_buffer[-2]
        f2 = self.raw_keypoints_buffer[-1]

        velocity = f2 - f1
        prev_velocity = f1 - f0
        acceleration = velocity - prev_velocity

        combined = np.concatenate([f2, velocity, acceleration])

        # pad/trim exactly to 774 just to be extra-safe
        if combined.shape[0] > 774:
            combined = combined[:774]
        elif combined.shape[0] < 774:
            combined = np.pad(combined, (0, 774 - combined.shape[0]), 'constant')

        return combined.astype(np.float32)


def draw_styled_landmarks(image, results):
    """Styled drawing helper (safe wrappers)."""
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80,22,10), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(80,44,121), thickness=1, circle_radius=1)
        )
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
        )
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )
