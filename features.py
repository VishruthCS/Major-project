import numpy as np
import mediapipe as mp
from collections import deque

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

class FeatureExtractor:
    def __init__(self):
        self.raw_keypoints_buffer = deque(maxlen=3)

    def extract_raw_keypoints(self, results):
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
        return np.concatenate([pose, lh, rh])

    def extract_enhanced_features(self, results):
        base_features = self.extract_raw_keypoints(results)
        self.raw_keypoints_buffer.append(base_features)

        if len(self.raw_keypoints_buffer) < 3:
            return np.zeros(len(base_features) * 3, dtype=np.float32)

        velocity = self.raw_keypoints_buffer[2] - self.raw_keypoints_buffer[1]
        prev_velocity = self.raw_keypoints_buffer[1] - self.raw_keypoints_buffer[0]
        acceleration = velocity - prev_velocity

        return np.concatenate([base_features, velocity, acceleration]).astype(np.float32)

def draw_styled_landmarks(image, results):
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
