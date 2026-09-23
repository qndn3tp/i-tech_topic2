import cv2
import mediapipe as mp


# Sense Component: Detect hands using the camera
class Sense:

    def __init__(self):
        # Use MediaPipe Hands to track hands in the live video.
        self.mp_hands = mp.solutions.hands.Hands()

    def detect_hands(self, frame):
        # Detect hands and return the MediaPipe hand results.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_hands.process(rgb_frame)
        return results if results else None

    def get_folded_fingers(self, hand_landmarks, handedness):
        # Return folded fingers for either the left or right hand.
        landmarks = hand_landmarks.landmark
        folded_fingers = []

        # The thumb folds sideways, so compare its horizontal x-coordinates.
        thumb_tip = landmarks[mp.solutions.hands.HandLandmark.THUMB_TIP]
        thumb_ip = landmarks[mp.solutions.hands.HandLandmark.THUMB_IP]
        if handedness == 'Right' and thumb_tip.x > thumb_ip.x:
            folded_fingers.append('thumb')
        elif handedness == 'Left' and thumb_tip.x < thumb_ip.x:
            folded_fingers.append('thumb')

        # The other fingers fold down, so compare each fingertip with its PIP joint.
        finger_joints = {
            'index': (mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP,
                      mp.solutions.hands.HandLandmark.INDEX_FINGER_PIP),
            'middle': (mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP,
                       mp.solutions.hands.HandLandmark.MIDDLE_FINGER_PIP),
            'ring': (mp.solutions.hands.HandLandmark.RING_FINGER_TIP,
                     mp.solutions.hands.HandLandmark.RING_FINGER_PIP),
            'pinky': (mp.solutions.hands.HandLandmark.PINKY_TIP,
                      mp.solutions.hands.HandLandmark.PINKY_PIP),
        }

        for finger_name, (tip_index, pip_index) in finger_joints.items():
            if landmarks[tip_index].y > landmarks[pip_index].y:
                folded_fingers.append(finger_name)

        return folded_fingers

