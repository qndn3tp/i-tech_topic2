import cv2
import mediapipe as mp
import math
import numpy as np


# Sense Component: Detect joints using the camera
# Things you need to improve: Make the skeleton tracking smoother and robust to errors.
class Sense:

    def __init__(self):
        # Initialize the Mediapipe Pose object to track joints

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose.Pose()
        # Use MediaPipe Hands to track up to two hands in the live video.
        self.mp_hands = mp.solutions.hands.Hands()

        # used later for having a moving avergage
        self.angle_window = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
        self.previous_angle = -1

    def detect_joints(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_pose.process(rgb_frame)
        return results if results else None

    def detect_hands(self, frame):
        # Detect hands and return the MediaPipe hand results.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_hands.process(rgb_frame)
        return results if results else None

    def get_folded_fingers(self, hand_landmarks, handedness):
        # Return the names of fingers that are currently folded.
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

    def calculate_angle(self, joint1, joint2, joint3):
        """
        Calculates the angle between three joints.

        Parameters:
        - joint1: Tuple of (x, y) for the first joint (e.g., shoulder)
        - joint2: Tuple of (x, y) for the middle joint (e.g., elbow)
        - joint3: Tuple of (x, y) for the last joint (e.g., wrist)

        Returns:
        - Angle in degrees between the three joints
        """
        # Calculate vectors
        vector1 = [joint1[0] - joint2[0], joint1[1] - joint2[1]]
        vector2 = [joint3[0] - joint2[0], joint3[1] - joint2[1]]

        # Calculate the dot product and magnitude of the vectors
        dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
        magnitude1 = math.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
        magnitude2 = math.sqrt(vector2[0] ** 2 + vector2[1] ** 2)

        # Calculate the angle in radians and convert to degrees
        angle = math.acos(dot_product / (magnitude1 * magnitude2))

        # get a moving average
        self.angle_window.pop(0)
        self.angle_window.append(angle)
        # Use np.convolve to calculate the moving average
        window_size = 10
        angle_mvg = np.convolve(np.asarray(self.angle_window), np.ones(window_size) / window_size, mode='valid')

        return math.degrees(angle_mvg)

    def extract_joint_coordinates(self, landmarks, joint):
        """
        Extracts the (x, y) coordinates of a specific joint.

        Parameters:
        - landmarks: The list of pose landmarks from MediaPipe
        - joint: The name of the joint (e.g., 'left_elbow')

        Returns:
        - A tuple of (x, y) coordinates of the specified joint
        """
        joint_index_map = {
            'left_shoulder': mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
            'right_shoulder': mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER,
            'left_elbow': mp.solutions.pose.PoseLandmark.LEFT_ELBOW,
            'right_elbow': mp.solutions.pose.PoseLandmark.RIGHT_ELBOW,
            'left_wrist': mp.solutions.pose.PoseLandmark.LEFT_WRIST,
            'right_wrist': mp.solutions.pose.PoseLandmark.RIGHT_WRIST,
        }

        joint_index = joint_index_map[joint]
        landmark = landmarks.landmark[joint_index]

        return landmark.x, landmark.y
