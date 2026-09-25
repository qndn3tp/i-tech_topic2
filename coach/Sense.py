import threading

import cv2
import mediapipe as mp
import numpy as np
import speech_recognition as sr


class Sense:

    def __init__(self):
        self.mp_hands = mp.solutions.hands.Hands()
        self._speech_lock = threading.Lock()
        self._speech_command_key = None
        self._speech_result = None
        self._speech_complete = False
        self._speech_status = None

    def detect_hands(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_hands.process(rgb_frame)
        return results if results else None

    def get_folded_fingers(self, frame_results):
        folded_by_hand = {'Right': [], 'Left': []}
        if frame_results and frame_results.multi_hand_landmarks:
            for hand_landmarks, hand_classification in zip(
                frame_results.multi_hand_landmarks,
                frame_results.multi_handedness,
            ):
                handedness = hand_classification.classification[0].label
                folded = self._get_folded_fingers_for_hand(
                    hand_landmarks, handedness
                )
                if handedness in folded_by_hand:
                    folded_by_hand[handedness] = folded
        return folded_by_hand

    def _get_folded_fingers_for_hand(self, hand_landmarks, handedness):
        landmarks = hand_landmarks.landmark
        folded_fingers = []

        thumb_tip = landmarks[mp.solutions.hands.HandLandmark.THUMB_TIP]
        thumb_ip = landmarks[mp.solutions.hands.HandLandmark.THUMB_IP]
        if handedness == 'Right' and thumb_tip.x > thumb_ip.x:
            folded_fingers.append('thumb')
        elif handedness == 'Left' and thumb_tip.x < thumb_ip.x:
            folded_fingers.append('thumb')

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

    def get_fold_accuracy(self, frame_results, target_hand, finger_name):
        finger_points = {
            'thumb': (mp.solutions.hands.HandLandmark.THUMB_MCP,
                      mp.solutions.hands.HandLandmark.THUMB_IP,
                      mp.solutions.hands.HandLandmark.THUMB_TIP),
            'index': (mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP,
                      mp.solutions.hands.HandLandmark.INDEX_FINGER_PIP,
                      mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP),
            'middle': (mp.solutions.hands.HandLandmark.MIDDLE_FINGER_MCP,
                       mp.solutions.hands.HandLandmark.MIDDLE_FINGER_PIP,
                       mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP),
            'ring': (mp.solutions.hands.HandLandmark.RING_FINGER_MCP,
                     mp.solutions.hands.HandLandmark.RING_FINGER_PIP,
                     mp.solutions.hands.HandLandmark.RING_FINGER_TIP),
            'pinky': (mp.solutions.hands.HandLandmark.PINKY_MCP,
                      mp.solutions.hands.HandLandmark.PINKY_PIP,
                      mp.solutions.hands.HandLandmark.PINKY_TIP),
        }
        points = finger_points.get(finger_name)
        if not points or not frame_results or not frame_results.multi_hand_landmarks:
            return 0

        for hand_landmarks, hand_classification in zip(
            frame_results.multi_hand_landmarks,
            frame_results.multi_handedness,
        ):
            if hand_classification.classification[0].label != target_hand:
                continue

            mcp_index, joint_index, tip_index = points
            landmarks = hand_landmarks.landmark
            mcp = landmarks[mcp_index]
            joint = landmarks[joint_index]
            tip = landmarks[tip_index]
            first_vector = np.array([mcp.x - joint.x, mcp.y - joint.y])
            second_vector = np.array([tip.x - joint.x, tip.y - joint.y])
            denominator = np.linalg.norm(first_vector) * np.linalg.norm(second_vector)
            if denominator == 0:
                return 0

            cosine = np.dot(first_vector, second_vector) / denominator
            angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
            return int(round(np.clip((180 - angle) / 180 * 100, 0, 100)))

        return 0

    def start_listening(self, command_key):
        with self._speech_lock:
            self._speech_command_key = command_key
            self._speech_result = None
            self._speech_complete = False
            self._speech_status = "Listening..."

        listener = threading.Thread(
            target=self._listen_for_command,
            args=(command_key,),
            daemon=True,
        )
        listener.start()

    def get_speech_result(self, command_key):
        with self._speech_lock:
            if self._speech_command_key != command_key or not self._speech_complete:
                return None
            return self._speech_result

    def is_speech_complete(self, command_key):
        with self._speech_lock:
            return (self._speech_command_key == command_key
                    and self._speech_complete)

    def get_speech_status(self, command_key):
        with self._speech_lock:
            if self._speech_command_key != command_key:
                return None
            return self._speech_status

    def _listen_for_command(self, command_key):
        recognizer = sr.Recognizer()
        transcript = ""
        try:
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = recognizer.record(source, duration=3)
            with self._speech_lock:
                if self._speech_command_key == command_key:
                    self._speech_status = "Transcribing..."
            transcript = recognizer.recognize_google(audio).lower()
            with self._speech_lock:
                if self._speech_command_key == command_key:
                    self._speech_status = f"Heard: '{transcript}'"
        except sr.UnknownValueError:
            self._set_speech_status(command_key, "Could not understand audio")
        except sr.RequestError:
            self._set_speech_status(command_key, "Speech service unavailable")
        except (OSError, AttributeError):
            self._set_speech_status(command_key, "Microphone error")

        with self._speech_lock:
            if self._speech_command_key == command_key:
                self._speech_result = transcript
                self._speech_complete = True

    def _set_speech_status(self, command_key, status):
        with self._speech_lock:
            if self._speech_command_key == command_key:
                self._speech_status = status

