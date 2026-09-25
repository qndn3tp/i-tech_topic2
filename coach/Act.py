# Act Component: renders each screen and provides audio feedback (TTS + tones). 
# Kept separate from Think so the look/sound of the game can change
# without touching the state machine or scoring.

import queue
import random
import threading

import cv2
import mediapipe as mp
import pyttsx3

try:
    import winsound
    def _beep(freq, duration_ms):
        winsound.Beep(freq, duration_ms)
except ImportError:
    def _beep(freq, duration_ms):
        pass


WINDOW_NAME = "Finger & Speech Coach"
CANVAS_SIZE = (500, 700, 3)  # height, width, channels

WHITE = (255, 255, 255)
GREEN = (0, 200, 0)
RED = (0, 0, 255)
YELLOW = (0, 220, 220)
GREY = (120, 120, 120)

TARGET_FINGER_LANDMARKS = {
    "thumb": (
        mp.solutions.hands.HandLandmark.THUMB_CMC,
        mp.solutions.hands.HandLandmark.THUMB_MCP,
        mp.solutions.hands.HandLandmark.THUMB_IP,
        mp.solutions.hands.HandLandmark.THUMB_TIP,
    ),
    "index": (
        mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP,
        mp.solutions.hands.HandLandmark.INDEX_FINGER_PIP,
        mp.solutions.hands.HandLandmark.INDEX_FINGER_DIP,
        mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP,
    ),
    "middle": (
        mp.solutions.hands.HandLandmark.MIDDLE_FINGER_MCP,
        mp.solutions.hands.HandLandmark.MIDDLE_FINGER_PIP,
        mp.solutions.hands.HandLandmark.MIDDLE_FINGER_DIP,
        mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP,
    ),
    "ring": (
        mp.solutions.hands.HandLandmark.RING_FINGER_MCP,
        mp.solutions.hands.HandLandmark.RING_FINGER_PIP,
        mp.solutions.hands.HandLandmark.RING_FINGER_DIP,
        mp.solutions.hands.HandLandmark.RING_FINGER_TIP,
    ),
    "pinky": (
        mp.solutions.hands.HandLandmark.PINKY_MCP,
        mp.solutions.hands.HandLandmark.PINKY_PIP,
        mp.solutions.hands.HandLandmark.PINKY_DIP,
        mp.solutions.hands.HandLandmark.PINKY_TIP,
    ),
}

# Spoken after a correct command.
SUCCESS_PHRASES = [
    "Great job!",
    "You're doing amazing!",
    "Keep it up!",
    "Nice work!",
    "That's the way!",
    "Excellent!",
    "Well done!",
    "You're on a roll!",
]

# Spoken after a missed/incorrect/timed-out command - keeps morale up
ENCOURAGEMENT_PHRASES = [
    "Almost there, try again!",
    "You can do it, keep going!",
    "No worries, next one!",
    "Stay with it, you're improving!",
    "That's okay, let's try the next one!",
    "Don't give up, you've got this!",
]


class Act:
    def __init__(self):
        # Speech runs on its own background thread, fed through a queue, so
        # calling speak() never blocks the game loop (camera feed / timer
        # bar keep updating while a line is being read aloud), and lines
        # play one at a time in order instead of overlapping.
        #
        # IMPORTANT: the pyttsx3 engine is created ONCE, inside the worker
        # thread, and reused for every queued line. Recreating a fresh
        # engine per utterance (or creating it outside this thread) is what
        # causes total silence / only-the-first-line-plays symptoms with
        # pyttsx3's Windows SAPI5 driver - one persistent engine on one
        # dedicated thread is the combination that actually stays reliable.
        self._speech_queue = queue.Queue()
        self._speech_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self._speech_thread.start()

    def _speech_worker(self):
        engine = pyttsx3.init()
        while True:
            text = self._speech_queue.get()
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception as exc:
                # Surface TTS errors in the console instead of failing silently.
                print(f"[Act] speech error: {exc}")
            finally:
                self._speech_queue.task_done()

    def speak(self, text):
        """Queues a line to be spoken; returns immediately (non-blocking)."""
        self._speech_queue.put(text)

    def play_result_tone(self, success):
        if success:
            _beep(1000, 150)  # positive tone
        else:
            _beep(300, 300)   # negative tone

    def give_feedback(self, success):
        """Called once per finished command: plays the correct/wrong tone
        AND speaks an encouraging line, so every challenge - win or lose -
        ends with some motivation."""
        self.play_result_tone(success)
        phrase = random.choice(SUCCESS_PHRASES if success else ENCOURAGEMENT_PHRASES)
        self.speak(phrase)

    def render_start(self):
        """Start page: title + 'press SPACE to begin' prompt."""
        img = _blank_canvas()
        _put_text(img, "Finger & Speech Coach", (45, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, WHITE, 3, cv2.LINE_AA)
        _put_text(img, "Curl the finger you're told,", (45, 205),
              cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREY, 2, cv2.LINE_AA)
        _put_text(img, "or say the word shown.", (45, 245),
              cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREY, 2, cv2.LINE_AA)
        _put_text(img, "10 commands per level", (45, 305),
              cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREY, 2, cv2.LINE_AA)
        _put_text(img, "Press SPACE to start", (45, 390),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, YELLOW, 2, cv2.LINE_AA)
        _show(img)

    def render_activity(self, frame, command, command_index, total_commands,
                         score, time_left, time_limit, last_result=None,
                         hand_results=None, speech_status=None,
                         fold_accuracy=None):
        """
        Activity page: current command, countdown bar, running score, and
        (if a finger command) the camera feed with hand landmarks drawn on it.
        """
        img = frame if frame is not None else _blank_canvas()

        if hand_results and hand_results.multi_hand_landmarks:
            for hand_landmarks, hand_classification in zip(
                hand_results.multi_hand_landmarks,
                hand_results.multi_handedness,
            ):
                detected_hand = hand_classification.classification[0].label
                if (command.type.value == "finger"
                        and detected_hand == command.hand):
                    _draw_target_finger(img, hand_landmarks, command.finger)

        _put_text(img, command.prompt_text, (30, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 2, cv2.LINE_AA)
        _put_text(img, f"Command {command_index + 1} / {total_commands}",
              (30, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREY, 2, cv2.LINE_AA)
        _put_text(img, f"Score: {score}", (30, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2, cv2.LINE_AA)

        # Countdown bar - shrinks as time runs out, and the overall bar length
        # reflects how tight the current time limit is (gets shorter each command).
        bar_x, bar_y, bar_h = 30, 190, 18
        bar_w_full = 300
        fraction_left = max(0.0, min(1.0, time_left / time_limit))
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w_full, bar_y + bar_h), GREY, 2)
        cv2.rectangle(
            img, (bar_x, bar_y), (bar_x + int(bar_w_full * fraction_left), bar_y + bar_h),
            YELLOW if fraction_left > 0.3 else RED, -1,
        )

        if speech_status is not None:
            _put_text(img, speech_status, (30, 245),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, YELLOW, 2, cv2.LINE_AA)

        if fold_accuracy is not None:
            if fold_accuracy >= 90:
                feedback, feedback_color = "Perfect!", GREEN
            elif fold_accuracy >= 60:
                feedback, feedback_color = "Almost there, fold a little more.", YELLOW
            else:
                feedback, feedback_color = "Fold it more.", RED
            _put_text(img, f"Fold accuracy: {fold_accuracy}%",
                        (30, 295), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        WHITE, 2, cv2.LINE_AA)
            _put_text(img, feedback, (30, 340),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        feedback_color, 2, cv2.LINE_AA)

        if last_result is not None:
            text, color = ("Correct!", GREEN) if last_result else ("Try again next time", RED)
            _put_text(img, text, (30, 390 if fold_accuracy is not None else 295),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

        _show(img)

    def render_level_complete(self, score, total_commands, hardest_items):
        img = _blank_canvas()
        _put_text(img, "Level complete!", (45, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, WHITE, 3, cv2.LINE_AA)
        _put_text(img, f"Score: {score} / {total_commands}", (45, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, GREEN, 2, cv2.LINE_AA)

        if hardest_items:
            _put_text(img, "We'll practice these more", (45, 280),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREY, 2, cv2.LINE_AA)
            _put_text(img, "next time:", (45, 320),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREY, 2, cv2.LINE_AA)
            for i, item in enumerate(hardest_items):
                _put_text(img, f"- {item}", (65, 355 + i * 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, YELLOW, 2, cv2.LINE_AA)

        _put_text(img, "Press SPACE to play again", (45, 455),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2, cv2.LINE_AA)
        _put_text(img, "or Q to quit", (45, 490),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2, cv2.LINE_AA)
        _show(img)


def _blank_canvas():
    import numpy as np
    return np.zeros(CANVAS_SIZE, dtype="uint8")


def _put_text(img, text, position, font, scale, color, thickness,
              line_type=cv2.LINE_AA):
    cv2.putText(
        img, text, position, font, scale * 2, color, thickness, line_type
    )


def _draw_target_finger(img, hand_landmarks, finger_name):
    landmark_ids = TARGET_FINGER_LANDMARKS.get(finger_name)
    if landmark_ids is None:
        return

    height, width = img.shape[:2]
    points = [
        (
            int(hand_landmarks.landmark[landmark_id].x * width),
            int(hand_landmarks.landmark[landmark_id].y * height),
        )
        for landmark_id in landmark_ids
    ]

    for start, end in zip(points, points[1:]):
        cv2.line(img, start, end, YELLOW, 4, cv2.LINE_AA)
    for point in points:
        cv2.circle(img, point, 7, RED, -1, cv2.LINE_AA)


def _show(img):
    if not hasattr(_show, "window_initialized"):
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(
            WINDOW_NAME,
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN,
        )
        _show.window_initialized = True
    cv2.imshow(WINDOW_NAME, img)
    