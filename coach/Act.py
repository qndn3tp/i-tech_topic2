# Act Component: Provide feedback to the user

import cv2
import mediapipe as mp
import numpy as np
import random
import pyttsx3


# Act Component: Visualization to motivate user, visualization such as the skeleton and debugging information.
# Things to add: Other graphical visualization, a proper GUI, more verbal feedback
class Act:

    def __init__(self):
        # Balloon size and transition tracking for visualization
        self.balloon_size = 50
        self.transition_count = 0
        self.max_transitions = 10  # Explodes after 10 transitions
        self.exploded = False  # Track whether the balloon exploded
        self.explosion_fragments = []  # Store explosion fragments
        self.explosion_frame_count = 0  # Frame counter for explosion duration
        self.explosion_duration = 30  # Number of frames to show explosion effect
        self.engine = pyttsx3.init()

        self.motivating_utterances = ['keep on going', 'you are doing great. I see it', 'only a few left', 'that is awesome', 'you have almost finished the exercise']
        # Handles balloon inflation and reset after explosion

    def handle_balloon_inflation(self):
        """
        Increases the size of the balloon with each successful repetition.
        """
        if not self.exploded:  # Only inflate if balloon hasn't exploded

            self.transition_count += 1
            self.balloon_size += 10  # Inflate balloon by 10 units per transition

            text = random.choice(self.motivating_utterances)
            self.engine.say("%s %s" % (self.transition_count, text))
            self.engine.runAndWait() # This is a blocking call. You need to run it in a thread.

            # Check if balloon should explode

            if self.transition_count >= self.max_transitions:
                self.explode_balloon()

    def explode_balloon(self):
        """
        Handles the visual effect of the balloon exploding.
        """

        self.exploded = True  # Mark the balloon as exploded
        self.create_explosion_fragments()  # Generate the explosion fragments
        self.engine.say("boooom booooom booom")
        self.engine.runAndWait()

    def reset_balloon(self):
        """
        Resets the balloon after it explodes.
        """

        self.transition_count = 0
        self.balloon_size = 50  # Reset balloon size
        self.exploded = False  # Reset explosion state
        self.explosion_frame_count = 0  # Reset the explosion frame counter
        self.explosion_fragments.clear()  # Clear the fragments after explosion

        self.engine.say("You did great! Let's reset the balloon.")
        self.engine.runAndWait()
        # Create explosion fragments with random sizes and positions

    def create_explosion_fragments(self):
        # Generate random "fragments" for explosion effect
        for _ in range(20):
            fragment = {
                'position': (random.randint(200, 300), random.randint(200, 400)),
                'size': random.randint(5, 15),
                'color': (0, 0, 255),  # Red fragments
                'dx': random.randint(-10, 10),  # X-direction movement
                'dy': random.randint(-10, 10)  # Y-direction movement
            }
            self.explosion_fragments.append(fragment)

        # Visualization of the balloon and explosion in OpenCV

    def visualize_balloon(self):
        """
        Renders the balloon .
        """

        # Create a black background
        img = np.zeros((500, 500, 3), dtype=np.uint8)

        if not self.exploded:
            # Draw the balloon (a circle) with dynamic size if it hasn't exploded
            cv2.circle(img, (250, 300), self.balloon_size, (0, 0, 255), -1)  # Red balloon
        else:
            # Draw explosion fragments if balloon has exploded
            for fragment in self.explosion_fragments:
                x, y = fragment['position']
                size = fragment['size']
                color = fragment['color']

                # Move fragments in random directions
                x += fragment['dx']
                y += fragment['dy']
                fragment['position'] = (x, y)

                # Draw each fragment as a small circle
                cv2.circle(img, (x, y), size, color, -1)

            self.explosion_frame_count += 1

            # Reset the balloon after the explosion effect finishes
            if self.explosion_frame_count >= self.explosion_duration:
                self.reset_balloon()

        cv2.putText(img, f'Fold the commanded finger to pop the balloon!', (0, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 2, cv2.LINE_AA)

        # Add transition count and text
        cv2.putText(img, f'Repetitions: {self.transition_count}', (150, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, f'Balloon Size: {self.balloon_size}', (150, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Show the image in the window
        cv2.imshow('Finger Coaching', img)

        # Wait for 1 ms and check if the window should be closed
        cv2.waitKey(1)

    def provide_feedback(self, frame, hand_results, folded_fingers_by_hand,
                         target_hand, target_finger, score):
        """
        Displays both detected hands and the current finger coaching text.

        :param frame: The currently processed frame form the webcam.
        :param hand_results: The detected hand landmarks.
        :param folded_fingers_by_hand: Folded fingers detected for each hand.
        :param target_hand: The hand currently requested by Think.
        :param target_finger: The finger currently requested by Think.
        :param score: The current finger-folding score.

        """

        if hand_results and hand_results.multi_hand_landmarks:
            # Draw every detected hand so both left- and right-hand commands
            # can be followed visually.
            for hand_landmarks, hand_classification in zip(
                hand_results.multi_hand_landmarks,
                hand_results.multi_handedness,
            ):
                mp.solutions.drawing_utils.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp.solutions.hands.HAND_CONNECTIONS,
                )

        # Draw the text on the image
        cv2.putText(
            frame,
            f'Fold your {target_hand.lower()} {target_finger}!  Score: {score}',
            (50, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        # Show the folded-finger list for each hand independently.
        has_folded_fingers = any(folded_fingers_by_hand.values())
        if has_folded_fingers:
            for index, (hand, folded_fingers) in enumerate(
                folded_fingers_by_hand.items()
            ):
                if not folded_fingers:
                    continue
                cv2.putText(
                    frame,
                    f'{hand}: Folded {", ".join(folded_fingers)}',
                    (50, 80 + index * 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )
        else:
            # Ask the user to place a hand in view when no hand is detected.
            cv2.putText(
                frame,
                "Please face your palm toward the camera.",
                (50, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,                            
            )

        # Display the frame (for debugging purposes)
        cv2.imshow('Sport Coaching Program', frame)
