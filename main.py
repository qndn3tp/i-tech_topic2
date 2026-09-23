import cv2
from coach import Sense
from coach import Think
from coach import Act


# Main Program Loop
def main():
    """
    Main function to initialize the exercise tracking application.

    This function sets up the webcam feed, initializes the Sense, Think, and Act components,
    and starts the main loop to continuously process frames from the webcam.
    """

    
    # Initialize the components: Sense for input, Think for decision-making, Act for output
    sense = Sense.Sense()
    act = Act.Act()
    think = Think.Think(act)


    # Search and print available camera devices (may take a while to complete)
    #searchValidCameraIndexes()
    
    # Initialize the webcam capture
    cap = cv2.VideoCapture(1)  # Use camera index 1; change this if a different camera is connected

    # Main loop to process video frames
    while cap.isOpened():

        # Capture frame-by-frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Correct the mirrored webcam view so left and right match the user.
        frame = cv2.flip(frame, 1)

        # Sense: Detect hands and collect the folded fingers for each visible hand.
        hands = sense.detect_hands(frame)
        folded_fingers = []
        right_hand_folded_fingers = []
        if hands and hands.multi_hand_landmarks:
            for hand_landmarks, hand_classification in zip(
                hands.multi_hand_landmarks,
                hands.multi_handedness,
            ):
                handedness = hand_classification.classification[0].label
                folded = sense.get_folded_fingers(hand_landmarks, handedness)
                if handedness == 'Right':
                    right_hand_folded_fingers = folded
                    folded_fingers = folded
                    print(f'Right hand folded fingers: {folded or "none"}')

        think.update_finger_state(right_hand_folded_fingers)

        act.provide_feedback(
            frame=frame,
            hand_results=hands,
            folded_fingers=folded_fingers,
            target_finger=think.target_finger,
            score=think.score,
        )
        act.visualize_balloon()

        # Exit if the 'q' key is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


def searchValidCameraIndexes():
    # checks the first 10 indexes. May take a while to complete
    
    print(f"Searching available camera index nrs")
    valid_cams = []
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap is None or not cap.isOpened():
            print(f"Warning: unable to open video source: {i}")
        else:
            print(f"Found valid video source: {i}")
            valid_cams.append(i)
            
    print(f"Available camera index nrs: {valid_cams}")

if __name__ == "__main__":
    main()
