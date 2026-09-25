import cv2
from coach import Sense, Think, Act


def main():
    sense = Sense.Sense()      # hand tracking + speech
    act = Act.Act()            # rendering + TTS + tones
    think = Think.Think(act)   # state machine, scoring, adaptive command picking

    searchValidCameraIndexes()

    # Initialize the webcam capture
    cap = cv2.VideoCapture(1) 

    try:
        while True:
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                break

            ret, frame = cap.read() if cap.isOpened() else (False, None)
            if ret:
                # Correct the mirrored webcam view so left and right match the user.
                frame = cv2.flip(frame, 1)

            think.tick(sense, frame, key)
    finally:
        cap.release()
        cv2.destroyAllWindows()


def searchValidCameraIndexes():
    valid_cameras = []

    for index in range(10):
        cap = cv2.VideoCapture(index)
        try:
            if cap.isOpened():
                valid_cameras.append(index)
        finally:
            cap.release()

    return valid_cameras


if __name__ == "__main__":
    main()