# Think Component: Finger command and scoring

class Think(object):

    def __init__(self, act_component):
        """
        Initializes finger commands and scoring.
        :param act_component: Reference to the Act component to trigger visual feedback
        """
        self.act_component = act_component

        self.target_fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']
        self.target_finger_index = 0
        self.target_finger_was_folded = False
        self.score = 0

    def update_finger_state(self, right_hand_folded_fingers):
        # Score the current command once when its right-hand finger is folded.
        target_finger = self.target_fingers[self.target_finger_index]
        target_finger_is_folded = target_finger in right_hand_folded_fingers

        if target_finger_is_folded and not self.target_finger_was_folded:
            self.score += 1
            self.act_component.handle_balloon_inflation()
            self.target_finger_index = (
                self.target_finger_index + 1
            ) % len(self.target_fingers)
            self.target_finger_was_folded = False
        else:
            self.target_finger_was_folded = target_finger_is_folded

    @property
    def target_finger(self):
        return self.target_fingers[self.target_finger_index]
