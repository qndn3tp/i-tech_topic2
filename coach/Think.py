# Think Component: Finger command and scoring

class Think(object):

    def __init__(self, act_component):
        """
        Initializes finger commands and scoring.
        :param act_component: Reference to the Act component to trigger visual feedback
        """
        self.act_component = act_component

        fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']
        # Each hand gets its own command for every finger.
        self.target_commands = [
            (hand, finger)
            for hand in ('Right', 'Left')
            for finger in fingers
        ]
        self.target_command_index = 0
        self.target_finger_was_folded = False
        self.score = 0

    def update_finger_state(self, folded_fingers_by_hand):
        """Score a command only when its target hand has only that finger folded."""
        target_hand, target_finger = self.target_commands[self.target_command_index]
        folded_fingers = folded_fingers_by_hand.get(target_hand, [])
        # Extra folded fingers make the command invalid.
        target_finger_is_folded = folded_fingers == [target_finger]

        if target_finger_is_folded and not self.target_finger_was_folded:
            # Count once per fold, then move to the next hand/finger command.
            self.score += 1
            self.act_component.handle_balloon_inflation()
            self.target_command_index = (
                self.target_command_index + 1
            ) % len(self.target_commands)
            self.target_finger_was_folded = False
        else:
            self.target_finger_was_folded = target_finger_is_folded

    @property
    def target_hand(self):
        return self.target_commands[self.target_command_index][0]

    @property
    def target_finger(self):
        return self.target_commands[self.target_command_index][1]
