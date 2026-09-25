# Think Component: state machine, command selection/scoring, and deciding
# whether the current finger/word command was completed successfully.
#
# Think owns the game's "brain" - which screen we're on, which command is
# active, how much time is left, and what counts as success - and calls
# into Act to render each screen and to speak/beep. This mirrors how
# coach/Think.py calls into Act to trigger the balloon animation.

import random
import time
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum, auto


class GameState(Enum):
    START = auto()
    ACTIVITY = auto()
    LEVEL_COMPLETE = auto()


class CommandType(Enum):
    FINGER = "finger"
    WORD = "word"


@dataclass
class Command:
    """A single instruction shown to the user, e.g. 'curl your right pointer
    finger' or 'say: meeting'."""
    type: CommandType
    # For FINGER commands: hand + finger, e.g. ("Right", "index")
    # For WORD commands: word is set, hand/finger stay None
    hand: str = None
    finger: str = None
    word: str = None

    @property
    def key(self):
        """Unique id used to track per-command difficulty history."""
        if self.type == CommandType.FINGER:
            return f"finger:{self.hand}:{self.finger}"
        return f"word:{self.word}"

    @property
    def prompt_text(self):
        """Human-readable instruction shown on screen / read aloud."""
        if self.type == CommandType.FINGER:
            return f"Curl your {self.hand.lower()} {self.finger} finger!"
        return f"Say: \"{self.word}\""


# Fingers available for finger commands. Extend/trim per hand-sensing capability.
FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
HANDS = ["Right", "Left"]

# Common workplace words for the speech-therapy half of the exercise.
WORKPLACE_WORDS = [
    "meeting", "deadline", "project", "email", "schedule",
    "budget", "report", "client", "manager", "team",
    "invoice", "calendar", "printer", "coffee", "colleague",
]

COMMANDS_PER_LEVEL = 10
START_TIME_LIMIT = 8.0     # seconds given for the first command
TIME_LIMIT_STEP = 0.3      # shaved off the limit after each command
MIN_TIME_LIMIT = 1.5       # never gets faster than this
RESULT_HOLD_SECONDS = 0.8  # how long the correct/incorrect banner stays up


class Think(object):
    def __init__(self, act_component):
        """
        :param act_component: Reference to the Act component, used to render
            each screen and to speak/beep feedback.
        """
        self.act = act_component
        self.state = GameState.START

        self._pool = (
            [Command(CommandType.FINGER, hand=h, finger=f) for h in HANDS for f in FINGERS]
            + [Command(CommandType.WORD, word=w) for w in WORKPLACE_WORDS]
        )
        # key -> number of times the user has gotten this one wrong / timed out.
        self.fail_counts = defaultdict(int)
        # key -> number of times attempted, used only for reporting.
        self.attempt_counts = defaultdict(int)

        self._reset_level()

    def _reset_level(self):
        self.command_index = 0
        self.score = 0
        self.current_command = None
        self.command_started_at = None
        self.current_time_limit = START_TIME_LIMIT
        self.last_result = None
        self._last_result_shown_at = None

    # --- adaptive command selection ----------------------------------------

    def _weight(self, command: Command):
        # Every command has a base weight of 1; each past failure adds +2,
        # so struggled-with fingers/words come up noticeably more often.
        return 1 + 2 * self.fail_counts[command.key]

    def _next_command(self, avoid: Command = None) -> Command:
        """Weighted-random pick. Optionally avoid repeating the same command
        back-to-back so it doesn't feel stuck."""
        pool = self._pool
        if avoid is not None and len(pool) > 1:
            pool = [c for c in pool if c.key != avoid.key]
        weights = [self._weight(c) for c in pool]
        return random.choices(pool, weights=weights, k=1)[0]

    def hardest_items(self, n=3):
        """For the end-of-level summary: the n commands the user found hardest."""
        ranked = sorted(self.fail_counts.items(), key=lambda kv: kv[1], reverse=True)
        return [key for key, fails in ranked[:n] if fails > 0]

    # --- deciding success ---------------------------------------------------

    def check_finger(self, folded_fingers_by_hand, target_hand, target_finger) -> bool:
        folded = folded_fingers_by_hand.get(target_hand, [])
        return folded == [target_finger]

    def check_word(self, sense, target_word) -> bool | None:
        """Check whether the completed transcript contains the target word."""
        transcript = sense.get_speech_result(self.current_command.key)
        if transcript is None:
            return None
        return target_word.lower() in transcript

    # --- state machine -------------------------------------------------------

    def start_level(self, sense):
        self._reset_level()
        self.state = GameState.ACTIVITY
        self._start_next_command(sense)

    def _start_next_command(self, sense):
        self.current_command = self._next_command(avoid=self.current_command)
        self.command_started_at = time.time()
        self.last_result = None

        if self.current_command.type == CommandType.WORD:
            sense.start_listening(self.current_command.key)

        self.act.speak(self.current_command.prompt_text)

    def _finish_command(self, success):
        self.attempt_counts[self.current_command.key] += 1
        if not success:
            self.fail_counts[self.current_command.key] += 1

        self.last_result = success
        self._last_result_shown_at = time.time()
        self.act.give_feedback(success)  # tone + motivational voice line

        if success:
            self.score += 1

        self.command_index += 1
        # Commands get faster as the level goes on.
        self.current_time_limit = max(
            MIN_TIME_LIMIT, START_TIME_LIMIT - self.command_index * TIME_LIMIT_STEP
        )

    def tick(self, sense, frame, key):
        """One frame's worth of decision-making + rendering. Called every
        loop iteration from main2.py."""
        if self.state == GameState.START:
            self.act.render_start()
            if key == ord(' '):
                self.start_level(sense)

        elif self.state == GameState.ACTIVITY:
            self._tick_activity(sense, frame)

        elif self.state == GameState.LEVEL_COMPLETE:
            self.act.render_level_complete(self.score, COMMANDS_PER_LEVEL, self.hardest_items())
            if key == ord(' '):
                self.start_level(sense)

    def _tick_activity(self, sense, frame):
        elapsed = time.time() - self.command_started_at
        time_left = self.current_time_limit - elapsed
        hand_results = None

        # Briefly hold on the correct/incorrect banner before moving on.
        if self.last_result is not None:
            if time.time() - self._last_result_shown_at < RESULT_HOLD_SECONDS:
                speech_status = (
                    sense.get_speech_status(self.current_command.key)
                    if self.current_command.type == CommandType.WORD else None
                )
                self.act.render_activity(
                    frame, self.current_command, self.command_index,
                    COMMANDS_PER_LEVEL, self.score, 0, self.current_time_limit,
                    last_result=self.last_result, hand_results=hand_results,
                    speech_status=speech_status, fold_accuracy=None,
                )
                return
            if self.command_index >= COMMANDS_PER_LEVEL:
                self.state = GameState.LEVEL_COMPLETE
                return
            self._start_next_command(sense)
            elapsed = 0.0
            time_left = self.current_time_limit

        command = self.current_command
        result = None
        fold_accuracy = None

        if command.type == CommandType.FINGER:
            hand_results = sense.detect_hands(frame) if frame is not None else None
            folded_by_hand = sense.get_folded_fingers(hand_results)
            fold_accuracy = sense.get_fold_accuracy(
                hand_results, command.hand, command.finger
            )
            if self.check_finger(folded_by_hand, command.hand, command.finger):
                result = True
        else:  # CommandType.WORD
            outcome = self.check_word(sense, command.word)
            if outcome is not None:
                result = outcome

        speech_status = (
            sense.get_speech_status(command.key)
            if command.type == CommandType.WORD else None
        )

        if result is None and time_left <= 0:
            if command.type == CommandType.FINGER or sense.is_speech_complete(command.key):
                result = False  # timed out or speech was not recognized

        self.act.render_activity(
            frame, command, self.command_index, COMMANDS_PER_LEVEL,
            self.score, max(0.0, time_left), self.current_time_limit,
            hand_results=hand_results, speech_status=speech_status,
            fold_accuracy=fold_accuracy,
        )

        if result is not None:
            self._finish_command(result)