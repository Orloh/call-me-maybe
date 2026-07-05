from abc import ABC, abstractmethod
from enum import Enum, auto
import string

class BaseFSM(ABC):
    """
    Abstract base class for character-by-character validation.
    """
    def __init__(self):
        self.state: Enum =self._initial_state()

    @abstractmethod
    def _initial_state(self) -> Enum:
        """Defines the starting state of the FSM."""
        pass

    @abstractmethod
    def advance(self, char: str) -> bool:
        """
        Processes a single char and updates internal state.
        Returns True if the char is legally consumed.
        Returns False if the char is illegal OR belongs to the PDA
        """
        pass

    @abstractmethod
    def allowed_characters(self) -> set[str]:
        """
        Returns all legal characters for the CURRENT state.
        This is the engine's direct pipeline to the Trie for token pruning.
        """
        pass

    @abstractmethod
    def is_terminal(self) -> bool:
        """ Signals if the parsed values has successfully concluded."""
        pass


class NumberState(Enum):
    START = auto()
    INTEGER_PART = auto()
    FRACTIONAL_PART = auto()
    TERMINAL = auto()


class NumberFSM(BaseFSM):
    """
    Validates flat integer and float types character-by-character.
    """
    TERMINATORS = {",", "}", "]", " ", "\n", "\t"}
    DIGITS = set("0123456789")
    MINUS_SIGN = {"-"}
    DECIMAL_POINT = {"."}

    def _initial_state(self) -> Enum:
        return NumberState.START

    def advance(self, char: str) -> bool:
        if self.state == NumberState.START:
            if char == "-" or char.isdigit():
                self.state = NumberState.INTEGER_PART
                return True
            return False

        elif self.state == NumberState.INTEGER_PART:
            if char.isdigit():
                return True
            elif char == ".":
                self.state = NumberState.FRACTIONAL_PART
                return True
            elif char in self.TERMINATORS:
                self.state = NumberState.TERMINAL
                return False
            return False

        elif self.state == NumberState.FRACTIONAL_PART:
            if char.isdigit():
                return True
            elif char in self.TERMINATORS:
                self.state = NumberState.TERMINAL
                return False
            return False

        return False

    def allowed_characters(self) -> set[str]:
        if self.state == NumberState.START:
            return self.DIGITS | self.MINUS_SIGN
        elif self.state == NumberState.INTEGER_PART:
            return self.DIGITS | self.DECIMAL_POINT | self.TERMINATORS
        elif self.state == NumberState.FRACTIONAL_PART:
            return self.DIGITS | self.TERMINATORS
        return set()

    def is_terminal(self) -> bool:
        return self.state == NumberState.TERMINAL


class StringState(Enum):
    EXPECTING_OPEN_QUOTE = auto()
    INSIDE_STRING = auto()
    ESCAPE_SEQUENCE = auto()
    TERMINAL = auto()

class StringLiteralFSM(BaseFSM):
    """
    Validates JSON string literals character-by-character.
    Strictly enforces opening/closing quotes and handles valid JSON escape sequences.
    """
    QUOTE = '"'
    ESCAPE = '\\'
    VALID_ESCAPES = set('"\\/bfnrtu')
    ILLEGAL_RAW_CHARS = set(chr(i) for i in range(32))

    def __init__(self):
        super().__init__()
        self.parsed_value = ""
    
    def _initial_state(self) -> Enum:
        return StringState.EXPECTING_OPEN_QUOTE

    def advance(self, char: str) -> bool:
        if self.state == StringState.EXPECTING_OPEN_QUOTE:
            if char == self.QUOTE:
                self.state = StringState.INSIDE_STRING
                self.parsed_value += char
                return True
            return False

        elif self.state == StringState.INSIDE_STRING:
            if char == self.ESCAPE:
                self.state = StringState.ESCAPE_SEQUENCE
                self.parsed_value += char
                return True
            elif char == self.QUOTE:
                self.state = StringState.TERMINAL
                self.parsed_value += char
                return True
            elif char in self.ILLEGAL_RAW_CHARS:
                return False
            else:
                self.parsed_value += char
                return True

        elif self.state == StringState.ESCAPE_SEQUENCE:
            if char in self.VALID_ESCAPES:
                self.state = StringState.INSIDE_STRING
                self.parsed_value += char
                return True
            return False

        return False

    def allowed_characters(self) -> set[str]:
        if self.state == StringState.EXPECTING_OPEN_QUOTE:
            return {self.QUOTE}

        elif self.state == StringState.INSIDE_STRING:
            allowed = set(string.printable) - self.ILLEGAL_RAW_CHARS
            return allowed

        elif self.state == StringState.ESCAPE_SEQUENCE:
            return self.VALID_ESCAPES

        return set()

    def is_terminal(self) -> bool:
        return self.state == StringState.TERMINAL

    
class ExactMatchState(Enum):
    MATCHING = auto()
    TERMINAL = auto()


class ExactMatchFSM(BaseFSM):
    """
    Validates exact string matches character-by-character.
    Dynamically tracks valid candidates, making it perfect for
    booleans ('true', 'false'), nulls ('null') and schema Enums.
    """

    def __init__(self, valid_strings: str | list[str]) -> None:
        if not valid_strings:
            raise ValueError("ExactMatchFSM require at least one valid string.")

        self.active_candidates = valid_strings
        self.current_index = 0
        super().__init__()

    def _initial_state(self) -> Enum:
       return ExactMatchState.MATCHING

    def advance(self, char: str) -> bool:
        if self.state == ExactMatchState.TERMINAL:
            return False

        valid_next = [
            candidate for candidate in self.active_candidates
            if self.current_index < len(candidate) and candidate[self.current_index] == char
        ]

        if not valid_next:
            return False

        self.active_candidates = valid_next
        self.current_index += 1

        if any(self.current_index == len(candidate) for candidate in self.active_candidates):
            self.state = ExactMatchState.TERMINAL

        return True

    def allowed_characters(self) -> set[str]:
        if self.state == ExactMatchState.TERMINAL:
            return set()

        return {
            candidate[self.current_index]
            for candidate in self.active_candidates
            if self.current_index < len(candidate)
        }

    def is_terminal(self) -> bool:
        return self.state == ExactMatchState.TERMINAL
