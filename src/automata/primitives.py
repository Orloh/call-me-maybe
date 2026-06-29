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
