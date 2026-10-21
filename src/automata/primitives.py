from abc import ABC, abstractmethod
from enum import Enum, auto


class BaseFSM(ABC):
    """
    Abstract base class for character-by-character validation.
    """
    def __init__(self) -> None:
        self.state: Enum = self._initial_state()

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
    def accepts_char(self, char: str) -> bool:
        """
        Non-mutating mirror of advance(): returns True iff advance(char)
        would consume the char in the CURRENT state. Used by the DFS to
        pre-filter trie edges without cloning.
        """
        pass

    @abstractmethod
    def terminates_on(self, char: str) -> bool:
        """
        Non-mutating mirror of the hand-off protocol: returns True iff
        advance(char) would return False AND the FSM would be terminal
        afterwards (i.e. the char belongs to the PDA for re-dispatch).
        """
        pass

    @abstractmethod
    def clone(self) -> 'BaseFSM':
        """
        Cheap copy: shares immutable fields, copies only mutable state.
        Much faster than copy.deepcopy (no memo dict, no traversal).
        """
        pass

    @abstractmethod
    def is_terminal(self) -> bool:
        """ Signals if the parsed values has successfully concluded."""
        pass


class NumberState(Enum):
    """States of the JSON number FSM."""

    START = auto()
    AFTER_MINUS = auto()  # Just saw "-", need digit
    AFTER_ZERO = auto()  # Just saw "0", need "."/terminator
    INTEGER_PART = auto()  # Non-zero digits
    AFTER_DOT = auto()  # Just saw ".", need digit
    FRACTIONAL_PART = auto()  # Digits after "."
    TERMINAL = auto()


class NumberFSM(BaseFSM):
    """
    Validates flat integer and float types character-by-character.
    """
    TERMINATORS = {",", "}", "]", " ", "\n", "\r", "\t"}
    DIGITS = set("0123456789")

    def __init__(self, mode: str = "number") -> None:
        self.mode = mode  # "number" or "integer"
        super().__init__()

    def _initial_state(self) -> Enum:
        return NumberState.START

    def advance(self, char: str) -> bool:
        if self.state == NumberState.START:
            if char == "-":
                self.state = NumberState.AFTER_MINUS
                return True
            elif char == "0":
                self.state = NumberState.AFTER_ZERO
                return True
            elif char in self.DIGITS:
                self.state = NumberState.INTEGER_PART
                return True
            return False

        elif self.state == NumberState.AFTER_MINUS:
            # After "-", must have a digit (not another "-" or ".")
            if char == "0":
                self.state = NumberState.AFTER_ZERO
                return True
            elif char in self.DIGITS:
                self.state = NumberState.INTEGER_PART
                return True
            return False

        elif self.state == NumberState.AFTER_ZERO:
            # After "0", can only have ".", "e", "E", or terminator
            if self.mode == "integer":
                # Integers can't have a decimal point
                if char in ("e", "E"):
                    return False
                elif char in self.TERMINATORS:
                    self.state = NumberState.TERMINAL
                    return False
                return False
            if char == ".":
                self.state = NumberState.AFTER_DOT
                return True
            elif char in ("e", "E"):
                # Exponent not implemented yet, reject
                return False
            elif char in self.TERMINATORS:
                self.state = NumberState.TERMINAL
                return False
            return False

        elif self.state == NumberState.INTEGER_PART:
            if char in self.DIGITS:
                return True
            elif self.mode == "number" and char == ".":
                self.state = NumberState.AFTER_DOT
                return True
            elif char in ("e", "E"):
                # Exponent not implemented yet, reject
                return False
            elif char in self.TERMINATORS:
                self.state = NumberState.TERMINAL
                return False
            return False

        elif self.state == NumberState.AFTER_DOT:
            # After ".", must have at least one digit
            if char in self.DIGITS:
                self.state = NumberState.FRACTIONAL_PART
                return True
            return False

        elif self.state == NumberState.FRACTIONAL_PART:
            if char in self.DIGITS:
                return True
            elif char in self.TERMINATORS:
                self.state = NumberState.TERMINAL
                return False
            return False

        return False

    def accepts_char(self, char: str) -> bool:
        if self.state == NumberState.START:
            return char == "-" or char in self.DIGITS
        elif self.state == NumberState.AFTER_MINUS:
            # After "-", must have a digit
            return char in self.DIGITS
        elif self.state == NumberState.AFTER_ZERO:
            # After "0", can only have "." or a terminator (exponents rejected)
            if self.mode == "integer":
                return char in self.TERMINATORS
            return char == "." or char in self.TERMINATORS
        elif self.state == NumberState.INTEGER_PART:
            if self.mode == "integer":
                return char in self.DIGITS
            return char in self.DIGITS or char == "."
        elif self.state == NumberState.AFTER_DOT:
            # After ".", must have at least one digit
            return char in self.DIGITS
        elif self.state == NumberState.FRACTIONAL_PART:
            return char in self.DIGITS
        return False

    def terminates_on(self, char: str) -> bool:
        if self.state == NumberState.TERMINAL:
            return True
        # Can only terminate if we've consumed at least one digit
        if self.state in (
            NumberState.AFTER_ZERO,
            NumberState.INTEGER_PART,
            NumberState.FRACTIONAL_PART
        ):
            return char in self.TERMINATORS
        return False

    def clone(self) -> 'NumberFSM':
        new = NumberFSM.__new__(NumberFSM)
        new.state = self.state
        new.mode = self.mode
        return new

    def is_terminal(self) -> bool:
        return self.state == NumberState.TERMINAL


class StringState(Enum):
    """States of the JSON string-literal FSM."""

    EXPECTING_OPEN_QUOTE = auto()
    INSIDE_STRING = auto()
    ESCAPE_SEQUENCE = auto()
    HEX_SEQUENCE = auto()
    TERMINAL = auto()


class StringLiteralFSM(BaseFSM):
    """
    Validates JSON string literals character-by-character.
    Strictly enforces opening/closing quotes and handles valid JSON
    escape sequences.
    """
    QUOTE = '"'
    ESCAPE = '\\'
    VALID_ESCAPES = set('"\\/bfnrtu')
    HEX_DIGITS = set("0123456789abcdefABCDEF")
    ILLEGAL_RAW_CHARS = set(chr(i) for i in range(32))

    def __init__(self) -> None:
        super().__init__()
        self.parsed_value = ""
        self.hex_digits_remaining = 0

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
            if char == "u":
                self.state = StringState.HEX_SEQUENCE
                self.hex_digits_remaining = 4
                self.parsed_value += char
                return True
            if char in self.VALID_ESCAPES:
                self.state = StringState.INSIDE_STRING
                self.parsed_value += char
                return True
            return False

        elif self.state == StringState.HEX_SEQUENCE:
            if char in self.HEX_DIGITS:
                self.parsed_value += char
                self.hex_digits_remaining -= 1
                if self.hex_digits_remaining == 0:
                    self.state = StringState.INSIDE_STRING
                return True
            return False

        return False

    def accepts_char(self, char: str) -> bool:
        if self.state == StringState.EXPECTING_OPEN_QUOTE:
            return char == self.QUOTE
        elif self.state == StringState.INSIDE_STRING:
            return char not in self.ILLEGAL_RAW_CHARS
        elif self.state == StringState.ESCAPE_SEQUENCE:
            return char in self.VALID_ESCAPES
        elif self.state == StringState.HEX_SEQUENCE:
            return char in self.HEX_DIGITS
        return False

    def terminates_on(self, char: str) -> bool:
        return self.state in (StringState.TERMINAL, )

    def clone(self) -> 'StringLiteralFSM':
        new = StringLiteralFSM.__new__(StringLiteralFSM)
        new.state = self.state
        new.parsed_value = self.parsed_value
        new.hex_digits_remaining = self.hex_digits_remaining
        return new

    def is_terminal(self) -> bool:
        return self.state == StringState.TERMINAL


class ExactMatchState(Enum):
    """States of the exact-match FSM."""

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
            raise ValueError(
                "ExactMatchFSM require at least one valid string."
            )

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
            if (
                self.current_index < len(candidate)
                and candidate[self.current_index] == char
            )
        ]

        if not valid_next:
            return False

        self.active_candidates = valid_next
        self.current_index += 1

        fully_matched = any(
            self.current_index == len(candidate)
            for candidate in self.active_candidates
        )
        if fully_matched:
            self.state = ExactMatchState.TERMINAL

        return True

    def accepts_char(self, char: str) -> bool:
        if self.state == ExactMatchState.TERMINAL:
            return False

        return any(
            self.current_index < len(candidate)
            and candidate[self.current_index] == char
            for candidate in self.active_candidates
        )

    def terminates_on(self, char: str) -> bool:
        # ExactMatchFSM only hands back to the PDA when it is already
        # TERMINAL: any advance() then returns False with is_terminal().
        return self.state == ExactMatchState.TERMINAL

    def clone(self) -> 'ExactMatchFSM':
        # active_candidates is only ever reassigned, never mutated in
        # place, so sharing the list reference across clones is safe.
        new = ExactMatchFSM.__new__(ExactMatchFSM)
        new.state = self.state
        new.active_candidates = self.active_candidates
        new.current_index = self.current_index
        return new

    def is_terminal(self) -> bool:
        return self.state == ExactMatchState.TERMINAL
