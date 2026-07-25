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

    def allowed_characters(self) -> set[str]:
        """
        Legacy finite mask for the CURRENT state.
        Not used by the production DFS (which uses accepts_char).
        Subclasses MAY override for finite character sets (digits, keys, etc.).
        Default returns empty set.
        """
        return set()

    @abstractmethod
    def is_terminal(self) -> bool:
        """ Signals if the parsed values has successfully concluded."""
        pass


class NumberState(Enum):
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
    TERMINATORS = {",", "}", "]", " ", "\n", "\t"}
    DIGITS = set("0123456789")
    MINUS_SIGN = {"-"}
    DECIMAL_POINT = {"."}

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
            elif char.isdigit():
                self.state = NumberState.INTEGER_PART
                return True
            return False

        elif self.state == NumberState.AFTER_MINUS:
            # After "-", must have a digit (not another "-" or ".")
            if char == "0":
                self.state = NumberState.AFTER_ZERO
                return True
            elif char.isdigit():
                self.state = NumberState.INTEGER_PART
                return True
            return False

        elif self.state == NumberState.AFTER_ZERO:
            # After "0", can only have ".", "e", "E", or terminator
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
            if char.isdigit():
                return True
            elif char == ".":
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
            if char.isdigit():
                self.state = NumberState.FRACTIONAL_PART
                return True
            return False

        elif self.state == NumberState.FRACTIONAL_PART:
            if char.isdigit():
                return True
            elif char in self.TERMINATORS:
                self.state = NumberState.TERMINAL
                return False
            return False

        return False

    def accepts_char(self, char: str) -> bool:
        if self.state == NumberState.START:
            return char == "-" or char.isdigit()
        elif self.state == NumberState.AFTER_MINUS:
            # After "-", must have a digit
            return char.isdigit()
        elif self.state == NumberState.AFTER_ZERO:
            # After "0", can only have ".", "e", "E", or terminator
            return (
                char == "." or char in ("e", "E") or char in self.TERMINATORS
            )
        elif self.state == NumberState.INTEGER_PART:
            return char.isdigit() or char == "." or char in ("e", "E")
        elif self.state == NumberState.AFTER_DOT:
            # After ".", must have at least one digit
            return char.isdigit()
        elif self.state == NumberState.FRACTIONAL_PART:
            return char.isdigit()
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
        return new

    def allowed_characters(self) -> set[str]:
        if self.state == NumberState.START:
            return self.DIGITS | self.MINUS_SIGN
        elif self.state == NumberState.AFTER_MINUS:
            # After "-", must have a digit
            return self.DIGITS
        elif self.state == NumberState.AFTER_ZERO:
            # After "0", can only have ".", "e", "E", or terminator
            return self.DECIMAL_POINT | {"e", "E"} | self.TERMINATORS
        elif self.state == NumberState.INTEGER_PART:
            return self.DIGITS | self.DECIMAL_POINT | {"e", "E"}
        elif self.state == NumberState.AFTER_DOT:
            # After ".", must have at least one digit
            return self.DIGITS
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
    Strictly enforces opening/closing quotes and handles valid JSON
    escape sequences.
    """
    QUOTE = '"'
    ESCAPE = '\\'
    VALID_ESCAPES = set('"\\/bfnrtu')
    ILLEGAL_RAW_CHARS = set(chr(i) for i in range(32))

    def __init__(self) -> None:
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

    def accepts_char(self, char: str) -> bool:
        if self.state == StringState.EXPECTING_OPEN_QUOTE:
            return char == self.QUOTE
        elif self.state == StringState.INSIDE_STRING:
            return char not in self.ILLEGAL_RAW_CHARS
        elif self.state == StringState.ESCAPE_SEQUENCE:
            return char in self.VALID_ESCAPES
        return False

    def terminates_on(self, char: str) -> bool:
        # StringLiteralFSM only hands back to the PDA when it is already
        # TERMINAL: any advance() then returns False with is_terminal().
        return self.state == StringState.TERMINAL

    def clone(self) -> 'StringLiteralFSM':
        new = StringLiteralFSM.__new__(StringLiteralFSM)
        new.state = self.state
        new.parsed_value = self.parsed_value
        return new

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
