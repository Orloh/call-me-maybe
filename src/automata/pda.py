from typing import Callable
from enum import Enum, auto
from .primitives import BaseFSM, StringLiteralFSM
from .compiler import CompiledSchema


class Scope(Enum):
    """
    Represents the structural context stored in the PDA's LIFO stack.
    """
    OBJECT = auto()
    ARRAY = auto()


class PDAState(Enum):
    """
    Defines the high-level structural states of the JSONPushdownAutomaton.
    """
    # Object states
    EXPECTING_OBJECT_START = auto()
    EXPECTING_KEY = auto()
    EXPECTING_COLON = auto()

    # Array states
    EXPECTING_ARRAY_START = auto()

    #Shared states
    EXPECTING_VALUE = auto()
    EXPECTING_COMMA_OR_END = auto()
    TERMINAL = auto()


class JSONPushdownAutomaton:
    # Structural Grammar Definitions
    CHAR_OBJECT_START = frozenset("{")
    CHAR_ARRAY_START = frozenset("[")
    CHAR_KEY_QUOTE = frozenset('"')
    CHAR_COLON = frozenset(":")
    CHARS_OBJECT_NEXT = frozenset({",", "}"})
    CHARS_ARRAY_NEXT = frozenset({",", "]"})
    CHARS_EMPTY = frozenset()
    WHITESPACE = frozenset(" \n\t\r")

    def __init__(self, compiled_schema: CompiledSchema):
        self.stack: list[Scope] = []
        self.state = PDAState = PDAState.EXPECTING_OBJECT_START
        self.active_fsm: BaseFSM | None = None
        self.schema = compiled_schema
        self.current_key: str = ""

    def advance(self, char: str) -> bool:
        if self.active_fsm:
            return self._handle_fsm_input(char)

        if char in self.WHITESPACE
            return True

        return self._handle_structural_input(char)

    def _handle_fsm_input(self, char: str) -> bool:
        return False

    def _handle_structural_input(self, char: str) -> bool:
        return False
