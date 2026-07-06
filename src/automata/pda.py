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
        """
        Master router.
        If active_fsm is alive, feed the char to the FSM.
        If not, the PDA processes the structural char ({, }, [, ], :, ,)
        """
        if self.active_fsm:
            return self._handle_fsm_input(char)

        if char in self.WHITESPACE
            return True

        return self._handle_structural_input(char)

    def _handle_fsm_input(self, char: str) -> bool:
        if not self.active_fsm:
            return False

        if self.active_fsm.advance(char):
            return True

        if self.active_fsm.is_terminal():
            if self.state == PDAState.EXPECTING_COLON and isinstance(self.active_fsm, StringLiteralFSM):
                self.current_key = self.active_fsm.parsed_value.strip('"')

            self.active_fsm = None

            return self._handle_structural_input(char)

        return False


    def _handle_structural_input(self, char: str) -> bool:
        return False
        
    def allowed_characters(self) -> frozenset[str] | set[str]:
        """
        If active_fsm is alive, return the FSM's allowed chars.
        Otherwise, return the PDA's allowed structural chars.
        """
        if self.active_fsm:
            return self.active_fsm.allowed_characters()

        match self.state:
            case PDAState.TERMINAL:
                return self.CHARS_EMPTY

            case PDAState.EXPECTING_OBJECT_START:
                return self.CHAR_OBJECT_START 

            case PDAState.EXPECTING_KEY:
                return self.CHAR_KEY_QUOTE

            case PDAState.EXPECTING_COLON:
                return self.CHAR_COLON

            case PDAState.EXPECTING_ARRAY_START:
                return self.CHAR_ARRAY_START

            case PDAState.EXPECTING_COMMA_OR_END:
                if self.stack[-1] == Scope.OBJECT:
                    return self.CHARS_OBJECT_NEXT
                elif self.stack[-1] == Scope.ARRAY:
                    return self.CHARS_ARRAY_NEXT

            case _:
                return self.CHARS_EMPTY

        return self.CHARS_EMPTY
