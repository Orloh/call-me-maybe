from enum import Enum, auto
from .primitives import BaseFSM, ExactMatchFSM
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

    # Shared states
    EXPECTING_VALUE = auto()
    EXPECTING_COMMA_OR_END = auto()
    TERMINAL = auto()


class JSONPushdownAutomaton:
    # Structural Grammar Definitions
    CHAR_OBJECT_START = frozenset("{")
    CHAR_ARRAY_START = frozenset("[")
    CHAR_KEY_QUOTE = frozenset('"')
    CHAR_COLON = frozenset(":")
    CHAR_COMMA = frozenset(",")
    CHAR_OBJECT_CLOSE = frozenset("}")
    CHARS_OBJECT_NEXT = frozenset({",", "}"})
    CHARS_ARRAY_NEXT = frozenset({",", "]"})
    CHARS_EMPTY: frozenset[str] = frozenset()
    WHITESPACE = frozenset(" \n\t\r")

    def __init__(self, compiled_schema: CompiledSchema) -> None:
        self.stack: list[Scope] = []
        self.state = PDAState.EXPECTING_OBJECT_START
        self.active_fsm: BaseFSM | None = None
        self.schema = compiled_schema
        self.current_key: str = ""
        self.remaining_keys: set[str] = set(self.schema.keys())

    def advance(self, char: str) -> bool:
        """
        Master router.
        If active_fsm is alive, feed the char to the FSM.
        If not, the PDA processes the structural char ({, }, [, ], :, ,)
        """
        if self.active_fsm:
            return self._handle_fsm_input(char)

        if char in self.WHITESPACE:
            return True

        return self._handle_structural_input(char)

    def accepts_char(self, char: str) -> bool:
        """
        Non-mutating mirror of advance(): returns True iff advance(char)
        would accept the char in the CURRENT state. The DFS uses this to
        pre-filter trie edges before paying for a clone.

        Invariant: pda.accepts_char(c) == pda.clone().advance(c)
        """
        if self.active_fsm:
            if self.active_fsm.accepts_char(char):
                return True
            if self.active_fsm.terminates_on(char):
                # Mirror of the FSM hand-off: the clone clears the FSM
                # and re-dispatches the char to whitespace/structural.
                return (
                    char in self.WHITESPACE
                    or self._structural_accepts(char)
                )
            return False

        if char in self.WHITESPACE:
            return True

        return self._structural_accepts(char)

    def _structural_accepts(self, char: str) -> bool:
        """
        Non-mutating mirror of _handle_structural_input().
        """
        match self.state:
            case PDAState.EXPECTING_OBJECT_START:
                return char == "{"

            case PDAState.EXPECTING_KEY:
                return char == '"' and bool(self.remaining_keys)

            case PDAState.EXPECTING_COLON:
                return char == ":"

            case PDAState.EXPECTING_VALUE:
                # _on_value routes through the fresh FSM for current_key;
                # on a fresh FSM, advance() ≡ accepts_char().
                fsm_factory = self.schema.get(self.current_key)
                if fsm_factory is None:
                    return False
                return fsm_factory().accepts_char(char)

            case PDAState.EXPECTING_COMMA_OR_END:
                return self._comma_or_end_accepts(char)

            case _:
                return False

    def _comma_or_end_accepts(self, char: str) -> bool:
        """
        Non-mutating mirror of _on_comma_or_end().
        The stack is guaranteed non-empty in EXPECTING_COMMA_OR_END.
        """
        top = self.stack[-1]
        if char == ',':
            if top == Scope.OBJECT:
                return bool(self.remaining_keys)
            return top == Scope.ARRAY
        if char == '}':
            return top == Scope.OBJECT and not self.remaining_keys
        if char == ']':
            return top == Scope.ARRAY
        return False

    def clone(self) -> 'JSONPushdownAutomaton':
        """
        Cheap copy: shares read-only/immutable fields (schema, state,
        current_key), copies only what mutations touch (stack,
        remaining_keys, active_fsm). ~30x faster than copy.deepcopy.
        """
        new = JSONPushdownAutomaton.__new__(JSONPushdownAutomaton)
        new.stack = self.stack.copy()
        new.state = self.state
        new.active_fsm = (
            self.active_fsm.clone() if self.active_fsm else None
        )
        new.schema = self.schema
        new.current_key = self.current_key
        new.remaining_keys = set(self.remaining_keys)
        return new

    def _handle_fsm_input(self, char: str) -> bool:
        if not self.active_fsm:
            return False

        if self.active_fsm.advance(char):
            return True

        if self.active_fsm.is_terminal():
            if (
                self.state == PDAState.EXPECTING_COLON
                and isinstance(self.active_fsm, ExactMatchFSM)
            ):
                matched_quoted_key = self.active_fsm.active_candidates[0]
                self.current_key = matched_quoted_key.strip('"')

                if self.current_key in self.remaining_keys:
                    self.remaining_keys.remove(self.current_key)

            self.active_fsm = None
            return self.advance(char)

        return False

    def _handle_structural_input(self, char: str) -> bool:
        match self.state:
            case PDAState.EXPECTING_OBJECT_START:
                return self._on_object_start(char)

            case PDAState.EXPECTING_KEY:
                return self._on_key(char)

            case PDAState.EXPECTING_COLON:
                return self._on_colon(char)

            case PDAState.EXPECTING_VALUE:
                return self._on_value(char)

            case PDAState.EXPECTING_COMMA_OR_END:
                return self._on_comma_or_end(char)

            case _:
                return False

    def _on_object_start(self, char: str) -> bool:
        """
        Handles the start of a JSON object, pushing it onto the LIFO stack.
        """
        if char == "{":
            self.stack.append(Scope.OBJECT)
            self.state = PDAState.EXPECTING_KEY
            return True
        return False

    def _on_key(self, char: str) -> bool:
        """Handles the start of JSON key."""
        if char == '"':
            if not self.remaining_keys:
                return False

            valid_quoted_keys = [f'"{k}"' for k in self.remaining_keys]
            self.active_fsm = ExactMatchFSM(valid_quoted_keys)
            self.active_fsm.advance('"')
            self.state = PDAState.EXPECTING_COLON
            return True
        return False

    def _on_colon(self, char: str) -> bool:
        """Handles the colon separationg a key from its value."""
        if char == ":":
            self.state = PDAState.EXPECTING_VALUE
            return True
        return False

    def _on_value(self, char: str) -> bool:
        """
        Routes the value generation based on the current key.
        Handles nested 'parameters' object structurally, and delegates
        flat values to their respecctive FSMs based on the CompiledSchema.
        """
        try:
            fsm_factory = self.schema[self.current_key]
            self.active_fsm = fsm_factory()
            self.state = PDAState.EXPECTING_COMMA_OR_END
            return self.active_fsm.advance(char)

        except KeyError:
            return False

    def _on_comma_or_end(self, char: str) -> bool:
        """Handles structural continuation (,) or scope closures (}, ])."""
        if char == ',':
            if self.stack[-1] == Scope.OBJECT:
                if not self.remaining_keys:
                    return False
                self.state = PDAState.EXPECTING_KEY
            elif self.stack[-1] == Scope.ARRAY:
                self.state = PDAState.EXPECTING_VALUE
            return True

        elif char == '}' and self.stack[-1] == Scope.OBJECT:
            if self.remaining_keys:
                return False

            self.stack.pop()
            if self.stack:
                self.state = PDAState.EXPECTING_COMMA_OR_END
            else:
                self.state = PDAState.TERMINAL
            return True

        elif char == ']' and self.stack[-1] == Scope.ARRAY:
            self.stack.pop()
            if self.stack:
                self.state = PDAState.EXPECTING_COMMA_OR_END
            else:
                self.state = PDAState.TERMINAL
            return True

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
                    if not self.remaining_keys:
                        return self.CHAR_OBJECT_CLOSE
                    else:
                        return self.CHAR_COMMA

                elif self.stack[-1] == Scope.ARRAY:
                    return self.CHARS_ARRAY_NEXT

            case _:
                return self.CHARS_EMPTY

        return self.CHARS_EMPTY
