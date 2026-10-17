from dataclasses import dataclass, field
from enum import Enum, auto

from .primitives import BaseFSM, ExactMatchFSM
from .compiler import ArraySpec, CompiledSchema, ObjectSpec, ValueSpec


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


@dataclass
class _Frame:
    """
    One nesting level. OBJECT frames track which required keys remain and
    the spec of the key currently being valued; ARRAY frames track their
    element spec and whether the array is still empty.
    """
    scope: Scope
    fields: CompiledSchema | None = None
    remaining_required: set[str] = field(default_factory=set)
    current_spec: ValueSpec | None = None
    item_spec: ValueSpec | None = None
    just_opened: bool = False

    def clone(self) -> '_Frame':
        return _Frame(
            scope=self.scope,
            fields=self.fields,
            remaining_required=set(self.remaining_required),
            current_spec=self.current_spec,
            item_spec=self.item_spec,
            just_opened=self.just_opened,
        )


class JSONPushdownAutomaton:
    # Structural Grammar Definitions
    CHAR_OBJECT_START = frozenset("{")
    CHAR_ARRAY_START = frozenset("[")
    CHAR_ARRAY_CLOSE = frozenset("]")
    CHAR_KEY_QUOTE = frozenset('"')
    CHAR_COLON = frozenset(":")
    CHAR_COMMA = frozenset(",")
    CHAR_OBJECT_CLOSE = frozenset("}")
    CHARS_OBJECT_NEXT = frozenset({",", "}"})
    CHARS_ARRAY_NEXT = frozenset({",", "]"})
    CHARS_EMPTY: frozenset[str] = frozenset()
    WHITESPACE = frozenset(" \n\r\t")

    def __init__(self, compiled_schema: CompiledSchema) -> None:
        self.stack: list[_Frame] = []
        self.state = PDAState.EXPECTING_OBJECT_START
        self.active_fsm: BaseFSM | None = None
        self.schema = compiled_schema
        self.pending_fields: CompiledSchema | None = compiled_schema
        self.pending_item_spec: ValueSpec | None = None

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
                top = self.stack[-1]
                if char == '"':
                    return bool(top.remaining_required)
                if char == "}":
                    return not top.remaining_required
                return False

            case PDAState.EXPECTING_COLON:
                return char == ":"

            case PDAState.EXPECTING_ARRAY_START:
                return char == "["

            case PDAState.EXPECTING_VALUE:
                return self._value_accepts(char)

            case PDAState.EXPECTING_COMMA_OR_END:
                return self._comma_or_end_accepts(char)

            case _:
                return False

    def _value_accepts(self, char: str) -> bool:
        """
        Non-mutating mirror of _on_value() for the current value context,
        which is either an open array element or an object field.
        """
        top = self.stack[-1]
        spec = (
            top.item_spec if top.scope == Scope.ARRAY else top.current_spec
        )

        if top.scope == Scope.ARRAY and top.just_opened and char == "]":
            return True

        if spec is None:
            return False
        if isinstance(spec, ArraySpec):
            return char == "["
        if isinstance(spec, ObjectSpec):
            return char == "{"
        return spec().accepts_char(char)

    def _comma_or_end_accepts(self, char: str) -> bool:
        """
        Non-mutating mirror of _on_comma_or_end().
        The stack is guaranteed non-empty in EXPECTING_COMMA_OR_END.
        """
        top = self.stack[-1]
        if char == ',':
            if top.scope == Scope.OBJECT:
                return bool(top.remaining_required)
            return top.scope == Scope.ARRAY
        if char == '}':
            return top.scope == Scope.OBJECT and not top.remaining_required
        if char == ']':
            return top.scope == Scope.ARRAY
        return False

    def remaining_key_count(self) -> int:
        """
        Number of required keys still outstanding in the current object
        frame (0 when the current scope is an array or the stack is empty).
        Read-only helper used by tracing.
        """
        if self.stack and self.stack[-1].scope == Scope.OBJECT:
            return len(self.stack[-1].remaining_required)
        return 0

    def clone(self) -> 'JSONPushdownAutomaton':
        """
        Cheap copy: shares read-only/immutable fields (schema, pending
        dict), copies the frame stack and the active FSM. ~30x faster
        than copy.deepcopy.
        """
        new = JSONPushdownAutomaton.__new__(JSONPushdownAutomaton)
        new.stack = [frame.clone() for frame in self.stack]
        new.state = self.state
        new.active_fsm = (
            self.active_fsm.clone() if self.active_fsm else None
        )
        new.schema = self.schema
        new.pending_fields = self.pending_fields
        new.pending_item_spec = self.pending_item_spec
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
                matched_key = self.active_fsm.active_candidates[0].strip('"')
                top = self.stack[-1]
                top.remaining_required.discard(matched_key)
                top.current_spec = (
                    top.fields.get(matched_key) if top.fields else None
                )

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

            case PDAState.EXPECTING_ARRAY_START:
                return self._on_array_start(char)

            case PDAState.EXPECTING_VALUE:
                return self._on_value(char)

            case PDAState.EXPECTING_COMMA_OR_END:
                return self._on_comma_or_end(char)

            case _:
                return False

    def _on_object_start(self, char: str) -> bool:
        """
        Handles the start of a JSON object, pushing a frame onto the stack.
        Used for both the root object and nested objects.
        """
        if char != "{" or self.pending_fields is None:
            return False

        self.stack.append(_Frame(
            scope=Scope.OBJECT,
            fields=self.pending_fields,
            remaining_required=set(self.pending_fields.keys()),
        ))
        self.pending_fields = None
        self.state = PDAState.EXPECTING_KEY
        return True

    def _on_key(self, char: str) -> bool:
        """Handles the start of a JSON key, or the close of an empty object."""
        top = self.stack[-1]

        if char == '"':
            if not top.remaining_required:
                return False

            valid_quoted_keys = [f'"{k}"' for k in top.remaining_required]
            self.active_fsm = ExactMatchFSM(valid_quoted_keys)
            self.active_fsm.advance('"')
            self.state = PDAState.EXPECTING_COLON
            return True

        if char == '}' and not top.remaining_required:
            return self._close_scope()

        return False

    def _on_colon(self, char: str) -> bool:
        """Handles the colon separating a key from its value."""
        if char == ":":
            self.state = PDAState.EXPECTING_VALUE
            return True
        return False

    def _on_value(self, char: str) -> bool:
        """
        Routes the value generation based on the current context: array
        elements use the frame's element spec, object fields use the spec
        of the key just matched. Objects and arrays open new scopes.
        """
        top = self.stack[-1]
        spec = (
            top.item_spec if top.scope == Scope.ARRAY else top.current_spec
        )

        if top.scope == Scope.ARRAY and top.just_opened and char == "]":
            return self._on_comma_or_end(char)

        if spec is None:
            return False

        if top.scope == Scope.ARRAY:
            top.just_opened = False

        if isinstance(spec, ArraySpec):
            self.pending_item_spec = spec.item
            self.state = PDAState.EXPECTING_ARRAY_START
            return self._on_array_start(char)

        if isinstance(spec, ObjectSpec):
            self.pending_fields = dict(spec.fields)
            self.state = PDAState.EXPECTING_OBJECT_START
            return self._on_object_start(char)

        self.active_fsm = spec()
        self.state = PDAState.EXPECTING_COMMA_OR_END
        return self.active_fsm.advance(char)

    def _on_array_start(self, char: str) -> bool:
        """Opens a new array scope on '['."""
        if char != "[":
            return False
        self.stack.append(_Frame(
            scope=Scope.ARRAY,
            item_spec=self.pending_item_spec,
            just_opened=True,
        ))
        self.pending_item_spec = None
        self.state = PDAState.EXPECTING_VALUE
        return True

    def _close_scope(self) -> bool:
        """Pops the current frame, returning to the parent or TERMINAL."""
        self.stack.pop()
        if self.stack:
            self.state = PDAState.EXPECTING_COMMA_OR_END
        else:
            self.state = PDAState.TERMINAL
        return True

    def _on_comma_or_end(self, char: str) -> bool:
        """Handles structural continuation (,) or scope closures (}, ])."""
        top = self.stack[-1]

        if char == ',':
            if top.scope == Scope.OBJECT:
                if not top.remaining_required:
                    return False
                self.state = PDAState.EXPECTING_KEY
            else:
                top.just_opened = False
                self.state = PDAState.EXPECTING_VALUE
            return True

        if char == '}' and top.scope == Scope.OBJECT:
            if top.remaining_required:
                return False
            return self._close_scope()

        if char == ']' and top.scope == Scope.ARRAY:
            return self._close_scope()

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
                top = self.stack[-1]
                if top.remaining_required:
                    return self.CHAR_KEY_QUOTE
                return self.CHAR_OBJECT_CLOSE

            case PDAState.EXPECTING_COLON:
                return self.CHAR_COLON

            case PDAState.EXPECTING_ARRAY_START:
                return self.CHAR_ARRAY_START

            case PDAState.EXPECTING_VALUE:
                top = self.stack[-1]
                spec = (
                    top.item_spec
                    if top.scope == Scope.ARRAY
                    else top.current_spec
                )
                if spec is None:
                    return self.CHARS_EMPTY
                if isinstance(spec, ArraySpec):
                    allowed: set[str] = set(self.CHAR_ARRAY_START)
                elif isinstance(spec, ObjectSpec):
                    allowed = set(self.CHAR_OBJECT_START)
                else:
                    allowed = set(spec().allowed_characters())
                if top.scope == Scope.ARRAY and top.just_opened:
                    allowed |= self.CHAR_ARRAY_CLOSE
                return allowed

            case PDAState.EXPECTING_COMMA_OR_END:
                top = self.stack[-1]
                if top.scope == Scope.OBJECT:
                    if top.remaining_required:
                        return self.CHAR_COMMA
                    return self.CHAR_OBJECT_CLOSE
                return self.CHARS_ARRAY_NEXT

            case _:
                return self.CHARS_EMPTY

        return self.CHARS_EMPTY
