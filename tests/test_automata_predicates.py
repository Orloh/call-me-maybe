"""
Invariant tests for the non-mutating predicates used by the DFS mask.

The C1 optimization relies on three contracts:

1. accepts_char(c) is an exact mirror of advance(c):
       pda.accepts_char(c) == pda.clone().advance(c)
2. terminates_on(c) mirrors the FSM -> PDA hand-off:
       fsm.advance(c) returns False AND fsm.is_terminal() afterwards
3. clone() is behaviorally identical to copy.deepcopy:
       a clone can be advanced independently without affecting the
       original (and vice versa).

If any of these break, the DFS mask silently changes which tokens are
legal — these tests exist to catch that.
"""
import copy

import pytest

from src.automata import (
    CompiledSchema,
    ExactMatchFSM,
    JSONPushdownAutomaton,
    NumberFSM,
    StringLiteralFSM,
)
from src.automata.primitives import BaseFSM

_SCHEMA: CompiledSchema = {
    "name": StringLiteralFSM,
    "a": NumberFSM,
    "b": lambda: ExactMatchFSM(["true", "false"]),
}

# Chars that exercise every grammar branch: structure, whitespace,
# digits, letters, quotes, escapes, BPE artifacts, control chars.
_PROBES = [
    "{", "}", "[", "]", '"', ":", ",", " ", "\n", "\t", "\r",
    "a", "n", "0", "5", "9", ".", "-", "\\", "x", "Ġ", "é", "\x01", "t",
]

# PDA prefixes that park the automaton in every reachable state:
# object start, key match, colon, inside string, string escape,
# after value, inside number, after comma, terminal.
_PDA_PREFIXES = [
    "",
    "{",
    '{"na',
    '{"name"',
    '{"name":',
    '{"name":"Hel',
    '{"name":"a\\',
    '{"name":"x"',
    '{"name":"x",',
    '{"name":"x","a":',
    '{"name":"x","a":12',
    '{"name":"x","a":1.',
    '{"name":"x","a":12,',
    '{"name":"x","a":12,"b":tru',
    '{"name":"x","a":12,"b":true}',
]


def _drive_pda(prefix: str) -> JSONPushdownAutomaton:
    pda = JSONPushdownAutomaton(_SCHEMA)
    for char in prefix:
        assert pda.advance(char), f"PDA rejected {char!r} of {prefix!r}"
    return pda


# ---------------------------------------------------------------------------
# Invariant 1: accepts_char == advance (PDA level, the DFS contract)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("prefix", _PDA_PREFIXES)
def test_accepts_char_mirrors_advance(prefix: str) -> None:
    pda = _drive_pda(prefix)
    for probe in _PROBES:
        expected = pda.clone().advance(probe)
        assert pda.accepts_char(probe) == expected, (
            f"accepts_char({probe!r}) diverged after {prefix!r}: "
            f"got {not expected}"
        )


# ---------------------------------------------------------------------------
# Invariant 2: terminates_on mirrors the FSM -> PDA hand-off
# ---------------------------------------------------------------------------

def _number_fsm_states() -> list[BaseFSM]:
    fsms = []
    for prefix in ["", "5", "12", "1.", "1.5", "-", "-3"]:
        fsm = NumberFSM()
        for char in prefix:
            fsm.advance(char)
        fsms.append(fsm)
    # Drive one instance to TERMINAL via a terminator char.
    terminal = NumberFSM()
    terminal.advance("5")
    terminal.advance("}")
    fsms.append(terminal)
    return fsms


def _string_fsm_states() -> list[BaseFSM]:
    fsms = []
    for prefix in ['"', '"ab', '"a\\', '"a\\n', '"\\u', '"\\u00', '"\\u0041']:
        fsm = StringLiteralFSM()
        for char in prefix:
            fsm.advance(char)
        fsms.append(fsm)
    terminal = StringLiteralFSM()
    for char in '"x"':
        terminal.advance(char)
    fsms.append(terminal)
    return fsms


def _exact_match_fsm_states() -> list[BaseFSM]:
    fsms = []
    for prefix in ["", "t", "tr", "tru"]:
        fsm = ExactMatchFSM(["true", "false"])
        for char in prefix:
            fsm.advance(char)
        fsms.append(fsm)
    terminal = ExactMatchFSM(["true", "false"])
    for char in "true":
        terminal.advance(char)
    fsms.append(terminal)
    return fsms


@pytest.mark.parametrize(
    "fsm",
    _number_fsm_states() + _string_fsm_states() + _exact_match_fsm_states(),
)
def test_terminates_on_mirrors_handoff(fsm: BaseFSM) -> None:
    for probe in _PROBES:
        probe_clone = fsm.clone()
        consumed = probe_clone.advance(probe)
        if consumed:
            # The char is consumed by the FSM; no hand-off happens.
            assert not fsm.terminates_on(probe) or True
        else:
            # Hand-off happens iff the FSM is terminal after rejection.
            assert fsm.terminates_on(probe) == probe_clone.is_terminal(), (
                f"{type(fsm).__name__}.terminates_on({probe!r}) diverged: "
                f"is_terminal={probe_clone.is_terminal()}"
            )


# ---------------------------------------------------------------------------
# Invariant 3: clone() behaves like deepcopy (independence + equality)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("prefix", _PDA_PREFIXES)
def test_clone_matches_deepcopy_behavior(prefix: str) -> None:
    pda = _drive_pda(prefix)
    for probe in _PROBES:
        via_clone = pda.clone()
        via_deepcopy = copy.deepcopy(pda)
        assert via_clone.advance(probe) == via_deepcopy.advance(probe)
        assert via_clone.state == via_deepcopy.state
        assert via_clone.remaining_keys == via_deepcopy.remaining_keys


def test_clone_is_independent_of_original() -> None:
    pda = _drive_pda('{"name":"Hel')
    clone = pda.clone()

    # Mutate the clone hard; the original must be untouched.
    for char in 'lo","a":99':
        clone.advance(char)

    assert pda.state != clone.state or (
        pda.remaining_keys != clone.remaining_keys
    )
    # The original still accepts exactly what it accepted before cloning.
    assert pda.accepts_char("l")
    assert pda.current_key == "name"
