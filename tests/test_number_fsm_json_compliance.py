"""
Tests for C3 fix: NumberFSM must enforce JSON number grammar.

Current bug (primitives.py:60-74):
- "-" alone is accepted (no digit consumed)
- "5." is accepted (no digit after dot)
- "007" is accepted (leading zeros)

Required fix:
- Track whether at least one digit has been consumed
- Only transition to TERMINAL if a digit was consumed
- After "0", reject additional digits (only ".", "e", "E", terminator allowed)
- After "-", require at least one digit before terminator

These tests will FAIL until the fix is applied, then PASS.
"""
import pytest

from src.automata import NumberFSM, JSONPushdownAutomaton, CompiledSchema


def _is_valid_number(s: str) -> bool:
    """
    Drive NumberFSM through string and check if it represents a valid JSON number.
    
    Returns True if:
    - All chars consumed successfully (advance() returns True)
    - FSM can reach TERMINAL state on a terminator char (e.g., ',')
    
    Returns False if:
    - Any char is rejected during consumption
    - FSM cannot reach TERMINAL on terminator (incomplete number)
    """
    fsm = NumberFSM()
    for char in s:
        if not fsm.advance(char):
            return False
    
    # Check if FSM is in a state that allows terminator → TERMINAL
    probe = fsm.clone()
    probe.advance(",")  # terminator
    return probe.is_terminal()


# ---------------------------------------------------------------------------
# Invalid JSON numbers (C3 bug cases)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("number", [
    "-",      # minus without digit
    "5.",     # trailing dot without fractional digit
    "00",     # leading zero
    "007",    # leading zeros
    "-00",    # minus + leading zeros
    ".",      # no integer part
    "-.",     # minus + dot, no digits
])
def test_number_fsm_rejects_invalid_json_numbers(number: str) -> None:
    """
    C3 fix verification: invalid JSON numbers must be rejected.
    
    These tests will FAIL with the current buggy implementation and
    PASS after the NumberFSM is fixed to enforce JSON number grammar.
    """
    assert not _is_valid_number(number), f"{number!r} should be invalid JSON"


# ---------------------------------------------------------------------------
# Valid JSON numbers (regression tests)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("number", [
    "0",       # zero
    "123",     # integer
    "-42",     # negative integer
    "3.14",    # float
    "-0.5",    # negative float
    "0.123",   # float starting with zero
    "100",     # integer with zeros
    "-100",    # negative integer with zeros
])
def test_number_fsm_accepts_valid_json_numbers(number: str) -> None:
    """
    Regression: valid JSON numbers must still be accepted after C3 fix.
    
    These tests should PASS both before and after the fix.
    """
    assert _is_valid_number(number), f"{number!r} should be valid JSON"


# ---------------------------------------------------------------------------
# PDA-level integration tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("json_str", [
    '{"a":-}',      # minus without digit
    '{"a":5.}',     # trailing dot
    '{"a":007}',    # leading zeros
])
def test_pda_rejects_invalid_numbers_in_json(json_str: str) -> None:
    """
    C3 fix verification: PDA must reject invalid numbers in JSON context.
    
    These tests will FAIL with current buggy implementation and PASS after fix.
    """
    schema: CompiledSchema = {"a": NumberFSM}
    pda = JSONPushdownAutomaton(schema)
    
    for char in json_str:
        if not pda.advance(char):
            return  # Rejected at some point — good!
    
    pytest.fail(f"PDA accepted invalid JSON number in: {json_str!r}")


@pytest.mark.parametrize("json_str", [
    '{"a":0}',
    '{"a":123}',
    '{"a":-42}',
    '{"a":3.14}',
    '{"a":-0.5}',
])
def test_pda_accepts_valid_numbers_in_json(json_str: str) -> None:
    """
    Regression: PDA must accept valid numbers in JSON context.
    """
    schema: CompiledSchema = {"a": NumberFSM}
    pda = JSONPushdownAutomaton(schema)
    
    for char in json_str:
        assert pda.advance(char), f"PDA rejected {char!r} in {json_str!r}"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_number_fsm_rejects_empty_string() -> None:
    """Empty string is not a valid number."""
    assert not _is_valid_number("")


def test_number_fsm_rejects_multiple_dots() -> None:
    """Multiple decimal points are invalid."""
    assert not _is_valid_number("1.2.3")


def test_number_fsm_accepts_zero_variants() -> None:
    """Zero in various forms is valid."""
    assert _is_valid_number("0")
    assert _is_valid_number("-0")
    assert _is_valid_number("0.0")
    assert _is_valid_number("-0.0")
