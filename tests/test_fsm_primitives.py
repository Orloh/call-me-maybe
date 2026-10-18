import pytest
from src.automata import (
    NumberFSM,
    ExactMatchFSM,
    StringLiteralFSM,
    NumberState,
    StringState
)


def test_number_fsm_valid_integer():
    fsm = NumberFSM()
    
    assert fsm.state == NumberState.START

    assert fsm.advance("4") is True
    assert fsm.state == NumberState.INTEGER_PART

    assert fsm.advance("2") is True
    assert fsm.state == NumberState.INTEGER_PART

    assert fsm.advance(",") is False
    assert fsm.is_terminal() is True


def test_number_fsm_valid_negative_float():
    fsm = NumberFSM()
    
    assert fsm.advance('-') is True
    assert fsm.state == NumberState.AFTER_MINUS
    
    assert fsm.advance('3') is True
    assert fsm.state == NumberState.INTEGER_PART
    
    assert fsm.advance('.') is True
    assert fsm.state == NumberState.AFTER_DOT
    
    assert fsm.advance('1') is True
    assert fsm.state == NumberState.FRACTIONAL_PART
    assert fsm.advance('4') is True
    
    assert fsm.advance('}') is False
    assert fsm.is_terminal() is True


def test_number_fsm_rejects_letters():
    fsm = NumberFSM()
    assert fsm.advance('a') is False
    assert fsm.state == NumberState.START
    
    fsm.advance('5')
    assert fsm.advance('x') is False


def test_number_fsm_rejects_multiple_decimals():
    fsm = NumberFSM()
    fsm.advance('3')
    fsm.advance('.')
    
    assert fsm.state == NumberState.AFTER_DOT
    
    assert fsm.advance('.') is False

    fsm.advance('3')
    assert fsm.state == NumberState.FRACTIONAL_PART
    assert fsm.advance('.') is False


def test_number_fsm_accepts_char_pruning():
    """
    Verifies that accepts_char reports exactly the characters that can
    extend a number (used by the DFS to prune trie edges).
    """
    fsm = NumberFSM()

    assert fsm.accepts_char('-')
    assert fsm.accepts_char('5')
    assert not fsm.accepts_char('.')
    assert not fsm.accepts_char(',')

    fsm.advance('5')
    assert fsm.accepts_char('.')
    assert not fsm.accepts_char(',')  # Terminators handled by terminates_on()
    assert not fsm.accepts_char('-')

    fsm.advance('.')
    assert fsm.accepts_char('7')
    assert not fsm.accepts_char(',')  # Terminators handled by terminates_on()
    assert not fsm.accepts_char('.')


def test_number_fsm_rejects_non_ascii_digits():
    """Only ASCII 0-9 are JSON digits; superscripts/Arabic-Indic are not."""
    for literal in ["²", "³", "¹", "٣", "０"]:
        fsm = NumberFSM()
        assert fsm.advance(literal) is False, literal
        assert fsm.accepts_char(literal) is False, literal

    # And they cannot extend a valid integer part either.
    fsm = NumberFSM()
    fsm.advance("1")
    assert fsm.advance("²") is False
    assert fsm.accepts_char("²") is False


def test_number_fsm_accepts_carriage_return_terminator():
    """`\\r` is legal JSON whitespace and must terminate a number."""
    fsm = NumberFSM()
    for char in "12":
        assert fsm.advance(char)

    assert fsm.terminates_on("\r") is True

    probe = fsm.clone()
    assert probe.advance("\r") is False
    assert probe.is_terminal() is True


def test_string_fsm_rejects_non_ascii_hex():
    """`\\u` escapes require ASCII hex digits."""
    fsm = StringLiteralFSM()
    for char in '"\\u00':
        assert fsm.advance(char)

    assert fsm.accepts_char("²") is False
    assert fsm.advance("²") is False


def test_number_fsm_accepts_char_mirror():
    """accepts_char must mirror advance()'s True/False for every char."""
    fsm = NumberFSM()
    probes = ['-', '5', '.', ',', 'x', '}', ' ']
    for c in probes:
        probe = fsm.clone()
        consumed = probe.advance(c)
        assert fsm.accepts_char(c) == consumed, f"accepts_char({c!r}) diverged"
        if not consumed:
            assert probe.is_terminal() == fsm.terminates_on(c), f"terminates_on({c!r}) diverged"

    # After advancing '5', test INTEGER_PART state
    fsm.advance('5')
    for c in probes:
        probe = fsm.clone()
        consumed = probe.advance(c)
        assert fsm.accepts_char(c) == consumed
        if not consumed:
            assert probe.is_terminal() == fsm.terminates_on(c)

    # After advancing '.', test FRACTIONAL_PART state
    fsm.advance('.')
    for c in probes:
        probe = fsm.clone()
        consumed = probe.advance(c)
        assert fsm.accepts_char(c) == consumed
        if not consumed:
            assert probe.is_terminal() == fsm.terminates_on(c)


def test_string_fsm_valid_simple_string():
    fsm = StringLiteralFSM()
    
    assert fsm.state == StringState.EXPECTING_OPEN_QUOTE
    assert fsm.parsed_value == ""

    assert fsm.advance("a") is False
    
    assert fsm.advance('"') is True
    assert fsm.state == StringState.INSIDE_STRING
    assert fsm.parsed_value  == '"'

    assert fsm.advance("h") is True
    assert fsm.advance("i") is True
    assert fsm.parsed_value == '"hi'

    assert fsm.advance('"') is True
    assert fsm.state == StringState.TERMINAL
    assert fsm.is_terminal() is True
    assert fsm.parsed_value == '"hi"'



def test_string_fsm_escape_sequence():
    fsm = StringLiteralFSM()
    fsm.advance('"')
    
    assert fsm.advance('\\') is True
    assert fsm.state == StringState.ESCAPE_SEQUENCE
    assert fsm.parsed_value == '"\\'
    
    assert fsm.advance('x') is False
    
    assert fsm.advance('"') is True
    assert fsm.state == StringState.INSIDE_STRING
    assert fsm.parsed_value == '"\\"'
    
    assert fsm.is_terminal() is False
    
    assert fsm.advance('"') is True
    assert fsm.is_terminal() is True
    assert fsm.parsed_value == '"\\""'


def test_string_fsm_rejects_raw_newlines():
    fsm = StringLiteralFSM()
    fsm.advance('"')
    
    assert fsm.advance('\n') is False


def test_exact_match_fsm_booleans():
    fsm = ExactMatchFSM(["true", "false"])

    assert fsm.accepts_char("t")
    assert fsm.accepts_char("f")
    assert not fsm.accepts_char("x")

    assert fsm.advance("t") is True
    assert fsm.accepts_char("r")
    assert not fsm.accepts_char("u")
    assert fsm.advance("r") is True
    assert fsm.advance("u") is True
    assert fsm.advance("e") is True
    
    assert fsm.is_terminal() is True
    
    assert fsm.advance("x") is False


def test_exact_match_fsm_accepts_char_mirror():
    """accepts_char must mirror advance() for ExactMatchFSM."""
    fsm = ExactMatchFSM(["true", "false"])
    probes = ['t', 'f', 'r', 'u', 'e', 'x', '"']
    for c in probes:
        probe = fsm.clone()
        consumed = probe.advance(c)
        assert fsm.accepts_char(c) == consumed
        if not consumed:
            assert probe.is_terminal() == fsm.terminates_on(c)


def test_exact_match_fsm_overlapping_enums():
    fsm = ExactMatchFSM(['"add"', '"add_numbers"'])
    
    assert fsm.advance('"') is True
    assert fsm.advance('a') is True
    assert fsm.advance('d') is True
    assert fsm.advance('d') is True
    
    assert fsm.accepts_char('"')
    assert fsm.accepts_char('_')
    assert not fsm.accepts_char('a')

    assert fsm.advance('"') is True
    assert fsm.is_terminal() is True

def test_exact_match_fsm_rejects_invalid_start():
    fsm = ExactMatchFSM(["null"])
    assert fsm.advance("f") is False
    assert fsm.is_terminal() is False
