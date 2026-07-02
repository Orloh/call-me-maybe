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
    assert fsm.state == NumberState.INTEGER_PART
    
    assert fsm.advance('3') is True
    
    assert fsm.advance('.') is True
    assert fsm.state == NumberState.FRACTIONAL_PART
    
    assert fsm.advance('1') is True
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
    
    assert fsm.state == NumberState.FRACTIONAL_PART
    
    assert fsm.advance('.') is False

    fsm.advance('3')
    assert fsm.state == NumberState.FRACTIONAL_PART
    assert fsm.advance('.') is False


def test_number_fsm_allowed_characters_pruning():
    """
    Verifies that the FSM correctly reports allowed characters 
    to the Trie for aggressive token pruning.
    """
    fsm = NumberFSM()
    
    allowed_start = fsm.allowed_characters()
    assert '-' in allowed_start
    assert '5' in allowed_start
    assert '.' not in allowed_start
    assert ',' not in allowed_start
    
    fsm.advance('5')
    allowed_int = fsm.allowed_characters()
    assert '.' in allowed_int
    assert ',' in allowed_int
    assert '-' not in allowed_int
    
    fsm.advance('.')
    allowed_frac = fsm.allowed_characters()
    assert '7' in allowed_frac
    assert ',' in allowed_frac
    assert '.' not in allowed_frac


def test_string_fsm_valid_simple_string():
    fsm = StringLiteralFSM()
    
    assert fsm.state == StringState.EXPECTING_OPEN_QUOTE

    assert fsm.advance("a") is False
    
    assert fsm.advance('"') is True
    assert fsm.state == StringState.INSIDE_STRING

    assert fsm.advance("h") is True
    assert fsm.advance("i") is True

    assert fsm.advance('"') is True
    assert fsm.is_terminal() is True

    assert fsm.state == StringState.TERMINAL


def test_string_fsm_escape_sequence():
    fsm = StringLiteralFSM()
    fsm.advance('"')
    
    assert fsm.advance('\\') is True
    assert fsm.state == StringState.ESCAPE_SEQUENCE
    
    assert fsm.advance('x') is False
    
    assert fsm.advance('"') is True
    assert fsm.state == StringState.INSIDE_STRING
    
    assert fsm.is_terminal() is False
    
    assert fsm.advance('"') is True
    assert fsm.is_terminal() is True


def test_string_fsm_rejects_raw_newlines():
    fsm = StringLiteralFSM()
    fsm.advance('"')
    
    assert fsm.advance('\n') is False


def test_exact_match_fsm_booleans():
    fsm = ExactMatchFSM(["true", "false"])

    assert fsm.allowed_characters() == {"t", "f"}
    
    assert fsm.advance("t") is True
    assert fsm.allowed_characters() == {"r"}
    assert fsm.advance("r") is True
    assert fsm.advance("u") is True
    assert fsm.advance("e") is True
    
    assert fsm.is_terminal() is True
    
    assert fsm.advance("x") is False


def test_exact_match_fsm_overlapping_enums():
    fsm = ExactMatchFSM(['"add"', '"add_numbers"'])
    
    assert fsm.advance('"') is True
    assert fsm.advance('a') is True
    assert fsm.advance('d') is True
    assert fsm.advance('d') is True
    
    assert fsm.allowed_characters() == {'"', '_'}
    
    assert fsm.advance('"') is True
    assert fsm.is_terminal() is True

def test_exact_match_fsm_rejects_invalid_start():
    fsm = ExactMatchFSM(["null"])
    assert fsm.advance("f") is False
    assert fsm.is_terminal() is False
