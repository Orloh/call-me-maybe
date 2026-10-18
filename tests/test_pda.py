import pytest

from src.automata import (
    JSONPushdownAutomaton,
    PDAState,
    Scope,
    NumberFSM,
    StringLiteralFSM,
    ExactMatchFSM,
)


@pytest.fixture
def mock_schema():
    """
    Simulates the output of our SchemaCompiler.
    """
    return {
        "location": lambda: StringLiteralFSM(),
        "days": lambda: NumberFSM(),
    }


def feed_string(pda: JSONPushdownAutomaton, string: str) -> bool:
    """
    Helper function to feed a whole string to the PDA char-by-char.
    If the PDA rejects any character, it returns False instantly.
    """
    for char in string:
        if not pda.advance(char):
            return False
    return True


def test_pda_initialization(mock_schema):
    """
    Proves the PDA starts in the correct state and only allows an opening brace.
    """
    pda = JSONPushdownAutomaton(mock_schema)
    assert pda.state == PDAState.EXPECTING_OBJECT_START
    assert pda.accepts_char("{")
    assert len(pda.stack) == 0
    assert pda.pending_fields == mock_schema


def test_on_object_start_logic(mock_schema):
    pda = JSONPushdownAutomaton(mock_schema)
    pda.state = PDAState.EXPECTING_OBJECT_START

    assert pda._on_object_start("{") is True
    assert pda.state == PDAState.EXPECTING_KEY
    assert pda.stack[-1].scope == Scope.OBJECT
    assert pda.stack[-1].remaining_required == {"location", "days"}
    assert pda._on_object_start("x") is False


def test_on_key_logic(mock_schema):
    pda = JSONPushdownAutomaton(mock_schema)
    pda.advance("{")
    pda.state = PDAState.EXPECTING_KEY

    assert pda._on_key('"') is True
    assert pda.state == PDAState.EXPECTING_COLON
    assert isinstance(pda.active_fsm, ExactMatchFSM)
    assert set(pda.active_fsm.active_candidates) == {'"location"', '"days"'}


def test_on_colon_logic(mock_schema):
    pda = JSONPushdownAutomaton(mock_schema)
    pda.state = PDAState.EXPECTING_COLON

    assert pda._on_colon(':') is True
    assert pda.state == PDAState.EXPECTING_VALUE
    assert pda._on_colon('x') is False


def test_on_value_flat_routing(mock_schema):
    pda = JSONPushdownAutomaton(mock_schema)
    for char in '{"days":':
        assert pda.advance(char)

    # "days" expects a NumberFSM based on mock_schema
    assert pda.advance('5') is True
    assert isinstance(pda.active_fsm, NumberFSM)
    assert pda.state == PDAState.EXPECTING_COMMA_OR_END


def test_on_comma_or_end_logic(mock_schema):
    pda = JSONPushdownAutomaton(mock_schema)
    for char in '{"location":"Madrid"':
        assert pda.advance(char)

    assert pda.advance(',') is True
    assert pda.state == PDAState.EXPECTING_KEY

    for char in '"days":5':
        assert pda.advance(char)

    assert pda.advance('}') is True
    assert pda.stack == []
    assert pda.state == PDAState.TERMINAL


def test_pda_accepts_curly_bracket(mock_schema):
    """
    Proves the PDA starts in the correct state and accepts an opening brace.
    """
    pda = JSONPushdownAutomaton(mock_schema)

    assert pda.state == PDAState.EXPECTING_OBJECT_START
    pda.advance("{")
    assert pda.state == PDAState.EXPECTING_KEY
    assert pda.accepts_char('"')


def test_pda_valid_phase2_json(mock_schema):
    """
    Proves the PDA can parse the complete flat parameters object.
    """
    pda = JSONPushdownAutomaton(mock_schema)

    valid_json = '{\n "location": "Madrid", \n "days": 5\n}'

    assert feed_string(pda, valid_json) is True
    assert pda.state == PDAState.TERMINAL
    assert pda.stack == []


def test_pda_rejects_hallucinated_keys(mock_schema):
    """
    Proves that if the LLM tries to generate a key not in the schema,
    the branch is instantly pruned.
    """
    pda = JSONPushdownAutomaton(mock_schema)

    # 'hallucination' is not in our mock_schema
    invalid_json = '{"hallucination": "fake_data"}'

    assert feed_string(pda, invalid_json) is False


def test_pda_prevents_premature_closure(mock_schema):
    """
    Proves the LLM cannot close the JSON if required keys are missing.
    """
    pda = JSONPushdownAutomaton(mock_schema)

    invalid_json = '{"location": "Madrid"}'
    assert feed_string(pda, invalid_json) is False


def test_pda_rejects_invalid_types(mock_schema):
    """
    Proves that the PDA strictly enforces the FSM data types.
    """
    pda = JSONPushdownAutomaton(mock_schema)

    # 'days' expects a NumberFSM. We are giving it a string.
    invalid_json = '{"location": "Madrid", "days": "five"}'

    assert feed_string(pda, invalid_json) is False
