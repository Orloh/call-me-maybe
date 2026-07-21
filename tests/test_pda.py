import pytest

from src.automata import JSONPushdownAutomaton, PDAState, Scope, NumberFSM, StringLiteralFSM, ExactMatchFSM
from src.automata.primitives import BaseFSM


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
    assert pda.allowed_characters() == frozenset("{")
    assert len(pda.stack) == 0
    assert pda.remaining_keys == {"location", "days"}


def test_on_object_start_logic(mock_schema):
    pda = JSONPushdownAutomaton(mock_schema)
    pda.state = PDAState.EXPECTING_OBJECT_START
    
    assert pda._on_object_start("{") is True
    assert pda.state == PDAState.EXPECTING_KEY
    assert pda.stack[-1] == Scope.OBJECT
    assert pda._on_object_start("x") is False


def test_on_key_logic(mock_schema):
    pda = JSONPushdownAutomaton(mock_schema)
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
    pda.state = PDAState.EXPECTING_VALUE
    pda.current_key = "days"
    
    # "days" expects a NumberFSM based on mock_schema
    assert pda._on_value('5') is True
    assert isinstance(pda.active_fsm, NumberFSM)
    assert pda.state == PDAState.EXPECTING_COMMA_OR_END


def test_on_comma_or_end_logic(mock_schema):
    pda = JSONPushdownAutomaton(mock_schema)
    pda.state = PDAState.EXPECTING_COMMA_OR_END
    pda.stack.append(Scope.OBJECT)
    
    pda.remaining_keys = {"location"}
    assert pda._on_comma_or_end(',') is True
    assert pda.state == PDAState.EXPECTING_KEY
 
    pda.state = PDAState.EXPECTING_COMMA_OR_END
    pda.remaining_keys = set()
    assert pda._on_comma_or_end('}') is True
    assert len(pda.stack) == 0
    assert pda.state == PDAState.TERMINAL


def test_pda_accepts_curly_bracket(mock_schema):
    """
    Proves the PDA starts in the correct state and accepts an opening brace.
    """
    pda = JSONPushdownAutomaton(mock_schema)
    
    assert pda.state == PDAState.EXPECTING_OBJECT_START
    pda.advance("{")
    assert pda.state == PDAState.EXPECTING_KEY
    assert pda.allowed_characters() == frozenset('"')


def test_pda_valid_phase2_json(mock_schema):
    """
    Proves the PDA can parse the complete flat parameters object.
    """
    pda = JSONPushdownAutomaton(mock_schema)

    valid_json = '{\n "location": "Madrid", \n "days": 5\n}'

    assert feed_string(pda, valid_json) is True
    assert pda.state == PDAState.TERMINAL
    assert len(pda.remaining_keys) == 0


def test_pda_rejects_hallucinated_keys(mock_schema):
    """
    Proves that if the LLM tries to generate a key not in the schema,
    the branch is instantly pruned.
    """
    pda = JSONPushdownAutomaton(mock_schema)
    
    # 'hallucination' is not in our mock_schema
    invalid_json = '{"hallucination": "fake_data"}'
    
    # The PDA should return False the moment it evaluates the colon
    # and tries to look up the key.
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
    
    # It will fail exactly on the quote mark '"' because NumberFSM doesn't allow it.
    assert feed_string(pda, invalid_json) is False
