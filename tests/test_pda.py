import pytest

from src.automata import JSONPushdownAutomaton, PDAState, NumberFSM, StringLiteralFSM, ExactMatchFSM
from src.automata.primitives import BaseFSM

@pytest.fixture
def mock_schema():
    """
    Simulates the output of our SchemaCompiler.
    """
    return {
        "name": lambda: ExactMatchFSM(['"get_weather"']),
        "location": lambda: StringLiteralFSM(),
        "days": lambda: NumberFSM(),
    }

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
    assert isinstance(pda.active_fsm, StringLiteralFSM)
    assert pda._on_key('x') is False

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
    
    # Test continuing the object with a comma
    assert pda._on_comma_or_end(',') is True
    assert pda.state == PDAState.EXPECTING_KEY
    
    # Test closing the object with a brace
    pda.state = PDAState.EXPECTING_COMMA_OR_END
    assert pda._on_comma_or_end('}') is True
    assert len(pda.stack) == 0
    assert pda.state == PDAState.TERMINAL
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


def test_pda_accepts_curly_bracket(mock_schema):
    """
    Proves the PDA starts in the correct state and accepts an opening brace.
    """
    pda = JSONPushdownAutomaton(mock_schema)
    
    assert pda.state == PDAState.EXPECTING_OBJECT_START
    pda.advance("{")
    assert pda.state == PDAState.EXPECTING_KEY
    assert pda.allowed_characters() == frozenset('"')
    

def test_pda_valid_flat_json(mock_schema):
    """
    Proves the PDA can parse a flat JSON object, ignoring whitespace,
    and handling the FSM handoff correctly.
    """
    pda = JSONPushdownAutomaton(mock_schema)
    
    valid_json = '{\n  "prompt": "What is the weather?", \n  "name": "get_weather"\n}'
    
    assert feed_string(pda, valid_json) is True
    assert pda.state == PDAState.TERMINAL
    assert len(pda.stack) == 0

def test_pda_valid_nested_parameters(mock_schema):
    """
    Proves the PDA stack correctly handles the nested "parameters" object.
    """
    pda = JSONPushdownAutomaton(mock_schema)
    
    # This is the exact structure your LLM will be forced to generate
    valid_json = '{"name": "get_weather", "parameters": {"location": "Madrid", "days": 5}}'
    
    assert feed_string(pda, valid_json) is True
    assert pda.state == PDAState.TERMINAL
    
    # The stack should have pushed the nested object, and successfully popped it
    assert len(pda.stack) == 0

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

def test_pda_rejects_invalid_types(mock_schema):
    """
    Proves that the PDA strictly enforces the FSM data types.
    """
    pda = JSONPushdownAutomaton(mock_schema)
    
    # 'days' expects a NumberFSM. We are giving it a string.
    invalid_json = '{"days": "five"}'
    
    # It will fail exactly on the quote mark '"' because NumberFSM doesn't allow it.
    assert feed_string(pda, invalid_json) is False

def test_pda_enforces_grammar(mock_schema):
    """
    Proves that structural JSON mistakes are caught.
    """
    pda = JSONPushdownAutomaton(mock_schema)
    
    # Missing quotes around the key (JSON requires strings for keys)
    invalid_json = '{days: 5}'
    
    assert feed_string(pda, invalid_json) is False
