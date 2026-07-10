import pytest
import math
from unittest.mock import MagicMock, patch, call
from src.engine import ConstrainedGenerator
from src.automata import PDAState


@pytest.fixture
def mock_generator():
    """
    Creates a ConstraineGenerator with completely mocked dependencies
    so we can isolate and test the logic inside the class methods.
    """
    mock_model = MagicMock()
    mock_pda = MagicMock()
    mock_trie = MagicMock()

    return ConstrainedGenerator(
            model=mock_model,
            pda=mock_pda,
            trie=mock_trie
    )


@patch("src.engine.generator.find_allowed_tokens")
def test_get_allowed_tokens_success(mock_find_allowed, mock_generator):
    """
    Proves that if the DFS finds valid tokens, they are returned
    and the DFS was called with the correct Trie root and PDA.
    """
    mock_find_allowed.return_value = [42, 89, 105]
    
    result = mock_generator._get_allowed_tokens()

    assert result == [42, 89, 105]
    mock_find_allowed.assert_called_once_with(mock_generator.trie.root, mock_generator.pda)


@patch("src.engine.generator.find_allowed_tokens")
def test_get_allowed_tokens_deadlock_raises_error(mock_find_allowed, mock_generator):
    """
    Prove that if the DFS retuns an empty list, the generator catches it
    and throws the correct Grammar deadlock exeption.
    """
    mock_find_allowed.return_value = []

    with pytest.raises(RuntimeError, match="Grammar deadlock"):
        mock_generator._get_allowed_tokens()


def test_apply_logits_mask_crushes_disallowed_tokens(mock_generator):
    """
    Proves that any token ID not in the allowed list has its probability
    score mathematically reduced to negative infinity.
    """
    raw_logits = [0.5, -1.2, 3.4, 10.0, 0.0]
    allowed_ids = [1, 3]

    masked_logits = mock_generator._apply_logits_mask(raw_logits, allowed_ids)

    assert masked_logits[1] == -1.2
    assert masked_logits[3] == 10.0

    assert masked_logits[0] == -math.inf
    assert masked_logits[2] == -math.inf
    assert masked_logits[4] == -math.inf

def test_generate_loop_orchestration(mock_generator):
    """
    Proves that the generate loop correctly ties all helper methods togheter,
    accumulates the generated text, and breaks immediately when the PDA reaches
    the TERMINAL state.
    """
    prompt = "Make a JSON"
    mock_generator.model.tokenize.return_value = [100]

    mock_generator._get_allowed_tokens = MagicMock(side_effect=[[1], [2]])
    mock_generator._select_next_token= MagicMock(side_effect=[1, 2])
    mock_generator.model.decode.side_effect = ["{", "}"]
    mock_generator._advance_pda = MagicMock()

    def mock_advance_side_effect(text_chunk):
        if text_chunk == "}":
            mock_generator.pda.state = PDAState.TERMINAL

    mock_generator._advance_pda.side_effect = mock_advance_side_effect

    mock_generator.pda.state = PDAState.EXPECTING_ARRAY_START
    
    result = mock_generator.generate(prompt, max_new_tokens=10)

    assert result == "{}"

    mock_generator.model.tokenize.assert_called_once_with(prompt)

    assert mock_generator._get_allowed_tokens.call_count == 2
    assert mock_generator._advance_pda.call_count == 2

    mock_generator._advance_pda.assert_has_calls([call("{"), call("}")])
    

def test_select_next_token_fast_forwards(mock_generator):
    """
    Proves that if the grammar only allows a single token,
    the engine instantly returns it and completely bypasses the LLM.
    """
    current_tokens = [101, 102]
    allowed_ids = [42]

    result = mock_generator._select_next_token(current_tokens, allowed_ids)

    assert result == 42

    mock_generator.model.get_logits_from_input_ids.assert_not_called()


def test_select_next_token_uses_llm_and_mask(mock_generator):
    """
    Proves that with multiple valid choices, the engine queries the LLM,
    applies the mask, and selects the highest scoring allowed token.
    """
    current_tokens = [101, 102]
    allowed_ids = [1, 3]

    mock_logits = [999.0, 2.0, 5.5, 10.0, -1.0]

    mock_generator.model.get_logits_from_input_ids.return_value = mock_logits

    result = mock_generator._select_next_token(current_tokens, allowed_ids)

    assert result == 3

    mock_generator.model.get_logits_from_input_ids.assert_called_once_with(current_tokens)


def test_advance_pda_success(mock_generator):
    """
    Proves that if the PDA accepts every character, the method completes
    successfully and feeds each character sequentially.
    """
    mock_generator.pda.advance.return_value = True
    text_chunk = '{"'

    mock_generator._advance_pda(text_chunk)

    mock_generator.pda.advance.assert_has_calls([call('{'), call('"')])
    assert mock_generator.pda.advance.call_count == 2


def test_advance_pda_raises_error_on_rejection(mock_generator):
    """
    Proves that if the PDA rejects a character, a RuntimeError is raised,
    indicating a catastrophic desync between the DFS and the PDA.
    """
    mock_generator.pda.advance.side_effect = [True, False]

    text_chunk = 'aX'

    with pytest.raises(RuntimeError, match="PDA rejected safely char 'X'. Check DFS logic!"):
        mock_generator._advance_pda(text_chunk)
