import pytest
import json
from unittest.mock import MagicMock, patch, mock_open

from src.engine import initialize_system_dependencies

def test_initialize_system_dependencies_success():
    """
    Proves that the bootstrap function successfully instantiates the model,
    reads the vocabulary file, builds the Trie, and returns them correctly.
    """
    
    dummy_vocab = {"hello": 0, "world": 1, "!": 2}
    dummy_vocab_json = json.dumps(dummy_vocab)
    fake_vocab_path = "/fake/path/to/vocab.json"
        
    with (
        patch("src.engine.bootstrap.Small_LLM_Model") as mock_model_class,
        patch("src.engine.bootstrap.PrefixTrie") as mock_trie_class,
        patch("builtins.open", mock_open(read_data=dummy_vocab_json)) as mock_file
    ):

        mock_model_instance = mock_model_class.return_value
        mock_model_instance.get_path_to_vocab_file.return_value = fake_vocab_path

        mock_trie_instance = mock_trie_class.return_value
        mock_trie_instance.size = 3

        model, trie = initialize_system_dependencies()

        mock_model_class.assert_called_once()
        mock_model_instance.get_path_to_vocab_file.assert_called_once()

        mock_file.assert_called_once_with(fake_vocab_path, "r", encoding="utf-8")

        mock_trie_class.assert_called_once()
        mock_trie_instance.build_from_vocab.assert_called_once_with(dummy_vocab)

        assert model is mock_model_instance
        assert trie is mock_trie_instance

@patch("src.engine.bootstrap.Small_LLM_Model")
def test_initialize_system_dependencies_file_not_found(mock_model_class):
    """
    Proves that if the model's vocabulary file is missing, the bootstrap
    fails loudly rather than swallowing the error.
    """
    mock_model_instance = mock_model_class.return_value
    mock_model_instance.get_path_to_vocab_file.return_value = "/missing/vocab.json"
    
    with patch("builtins.open", side_effect=FileNotFoundError("Missing vocab")):
        with pytest.raises(FileNotFoundError, match="Missing vocab"):
            initialize_system_dependencies()
