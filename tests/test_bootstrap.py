import os
from pathlib import Path
from typing import Any

import pytest
import json
from unittest.mock import MagicMock, patch, mock_open

from src.engine import initialize_system_dependencies


def _stub_model(
    vocab_path: Path, vocab: dict[str, int]
) -> tuple[MagicMock, list[Any]]:
    """A fake model whose decode is the identity over `vocab`, counting calls."""
    rev = {token_id: text for text, token_id in vocab.items()}
    calls: list[Any] = []
    model = MagicMock()
    model.get_path_to_vocab_file.return_value = str(vocab_path)

    def side_effect(ids: list[int]) -> str:
        calls.append(ids[0])
        return rev.get(ids[0], "")

    model.decode.side_effect = side_effect
    return model, calls

def test_initialize_system_dependencies_success():
    """
    Proves that the bootstrap function successfully instantiates the model,
    reads the vocabulary file, builds the Trie, and returns them correctly.
    """

    dummy_vocab = {"hello": 0, "world": 1, "!": 2}
    # Decode side-effect: return the token string for a single-token decode.
    reverse_vocab = {v: k for k, v in dummy_vocab.items()}

    dummy_vocab_json = json.dumps(dummy_vocab)
    fake_vocab_path = "/fake/path/to/vocab.json"

    with (
        patch("src.engine.bootstrap.Small_LLM_Model") as mock_model_class,
        patch("src.engine.bootstrap.PrefixTrie") as mock_trie_class,
        patch("builtins.open", mock_open(read_data=dummy_vocab_json)) as mock_file
    ):

        mock_model_instance = mock_model_class.return_value
        mock_model_instance.get_path_to_vocab_file.return_value = fake_vocab_path
        mock_model_instance.decode.side_effect = (
            lambda ids: reverse_vocab.get(ids[0], "") if ids else ""
        )

        mock_trie_instance = mock_trie_class.return_value
        mock_trie_instance.size = 3

        model, trie, token_to_decoded = initialize_system_dependencies()

        mock_model_class.assert_called_once()
        mock_model_instance.get_path_to_vocab_file.assert_called_once()

        mock_file.assert_called_once_with(fake_vocab_path, "r", encoding="utf-8")

        mock_trie_class.assert_called_once()
        # The decoded mapping should equal the raw vocab for ASCII tokens.
        mock_trie_instance.build_from_vocab.assert_called_once_with(dummy_vocab)

        assert model is mock_model_instance
        assert trie is mock_trie_instance
        assert token_to_decoded == {0: "hello", 1: "world", 2: "!"}

def test_initialize_system_dependencies_shadowed_decoded_tokens():
    """Two ids decoding to the same text: the last id wins in the trie."""
    raw_vocab = {"a": 0, "b": 1, "c": 2}
    decoded_by_id = {0: "x", 1: "x", 2: "y"}

    with (
        patch("src.engine.bootstrap.Small_LLM_Model") as mock_model_class,
        patch("src.engine.bootstrap.PrefixTrie") as mock_trie_class,
        patch("builtins.open", mock_open(read_data=json.dumps(raw_vocab))),
    ):
        inst = mock_model_class.return_value
        inst.get_path_to_vocab_file.return_value = "/fake/vocab.json"
        inst.decode.side_effect = lambda ids: decoded_by_id.get(ids[0], "")
        trie = mock_trie_class.return_value

        _, _, token_to_decoded = initialize_system_dependencies()

        trie.build_from_vocab.assert_called_once_with({"x": 1, "y": 2})
        assert token_to_decoded == {0: "x", 1: "x", 2: "y"}


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


_VOCAB = {"hello": 0, "world": 1, "!": 2}
_EXPECTED_MAP = {0: "hello", 1: "world", 2: "!"}


def test_bootstrap_caches_and_reuses_decoded_vocab(tmp_path: Path) -> None:
    vocab_file = tmp_path / "vocab.json"
    vocab_file.write_text(json.dumps(_VOCAB), encoding="utf-8")
    cache_dir = tmp_path / "cache"

    model, calls = _stub_model(vocab_file, _VOCAB)

    with patch(
        "src.engine.bootstrap.Small_LLM_Model", return_value=model
    ):
        _, trie, mapping = initialize_system_dependencies(cache_dir=cache_dir)
        assert calls == [0, 1, 2]
        assert mapping == _EXPECTED_MAP
        assert len(list(cache_dir.glob("*.json"))) == 1

        calls.clear()
        _, trie2, mapping2 = initialize_system_dependencies(
            cache_dir=cache_dir
        )

    assert calls == []  # cache hit: no decode calls
    assert mapping2 == mapping
    assert trie2.size == trie.size


def test_bootstrap_corrupt_cache_rebuilds(tmp_path: Path) -> None:
    vocab_file = tmp_path / "vocab.json"
    vocab_file.write_text(json.dumps(_VOCAB), encoding="utf-8")
    cache_dir = tmp_path / "cache"
    model, calls = _stub_model(vocab_file, _VOCAB)

    with patch(
        "src.engine.bootstrap.Small_LLM_Model", return_value=model
    ):
        initialize_system_dependencies(cache_dir=cache_dir)
        cache_file = next(cache_dir.glob("*.json"))
        cache_file.write_text("not valid json", encoding="utf-8")

        calls.clear()
        _, _, mapping = initialize_system_dependencies(cache_dir=cache_dir)

    assert calls == [0, 1, 2]  # rebuilt from scratch
    assert mapping == _EXPECTED_MAP


def test_bootstrap_stale_cache_rebuilds(tmp_path: Path) -> None:
    vocab_file = tmp_path / "vocab.json"
    vocab_file.write_text(json.dumps(_VOCAB), encoding="utf-8")
    cache_dir = tmp_path / "cache"
    model, calls = _stub_model(vocab_file, _VOCAB)

    with patch(
        "src.engine.bootstrap.Small_LLM_Model", return_value=model
    ):
        initialize_system_dependencies(cache_dir=cache_dir)

        os.utime(vocab_file, (os.path.getmtime(vocab_file) + 10,) * 2)

        calls.clear()
        initialize_system_dependencies(cache_dir=cache_dir)

    assert calls == [0, 1, 2]  # new key -> rebuild
    assert len(list(cache_dir.glob("*.json"))) == 2
