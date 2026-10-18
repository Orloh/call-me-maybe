"""
Integration test for bootstrap against the real Qwen tokenizer.

Closes the gap where the decoded-vocab path was only ever exercised with a
mock decode. Marked `benchmark` (builds the full 150k-token trie) and skipped
when the tokenizer/vocab is not present in the local HF cache.
"""
import json
from pathlib import Path

import pytest

from src.engine import bootstrap as bootstrap_mod

_HF_HUB = Path.home() / ".cache" / "huggingface" / "hub"
_SNAPSHOTS = (
    _HF_HUB / "models--Qwen--Qwen3-0.6B" / "snapshots"
)


def _find_real_vocab() -> Path | None:
    return next(iter(sorted(_SNAPSHOTS.glob("*/vocab.json"))), None)


@pytest.mark.benchmark
def test_bootstrap_builds_decoded_trie(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transformers = pytest.importorskip("transformers")

    vocab_path = _find_real_vocab()
    if vocab_path is None:
        pytest.skip("Qwen3-0.6B vocab.json not found in HF cache")

    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-0.6B"
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"tokenizer unavailable: {exc}")

    class _FakeModel:
        def get_path_to_vocab_file(self) -> str:
            return str(vocab_path)

        def decode(self, ids: object) -> str:
            return tokenizer.decode(ids, skip_special_tokens=True)

    monkeypatch.setattr(bootstrap_mod, "Small_LLM_Model", _FakeModel)

    model, trie, token_to_decoded = (
        bootstrap_mod.initialize_system_dependencies()
    )

    assert isinstance(model, _FakeModel)

    # Independently recompute the decoded map and compare sizes / contents.
    raw_vocab: dict[str, int] = json.loads(
        vocab_path.read_text(encoding="utf-8")
    )
    expected: dict[str, int] = {}
    expected_id_to_text: dict[int, str] = {}
    for token_id in raw_vocab.values():
        decoded = tokenizer.decode([token_id], skip_special_tokens=True)
        if not decoded:
            continue
        expected[decoded] = token_id
        expected_id_to_text[token_id] = decoded

    assert trie.size == len(expected)
    assert token_to_decoded == expected_id_to_text

    # The byte-level sentinels are decoded: raw "Ġ"/"Ċ" become " "/"\n".
    space_id = raw_vocab["Ġ"]
    newline_id = raw_vocab["Ċ"]
    assert token_to_decoded[space_id] == " "
    assert token_to_decoded[newline_id] == "\n"
    assert " " in trie.root.children

    # Shadowed tokens are the duplicate decoded strings, last-wins.
    shadowed = len(raw_vocab) - len(expected)
    assert shadowed > 0
