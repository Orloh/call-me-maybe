"""
End-to-end constrained generation tests.

W8: Prove that the full pipeline (trie + DFS + PDA + generator loop + decode)
produces parseable, schema-valid JSON.

Two tests:
  1. Fast synthetic-vocab test (unmarked, default suite) — Phase 2 extraction
     with a deterministic token sequence.
  2. Real Qwen-vocab test (marked benchmark) — Phase 1 routing with the
     actual 151k-token vocab. Skipped without HF cache.
"""
import json
from pathlib import Path

import pytest

from src.automata import SchemaCompiler, JSONPushdownAutomaton
from src.engine import ConstrainedGenerator, PromptBuilder
from src.schema import FunctionDefinition, ParameterField
from src.trie import PrefixTrie


# ---------------------------------------------------------------------------
# Scripted model (no torch, no transformers)
# ---------------------------------------------------------------------------

class _MockTensor1D:
    def __init__(self, data: list[int]) -> None:
        self._data = data

    def tolist(self) -> list[int]:
        return self._data


class _MockTensor:
    def __init__(self, data: list[list[int]]) -> None:
        self._data = data

    def dim(self) -> int:
        return 2

    def tolist(self) -> list[int]:
        return self._data[0]

    def __getitem__(self, idx: int) -> _MockTensor1D:
        return _MockTensor1D(self._data[idx])


class _ScriptedModel:
    def __init__(self, vocab: dict[str, int]) -> None:
        self._rev_vocab: dict[int, str] = {v: k for k, v in vocab.items()}
        self._vocab_size = len(vocab)

    def encode(self, text: str) -> _MockTensor:
        return _MockTensor([[]])

    def decode(self, ids: object) -> str:
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if isinstance(ids, list):
            ids = ids[0]
        return self._rev_vocab.get(ids, f"[UNK:{ids}]")

    def get_logits_from_input_ids(
        self, input_ids: list[int]  # noqa: ARG002
    ) -> list[float]:
        return [0.0] * self._vocab_size


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_trie(vocab: dict[str, int]) -> PrefixTrie:
    trie = PrefixTrie()
    trie.build_from_vocab(vocab)
    return trie


def _find_real_vocab() -> Path | None:
    snapshots = (
        Path.home()
        / ".cache" / "huggingface" / "hub"
        / "models--Qwen--Qwen3-0.6B" / "snapshots"
    )
    return next(iter(sorted(snapshots.glob("*/vocab.json"))), None)


# ---------------------------------------------------------------------------
# Test 1: Phase 2 extraction with deterministic synthetic vocab
# ---------------------------------------------------------------------------

_DETERMINISTIC_VOCAB = {
    "{": 0,
    '"name"': 1,  # full key '"name"' (opening quote + content + closing quote)
    ":": 2,
    '"Hello"': 3, # full string value '"Hello"' (opens, writes, closes)
    "}": 4,
}


def _fn_schema() -> FunctionDefinition:
    return FunctionDefinition(
        name="fn_demo",
        description="Demo function with a single string parameter.",
        parameters={"name": ParameterField(type="string")},
        returns=ParameterField(type="string"),
    )


def test_e2e_deterministic_synthetic_vocab() -> None:
    """
    Full generation loop with a minimal vocab where each step is
    deterministic (single allowed token).  Proves that the pipeline
    (DFS -> decode -> advance -> loop) produces parseable JSON with
    the expected schema structure.
    """
    trie = _build_trie(_DETERMINISTIC_VOCAB)
    model = _ScriptedModel(_DETERMINISTIC_VOCAB)

    extraction_prompt = PromptBuilder.build_parameters_prompt(
        "test", _fn_schema()
    )
    extractor_schema = SchemaCompiler.compile_extractor_table(_fn_schema())
    pda = JSONPushdownAutomaton(extractor_schema)
    gen = ConstrainedGenerator(model, pda, trie)

    output = gen.generate(extraction_prompt, max_new_tokens=50)
    parsed = json.loads(output)

    assert isinstance(parsed, dict), f"output should be a dict: {output!r}"
    assert "name" in parsed, f"missing 'name' key: {parsed}"
    assert isinstance(parsed["name"], str), (
        f"'name' should be a string: {parsed['name']!r}"
    )


# ---------------------------------------------------------------------------
# Test 2: Phase 1 routing with the real Qwen vocab
# ---------------------------------------------------------------------------

@pytest.mark.benchmark
def test_e2e_routing_real_vocab() -> None:
    """
    Full generation loop for Phase 1 (function routing) using the
    real Qwen3-0.6B vocag.  The routing JSON `{"name":"..."}` is
    tightly constrained by an ExactMatchFSM, so the generator reaches
    TERMINAL deterministically even with equal logits.

    Skipped if the model is not cached locally.
    """
    vocab_path = _find_real_vocab()
    if vocab_path is None:
        pytest.skip("Qwen3-0.6B vocab.json not found in HF cache")

    with open(vocab_path, encoding="utf-8") as fh:
        vocab: dict[str, int] = json.load(fh)

    assert len(vocab) > 100_000, "Real vocab seems abnormally small"

    trie = _build_trie(vocab)
    model = _ScriptedModel(vocab)

    func = FunctionDefinition(
        name="fn_greet",
        description="Generate a greeting for a person by name.",
        parameters={"name": ParameterField(type="string")},
        returns=ParameterField(type="string"),
    )

    router_prompt = PromptBuilder.build_function_name_prompt(
        "Greet Alice", [func]
    )
    router_schema = SchemaCompiler.compile_router_table([func])
    pda = JSONPushdownAutomaton(router_schema)
    gen = ConstrainedGenerator(model, pda, trie)

    output = gen.generate(router_prompt, max_new_tokens=50)
    parsed = json.loads(output)

    assert isinstance(parsed, dict), f"output should be a dict: {output!r}"
    assert "name" in parsed, f"missing 'name' key: {parsed}"
    assert parsed["name"] == "fn_greet", (
        f"unexpected function name: {parsed['name']!r}"
    )
