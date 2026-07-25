"""
Benchmarks and characterization tests for src.dfs.find_allowed_tokens.

Purpose: guard the C1 fix. The old DFS deep-copied the PDA once per
trie edge (~331k deepcopies / ~11.5s per call inside string literals
with the real 151k-token vocab). The current DFS pre-filters edges
with pda.accepts_char() and uses a cheap hand-written clone().

Expected post-fix profile: clone counts collapse for structural states
(S1/S2/S4); inside strings (S3) the enumeration of ~150k legal tokens
is inherent to the algorithm, but each clone is ~30x cheaper, so the
runtime collapses even though the clone count stays high.

The unmarked test `test_dfs_correctness_snapshot_baseline` runs in the
default suite and pins the exact behavior; it is the equivalence guard
for the optimization. Everything marked `benchmark` is opt-in:

    pytest -m benchmark -s
"""
import json
import random
import string
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator
from unittest.mock import patch

import pytest

from src.automata import (
    CompiledSchema,
    JSONPushdownAutomaton,
    NumberFSM,
    StringLiteralFSM,
)
from src.dfs import find_allowed_tokens
from src.trie import PrefixTrie, TrieNode

# ---------------------------------------------------------------------------
# Shared fixtures & helpers
# ---------------------------------------------------------------------------

# Mirrors a realistic extractor schema: one string param, two number params.
_SCHEMA: CompiledSchema = {
    "name": StringLiteralFSM,
    "a": NumberFSM,
    "b": NumberFSM,
}

# Each scenario is the prefix the PDA consumes to reach the target state.
_SCENARIOS: dict[str, str] = {
    "S1_object_start": "",
    "S2_key_match": "{",
    "S3_inside_string": '{"name":"Hel',
    "S4_inside_number": '{"name":"x","a":12',
}


def _build_pda(prefix: str) -> JSONPushdownAutomaton:
    """Returns a PDA advanced through every char of `prefix`."""
    pda = JSONPushdownAutomaton(_SCHEMA)
    for char in prefix:
        assert pda.advance(char), f"PDA rejected {char!r} of {prefix!r}"
    return pda


def _build_trie(vocab: dict[str, int]) -> PrefixTrie:
    trie = PrefixTrie()
    trie.build_from_vocab(vocab)
    return trie


@contextmanager
def _count_clones() -> Iterator[Callable[[], int]]:
    """
    Wraps JSONPushdownAutomaton.clone with a counter and yields a getter.
    One clone == one trie edge actually explored — the post-fix C1 metric.
    """
    count = 0
    original = JSONPushdownAutomaton.clone

    def counting_clone(self: JSONPushdownAutomaton) -> JSONPushdownAutomaton:
        nonlocal count
        count += 1
        return original(self)

    with patch.object(JSONPushdownAutomaton, "clone", counting_clone):
        yield lambda: count


def _measure(root: TrieNode, pda: JSONPushdownAutomaton) -> tuple[int, float, int]:
    """Single DFS run. Returns (clones, elapsed_seconds, n_allowed)."""
    with _count_clones() as get_count:
        start = time.perf_counter()
        allowed = find_allowed_tokens(root, pda)
        elapsed = time.perf_counter() - start
    return get_count(), elapsed, len(allowed)


def _median_runtime(fn: Callable[[], object], repeats: int = 3) -> float:
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    samples.sort()
    return samples[len(samples) // 2]


# ---------------------------------------------------------------------------
# Synthetic vocabulary (deterministic, adversarial: heavy shared prefixes)
# ---------------------------------------------------------------------------

_JSON_CRITICAL_TOKENS = [
    "{", "}", ":", ",",
    '"name"', '"a"', '"b"',
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "12", "34", "56", "789",
    "Hello", "world", "el", "lo", "x", "yz",
    "Ġ", "Ġthe", "true", "false",
]


def _generate_synthetic_vocab(size: int) -> dict[str, int]:
    """
    Builds `size` unique tokens with heavy shared prefixes (70% of tokens
    extend an existing one), an alphabet that mirrors byte-level BPE
    (includes 'Ġ'/'Ċ', no literal spaces), and length cap 20.
    """
    rng = random.Random(42)
    alphabet = string.ascii_letters + string.digits + "ĠĊ.,_-\"':{}[]"
    vocab: dict[str, int] = {}
    keys: list[str] = []

    def add(token: str) -> None:
        if token and token not in vocab:
            vocab[token] = len(vocab)
            keys.append(token)

    for token in _JSON_CRITICAL_TOKENS:
        add(token)

    while len(vocab) < size:
        base = rng.choice(keys) if rng.random() < 0.7 else ""
        suffix = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 3)))
        add((base + suffix)[:20])

    return vocab


@pytest.fixture(scope="module")
def vocab_150k() -> dict[str, int]:
    return _generate_synthetic_vocab(150_000)


@pytest.fixture(scope="module")
def trie_150k(vocab_150k: dict[str, int]) -> PrefixTrie:
    return _build_trie(vocab_150k)


# ---------------------------------------------------------------------------
# Correctness snapshot — runs in the DEFAULT suite (no benchmark marker).
# Exact hand-computed expectations; guards any future DFS optimization.
# ---------------------------------------------------------------------------

_SNAPSHOT_VOCAB = {
    "{": 0, "}": 1, '"name"': 2, '"a"': 3, '"b"': 4, '"xyz"': 5,
    ":": 6, ",": 7, "12": 8, "34": 9, "5": 10,
    "x": 11, "yz": 12, "Hello": 13, "true": 14, "Ġ": 15,
}

_SNAPSHOT_EXPECTED: dict[str, list[int]] = {
    # Only '{' can start the object.
    "S1_object_start": [0],
    # ExactMatchFSM over quoted keys accepts "name"/"a"/"b", prunes "xyz".
    "S2_key_match": [2, 3, 4],
    # Inside a string every printable token is legal; quoted-key tokens
    # die after the closing quote, so 2/3/4/5 are pruned.
    "S3_inside_string": [0, 1, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    # Inside a number: digits continue it; ',' terminates then triggers
    # EXPECTING_KEY; '}' is REJECTED because key 'b' is still required.
    "S4_inside_number": [7, 8, 9, 10],
}


@pytest.mark.parametrize("scenario", sorted(_SNAPSHOT_EXPECTED))
def test_dfs_correctness_snapshot_baseline(scenario: str) -> None:
    trie = _build_trie(_SNAPSHOT_VOCAB)
    pda = _build_pda(_SCENARIOS[scenario])

    allowed = find_allowed_tokens(trie.root, pda)

    assert sorted(allowed) == _SNAPSHOT_EXPECTED[scenario]


# ---------------------------------------------------------------------------
# Benchmarks (opt-in: pytest -m benchmark -s)
# ---------------------------------------------------------------------------

@pytest.mark.benchmark
def test_dfs_clone_count_per_state(trie_150k: PrefixTrie) -> None:
    """
    The post-fix C1 metric: clones per find_allowed_tokens call.
    Structural states must collapse below the ceiling; S3 stays high
    (enumerating ~134k legal tokens is inherent) but must be fast.
    """
    counts: dict[str, int] = {}

    print("\n--- clones per DFS call (150k synthetic vocab) ---")
    for name, prefix in _SCENARIOS.items():
        pda = _build_pda(prefix)
        count, elapsed, n_allowed = _measure(trie_150k.root, pda)
        counts[name] = count
        print(
            f"{name:<20} clones={count:>8,} "
            f"time={elapsed:>7.3f}s allowed={n_allowed:,}"
        )
        assert n_allowed > 0

    # The mask prunes structural states: no full-trie exploration.
    assert counts["S1_object_start"] < 1_000
    assert counts["S2_key_match"] < 1_000
    assert counts["S4_inside_number"] < 10_000


@pytest.mark.benchmark
def test_dfs_runtime_scales_with_vocab_size(
    vocab_150k: dict[str, int],
) -> None:
    """
    S3 runtime at increasing vocab sizes. Enumeration of legal tokens is
    inherently O(legal tokens), but the per-edge cost must be cheap.
    The pre-fix baseline was 0.53s / 2.76s / 8.26s for 10k/50k/150k.
    """
    items = list(vocab_150k.items())

    print("\n--- S3_inside_string runtime vs vocab size ---")
    previous: float | None = None
    for size in (10_000, 50_000, 150_000):
        trie = _build_trie(dict(items[:size]))
        pda = _build_pda(_SCENARIOS["S3_inside_string"])
        median = _median_runtime(
            lambda: find_allowed_tokens(trie.root, pda)
        )
        growth = f"{median / previous:>5.2f}x" if previous else "  base"
        print(f"vocab={size:>7,} median={median:>7.3f}s growth={growth}")
        previous = median

    # Pre-fix this was ~8.3s; the cheap clone must bring it well under 1s.
    assert previous is not None
    assert previous < 1.0


@pytest.mark.benchmark
def test_dfs_benchmark_report(
    vocab_150k: dict[str, int],
    trie_150k: PrefixTrie,
) -> None:
    """Full matrix: 4 states x 3 vocab sizes, single run per cell."""
    items = list(vocab_150k.items())
    tries = {
        size: trie_150k if size == 150_000 else _build_trie(dict(items[:size]))
        for size in (10_000, 50_000, 150_000)
    }

    print("\n--- DFS benchmark report (clones / seconds / allowed) ---")
    header = f"{'scenario':<20}" + "".join(
        f"{size:>18,}" for size in tries
    )
    print(header)
    for name, prefix in _SCENARIOS.items():
        row = f"{name:<20}"
        for size, trie in tries.items():
            count, elapsed, n_allowed = _measure(trie.root, _build_pda(prefix))
            row += f"{count:>8,}/{elapsed:>5.2f}s/{n_allowed:<3}"
        print(row)


@pytest.mark.benchmark
def test_dfs_real_vocab_benchmark() -> None:
    """
    Same measurement against the real Qwen3-0.6B vocab.json when present
    in the local HF cache. Skips silently otherwise.
    """
    snapshots = (
        Path.home()
        / ".cache" / "huggingface" / "hub"
        / "models--Qwen--Qwen3-0.6B" / "snapshots"
    )
    vocab_path = next(iter(sorted(snapshots.glob("*/vocab.json"))), None)
    if vocab_path is None:
        pytest.skip("Qwen3-0.6B vocab.json not found in HF cache")

    with open(vocab_path, encoding="utf-8") as fh:
        vocab: dict[str, int] = json.load(fh)
    trie = _build_trie(vocab)

    print(f"\n--- real vocab ({len(vocab):,} tokens) ---")
    for name, prefix in _SCENARIOS.items():
        count, elapsed, n_allowed = _measure(trie.root, _build_pda(prefix))
        print(
            f"{name:<20} clones={count:>8,} "
            f"time={elapsed:>7.3f}s allowed={n_allowed:,}"
        )
        assert n_allowed > 0
        # Pre-fix S3 was ~11.5s here; must now be fast in every state.
        assert elapsed < 1.0
