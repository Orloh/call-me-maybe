# AGENTS.md

## Project Purpose

From-scratch function-calling engine for Small Language Models. Uses constrained decoding + a pushdown automaton (PDA) + token tries to guarantee 100% schema-valid JSON output. Two-phase pipeline: (1) route to a function, (2) extract its parameters.

## Commands

```bash
make install     # uv sync --link-mode copy (installs deps, auto-installs uv if missing)
make run         # python -m src  (CUDA disabled via CUDA_VISIBLE_DEVICES="")
make debug       # same as run + --debug flag → per-token PDA/FSM trace in terminal
make lint        # flake8 src + mypy src . --disallow-untyped-defs --check-untyped-defs ...
make clean       # remove __pycache__, .pyc/.pyo, .mypy_cache, .pytest_cache
```

No dedicated `make test` target — use `pytest` directly:

```bash
pytest                          # run all tests (benchmarks excluded via addopts)
pytest tests/test_pda.py        # single file
pytest tests/test_pda.py::test_pda_integration  # single test
pytest -k "pda"                 # filter by keyword
pytest -m benchmark -s          # opt-in DFS performance benchmarks (slow, ~2 min)
```

Python 3.12+ required. Package manager is `uv` (not pip). The `llm_sdk` package is a **local editable dependency** (`tool.uv.sources.llm_sdk = {path="./llm_sdk", editable=true}`) — treat it as first-party code for reading, but **it is excluded from linting**: flake8/mypy only check `src`, and `pyproject.toml` has a `[[tool.mypy.overrides]]` entry that skips import resolution for `llm_sdk`. Don't "fix" mypy errors inside `llm_sdk/`.

## Architecture (Non-Obvious Wiring)

- **`llm_sdk.Small_LLM_Model`** (in `llm_sdk/llm_sdk/__init__.py`) wraps a HuggingFace causal LM (default `Qwen/Qwen3-0.6B`). Its `get_logits_from_input_ids(input_ids)` returns **raw logits** (no softmax) — the generator picks the max-logit allowed token id by indexing into this list.
- **`src/automata/`** is the grammar engine:
  - `primitives.py` — three FSMs (`NumberFSM`, `StringLiteralFSM`, `ExactMatchFSM`). Each implements `accepts_char(c) -> bool` (non-mutating mirror of `advance`) — this is the contract the trie DFS uses to pre-filter edges. `allowed_characters()` is a legacy finite mask (only correct for finite sets like NumberFSM/ExactMatchFSM; not used by production DFS).
  - `compiler.py` — `SchemaCompiler` translates `FunctionDefinition` pydantic models into routing tables: `dict[str, Callable[[], BaseFSM]]`. Each entry is a **factory** (fresh FSM per parse).
  - `pda.py` — `JSONPushdownAutomaton` enforces JSON structure via a `Scope` stack and `remaining_keys` tracking. Its `accepts_char()` is the mask the DFS consumes.
- **`src/trie/trie.py`** — `PrefixTrie` stores the entire model vocab as char-paths; `build_from_vocab()` is called once at bootstrap with the model's `vocab.json`.
- **`src/dfs/dfs.py`** — `find_allowed_tokens(trie_root, pda)` pre-filters each trie edge with `pda.accepts_char()` (cheap, non-mutating) and clones via `pda.clone()` (hand-written, ~30x faster than deepcopy) before recursing. The clone's `advance()` is the final arbiter. The invariant `pda.accepts_char(c) == pda.clone().advance(c)` must always hold — it is tested in `tests/test_automata_predicates.py`.
- **`src/engine/generator.py`** — `ConstrainedGenerator.generate()` loop: (1) `find_allowed_tokens`, (2) if only 1 id → fast-forward (skip LLM), else pick max-logit from allowed set, (3) advance PDA char-by-char over the decoded token text. The LLM only *ranks* the allowed set — it never decides legality.
- **`src/engine/pipeline.py`** — `FunctionCallingPipeline.process_prompt()` runs Phase 1 (routing: `compile_router_table` → PDA → generate `{"name":"..."}`) then Phase 2 (extraction: `compile_extractor_table` → new PDA → generate parameter JSON). A `ValueError("LLM hallucinated function: ...")` on a missing name is a hard fail-safe that should never fire — the FSM already constrains names.
- **`src/engine/bootstrap.py`** — `initialize_system_dependencies()` is the heavy one-time setup: instantiate model + load `vocab.json` + build trie. Called once from `__main__.py`.
- **`src/io_manager.py`** — all JSON I/O with custom semantic errors (`SchemaValidationError`, `InputFileNotFoundError`, `JSONParsingError`, `PermissionDeniedError`). Pydantic schemas use `extra='forbid'` — unknown fields are rejected.
- **`src/schema.py`** — canonical pydantic models: `FunctionDefinition`, `PromptItem`, `FunctionCallResult`, `ParameterField`. `AllowedTypes = Literal["string","number","integer","float","boolean","null"]`.

## Data Flow

`data/input/function_definitions/function_definitions.json` — 5 hardcoded functions the model can call.
`data/input/function_call_prompts/function_calling_tests.json` — 11 test prompts.
`data/output/function_calling_results.json` — written by `write_output` at the end of each run (gitignored).

## Conventions & Gotchas

- **`pythonpath = "."` in pytest config** — `src` and `llm_sdk` are importable as top-level packages. Tests use `from src.automata import ...` and `from llm_sdk import Small_LLM_Model` directly.
- **`SchemaCompiler` and `PromptBuilder` are non-instantiable** (`__init__` raises `NotImplementedError`) — all methods are `@classmethod`s.
- **Mypy is strict** (`--disallow-untyped-defs --check-untyped-defs --warn-return-any --warn-unused-ignores`). Write typed functions.
- **`ExactMatchFSM`** is used for anything with a fixed vocabulary: booleans (`true`/`false`), `null`, enum literals, JSON keys, function names. It filters candidates char-by-char via `accepts_char`.
- **The PDA enforces key completeness** — `remaining_keys` tracks required params; closing `}` while keys remain is rejected. Unknown keys are also rejected. Type mismatches (e.g., opening `"` for a `number` param) are rejected at the first char.
- **`accepts_char`/`terminates_on`/`clone` contract** — every FSM and the PDA implement three non-mutating methods used by the DFS mask: `accepts_char(c)` (mirror of `advance`), `terminates_on(c)` (mirror of the FSM→PDA hand-off), and `clone()` (cheap copy sharing immutable fields). When changing `advance()` logic in any FSM or the PDA, you MUST update its `accepts_char` mirror in the same commit — `tests/test_automata_predicates.py` checks the invariant.
- **`find_allowed_tokens` clones the PDA** per surviving trie edge — the caller's PDA is never mutated during the DFS. The PDA is only advanced in `_advance_pda()` after a token is selected.
- **Generator fast-forward** — when `len(allowed_ids) == 1`, the token is returned without querying the LLM. This is a performance optimization, not a correctness one.
- **`pyrightconfig.json`** adds `./llm_sdk` to `extraPaths` so static analysis can import it. Don't remove this.
- **`llm_sdk/pyproject.toml`** is a **separate package** with its own deps (torch, transformers, etc.). Changes to the SDK's interface require checking `pyrightconfig.json` and possibly the main `pyproject.toml`'s `uv.sources` entry.
- **`generator.py` has a stray `print(debug)` in `__init__`** (line 24) — leftover debug statement. Safe to remove but unrelated to any bug.

## Testing

- Tests live in `tests/`, one file per component (`test_pda.py`, `test_dfs.py`, `test_engine.py`, etc.).
- Heavy ML deps (`Small_LLM_Model`, `PrefixTrie`) are **mocked** in tests via `unittest.mock.patch` — don't try to instantiate them in tests.
- `test_pipeline.py` is the best reference for understanding the end-to-end flow.
- `test_io_manager.py` tests permission errors by chmod-ing files in `tmp_path` — these may fail in sandboxed environments.
- `test_dfs_benchmark.py` has two kinds of tests: an **unmarked correctness snapshot** (`test_dfs_correctness_snapshot_baseline`, runs in the default suite — hand-computed token-id sets pinning DFS behavior) and **`@pytest.mark.benchmark` tests** (excluded by default via `addopts = "-m 'not benchmark'"`; run with `pytest -m benchmark -s`). The benchmarks measure clone count per `find_allowed_tokens` call — the C1 metric.
- **C1 (DFS clone storm) — FIXED 2026-07-25.** Pre-fix baseline: 331,073 deepcopies / ~11.5s per call inside a string literal with the real vocab. Post-fix (mask via `accepts_char` + cheap `clone()`): structural states collapse to ~10¹ clones; inside strings the ~150k-token enumeration is inherent but the call now takes ~0.7s (~16x faster). Real-vocab numbers: S1=2, S2=7, S4=17 clones; S3=330,620 clones / 0.72s.

## Known Limitations

- **W2 (BPE/decoded charset mismatch):** The trie stores raw BPE strings (e.g., `Ġ` for space); the PDA operates on decoded Unicode text. Consequences: (a) structural whitespace between JSON tokens is impossible — output is always compact JSON; (b) inside string literals, byte-mapped control chars (e.g., `Ċ`→`\n`) pass the PDA but crash `_advance_pda` if selected; (c) non-ASCII Unicode works inside strings only because `accepts_char` is permissive, not because the vocab contains decoded chars. This is a known architectural friction point — a full fix would rebuild the trie on decoded vocab.

- **W7 (`trust_remote_code`):** `llm_sdk.Small_LLM_Model` defaults to `trust_remote_code=True` (required for some HF models including the default Qwen). Callers who care about supply-chain risk should pass `trust_remote_code=False` when the model supports it. No change planned in `llm_sdk/`.
