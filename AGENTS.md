# AGENTS.md

## Project Purpose

From-scratch function-calling engine for Small Language Models. Uses constrained decoding + a pushdown automaton (PDA) + token tries to guarantee 100% schema-valid JSON output. Two-phase pipeline: (1) route to a function, (2) extract its parameters.

## Commands

```bash
make install     # uv sync --no-cache (installs deps, auto-installs uv if missing)
make run         # python -m src (GPU-first)
make run CPU=1   # python -m src with CUDA_VISIBLE_DEVICES="" (CPU-only machine)
make debug       # pdb + --trace → per-token PDA/FSM trace in terminal
make lint        # flake8 . + mypy . with the subject's mandatory flags
make lint-strict # flake8 . + mypy . --strict
make clean       # remove __pycache__, .pyc/.pyo, .mypy_cache, .pytest_cache
```

On a CPU-only machine, prefix ad-hoc commands with `CUDA_VISIBLE_DEVICES=""`
or pass `CPU=1` to the make targets above, e.g.:

```bash
CUDA_VISIBLE_DEVICES="" uv run python -m src --input data/input/edge_floats.json
```

No dedicated `make test` target — use `pytest` directly:

```bash
pytest                          # run all tests (benchmarks excluded via addopts)
pytest tests/test_pda.py        # single file
pytest tests/test_pda.py::test_pda_integration  # single test
pytest -k "pda"                 # filter by keyword
pytest -m benchmark -s          # opt-in DFS performance benchmarks (slow, ~2 min)
```

Python 3.12+ required. Package manager is `uv` (not pip). The `llm_sdk` package is a **local editable dependency** (`tool.uv.sources.llm_sdk = {path="./llm_sdk", editable=true}`) — treat it as first-party code for reading, but **it is excluded from linting**: `.flake8` and `[tool.mypy] exclude` omit `llm_sdk`, `tests`, and `scripts`, so the subject-mandated `flake8 .` / `mypy .` gate only checks the implementation. `pyproject.toml` also has a `[[tool.mypy.overrides]]` entry that skips import resolution for `llm_sdk`. Don't "fix" mypy errors inside `llm_sdk/`.

## Architecture (Non-Obvious Wiring)

- **`llm_sdk.Small_LLM_Model`** (in `llm_sdk/llm_sdk/__init__.py`) wraps a HuggingFace causal LM (default `Qwen/Qwen3-0.6B`). Its `get_logits_from_input_ids(input_ids)` returns **raw logits** (no softmax) — the generator picks the max-logit allowed token id by indexing into this list.
- **`src/automata/`** is the grammar engine:
  - `primitives.py` — three FSMs (`NumberFSM`, `StringLiteralFSM`, `ExactMatchFSM`). Each implements `accepts_char(c) -> bool` (non-mutating mirror of `advance`) — this is the contract the trie DFS uses to pre-filter edges. Digit recognition is **ASCII-only** (`NumberFSM.DIGITS` / `StringLiteralFSM.HEX_DIGITS`), never `str.isdigit()`, so Unicode "digits" (², ٣, ０) cannot produce JSON that `json.loads` rejects. `NumberFSM` supports `mode="integer"` (rejects decimal point) and `mode="number"` (allows decimals); exponents (`e`/`E`) are intentionally unsupported. `StringLiteralFSM` enforces exactly 4 hex digits after `\u` escapes via `StringState.HEX_SEQUENCE`.
  - `compiler.py` — `SchemaCompiler` translates `FunctionDefinition` pydantic models into routing tables: `CompiledSchema = dict[str, ValueSpec]`, where `ValueSpec = Callable[[], BaseFSM] | ArraySpec | ObjectSpec`. Scalar params map to a **factory** (fresh FSM per parse); array params to a frozen `ArraySpec` (whose `item` is itself a `ValueSpec`, so arrays nest); object params to a frozen `ObjectSpec` (all-required `fields`). All via `_compile_field`, which recurses.
  - `pda.py` — `JSONPushdownAutomaton` enforces JSON structure via a **frame stack** (`stack: list[_Frame]`), one frame per open object/array. Each OBJECT frame holds its own `remaining_required` keys / `fields` / `current_spec`; each ARRAY frame holds its `item_spec` / `just_opened`. Nested objects, arrays of objects, and arrays of arrays are supported; empty `{}`/`[]` allowed; unknown/duplicate keys rejected per scope. `EXPECTING_OBJECT_START`/`EXPECTING_ARRAY_START` open scopes; `_close_scope()` pops and returns to the parent. `accepts_char()` is the DFS mask and must mirror `advance()`. `clone()` copies the frame stack (`_Frame.clone()`) plus pending fields/item spec; `remaining_key_count()` is a read-only tracing helper.
- **`src/trie/trie.py`** — `PrefixTrie` stores the entire model vocab as char-paths; `build_from_vocab()` is called once at bootstrap with the **decoded** token strings (obtained via the tokenizer's `decode()`, so `Ġ`→space, `Ċ`→newline, etc.). This guarantees the trie and PDA operate on the same character representation.
- **`src/dfs/dfs.py`** — `find_allowed_tokens(trie_root, pda)` pre-filters each trie edge with `pda.accepts_char()` (cheap, non-mutating) and clones via `pda.clone()` (hand-written, ~30x faster than deepcopy) before recursing. The clone's `advance()` is the final arbiter. The invariant `pda.accepts_char(c) == pda.clone().advance(c)` must always hold — it is tested in `tests/test_automata_predicates.py`.
- **`src/engine/generator.py`** — `ConstrainedGenerator.generate()` loop: (1) `find_allowed_tokens`, (2) if only 1 id → fast-forward (skip LLM), else pick max-logit from allowed set, (3) advance PDA char-by-char over the decoded token text (looked up in a pre-built `token_to_decoded` mapping, avoiding a `model.decode()` call per step). The LLM only *ranks* the allowed set — it never decides legality.
- **`src/engine/pipeline.py`** — `FunctionCallingPipeline.process_prompt()` runs Phase 1 (routing: `compile_router_table` → PDA → generate `{"name":"..."}`) then Phase 2 (extraction: `compile_extractor_table` → new PDA → generate parameter JSON). After parsing, `_coerce_parameters` turns `number`/`float` fields into floats (recursing into array elements) so `2` is emitted as `2.0`; `integer` fields stay ints. The router grammar only offers **real functions** (no `"none"` candidate); a defensive unknown name still returns `FunctionCallResult(name="none", parameters={})`. The CLI adds a `none` placeholder only when a prompt fails.
- **`src/engine/bootstrap.py`** — `initialize_system_dependencies()` is the heavy one-time setup: instantiate model + load `vocab.json` + build trie. Called once from `__main__.py`.
- **`src/io_manager.py`** — all JSON I/O with custom semantic errors (`SchemaValidationError`, `InputFileNotFoundError`, `JSONParsingError`, `PermissionDeniedError`). Pydantic schemas use `extra='forbid'` — unknown fields are rejected. `write_output()` is the **only** writer; the results file is the single output.
- **`src/app.py`** — testable CLI entry. `parse_args()` → `RunConfig`; `run(config, *, handle_signals=True) -> int` loads inputs **before** bootstrapping the model, records a failed prompt as a `{"name":"none","parameters":{}}` placeholder (so every prompt has exactly one result), and returns exit codes 0 (results written) / 2 (startup/write) / 130 (SIGINT). The SIGINT handler is installed only on the main thread and the previous handler is restored in a `finally`; tests pass `handle_signals=False` to opt out. The startup boundary catches `OSError` / `UnicodeError` / `RecursionError` (directory or non-UTF-8 inputs) and maps them to exit 2; `_write_records` maps any `OSError` to `WriteOutputError`. `src/__main__.py` is the entry point (sets `HF_HOME`) delegating to `app.main()`. Module-level names (`initialize_system_dependencies`, `FunctionCallingPipeline`) are monkeypatch targets in tests.
- **`src/schema.py`** — canonical pydantic models: `FunctionDefinition`, `PromptItem`, `FunctionCallResult`, `ParameterField`. `AllowedTypes = Literal["string","number","integer","float","boolean","null","array","object"]`. `ParameterField` is recursive: arrays add `items` (any `ParameterField`, so arrays nest); objects add `properties` and only support **all-required** semantics (a `required` list, if present, must name every property). `_validate_shape` raises `ValidationError` for a missing `items`/`properties`, misplaced keys, or partial `required`.

## Data Flow

`data/input/functions_definition.json` — 5 hardcoded functions the model can call.
`data/input/function_calling_tests.json` — 11 test prompts.
`data/output/function_calling_results.json` — written by `write_output` at the end of each run (gitignored).

## Conventions & Gotchas

- **`pythonpath = "."` in pytest config** — `src` and `llm_sdk` are importable as top-level packages. Tests use `from src.automata import ...` and `from llm_sdk import Small_LLM_Model` directly.
- **`SchemaCompiler` and `PromptBuilder` are non-instantiable** (`__init__` raises `NotImplementedError`) — all methods are `@classmethod`s.
- **Mypy is strict** — flags live in `[tool.mypy]` in `pyproject.toml` (`disallow_untyped_defs`, `check_untyped_defs`, `warn_return_any`, `warn_unused_ignores`, `mypy_path = "."`). `make lint` runs `flake8 .` + `mypy .` with the subject-mandated flags; `make lint-strict` runs `mypy . --strict`. `.flake8` and `[tool.mypy] exclude` omit `tests`, `scripts`, and `llm_sdk`. Write typed functions.
- **Pydantic boundaries** — all input/output data models (`FunctionDefinition`, `ParameterField`, `PromptItem`, `FunctionCallResult`) are Pydantic `BaseModel`s. FSMs, the PDA, the trie, and the generator are stateful algorithms and deliberately stay plain classes (validation on the DFS hot path would cost performance without adding safety).
- **`ExactMatchFSM`** is used for anything with a fixed vocabulary: booleans (`true`/`false`), `null`, enum literals, JSON keys, function names. It filters candidates char-by-char via `accepts_char`.
- **The PDA enforces key completeness** — `remaining_keys` tracks required params; closing `}` while keys remain is rejected. Unknown keys are also rejected. Type mismatches (e.g., opening `"` for a `number` param) are rejected at the first char.
- **`accepts_char`/`terminates_on`/`clone` contract** — every FSM and the PDA implement three non-mutating methods used by the DFS mask: `accepts_char(c)` (mirror of `advance`), `terminates_on(c)` (mirror of the FSM→PDA hand-off), and `clone()` (cheap copy sharing immutable fields). When changing `advance()` logic in any FSM or the PDA, you MUST update its `accepts_char` mirror in the same commit — `tests/test_automata_predicates.py` checks the invariant.
- **`find_allowed_tokens` clones the PDA** per surviving trie edge — the caller's PDA is never mutated during the DFS. The PDA is only advanced in `_advance_pda()` after a token is selected.
- **Generator fast-forward** — when `len(allowed_ids) == 1`, the token is returned without querying the LLM. This is a performance optimization, not a correctness one.
- **Structural whitespace filter** — `_get_allowed_tokens()` drops whitespace-only tokens that would not advance the PDA (e.g. newlines before `{`). JSON whitespace between tokens is optional, so output is compact JSON; whitespace consumed by `StringLiteralFSM` (string content) is kept. This prevents the model from stalling a structural state until `max_new_tokens`.
- **`pyrightconfig.json`** adds `./llm_sdk` to `extraPaths` so static analysis can import it. Don't remove this.
- **`llm_sdk/pyproject.toml`** is a **separate package** with its own deps (torch, transformers, etc.). Changes to the SDK's interface require checking `pyrightconfig.json` and possibly the main `pyproject.toml`'s `uv.sources` entry.
- **Output writes are atomic** — `io_manager._write_records` serializes to a temp sibling and `os.replace`s it into place. `app._save` defers SIGINT (`signal.pthread_sigmask`) during the write; a `threading.Lock` would deadlock because the SIGINT handler runs on the same thread.
- **No router/extractor retry** — decoding is greedy, so a retry would reproduce the same `MaxTokensExceededError`; it now propagates to `app._PROMPT_ERRORS`.
- **JSON-escaped keys/names** — the PDA key FSM and `SchemaCompiler` build candidates with `json.dumps(..., ensure_ascii=False)`, so property/function names containing quotes, backslashes, or unicode are representable.
- **HF cache location** — `make` exports `HF_HOME=$(PWD)/.hf_cache`; `src/__main__.py` also `setdefault`s `HF_HOME` to `<repo>/.hf_cache` *before* importing `transformers` (which resolves it at import time), so a direct `python -m src` shares the same cache. An explicit `HF_HOME` wins. Both `.hf_cache/` and `.cache/` are gitignored.
- **Decoded-vocab cache** — `initialize_system_dependencies(cache_dir=None)` caches the decoded maps under `.cache/decoded_vocab/` keyed by vocab path + mtime + size (atomic write, validated on load). Stale/corrupt caches are rebuilt; caching is skipped when the vocab path cannot be stat-ed (keeps mocked tests hermetic). `.cache/` is gitignored.
- **Pipeline compiles once** — `FunctionCallingPipeline.__init__` builds the router table and one extractor table per function; `process_prompt(user_prompt)` no longer takes `available_functions`.

## Testing

- Tests live in `tests/`, one file per component (`test_pda.py`, `test_dfs.py`, `test_engine.py`, etc.). `test_arrays.py` and `test_objects.py` cover recursive schema/compiler/PDA behavior (nested objects, arrays of objects, arrays of arrays); `test_automata_predicates.py` includes nested-object and array prefixes in the `accepts_char == advance` invariant.
- Heavy ML deps (`Small_LLM_Model`, `PrefixTrie`) are **mocked** in tests via `unittest.mock.patch` — don't try to instantiate them in tests.
- `test_pipeline.py` is the best reference for understanding the end-to-end flow.
- `test_bootstrap.py` covers the decoded-vocab disk cache (hit/stale/corrupt) and the clear non-BPE backend error; `test_bootstrap_integration.py` (benchmark-marked) drives `initialize_system_dependencies` with the real Qwen tokenizer and asserts the decoded trie + cache reuse.
- `test_io_manager.py` tests permission errors by chmod-ing files in `tmp_path` — these may fail in sandboxed environments.
- `test_dfs_benchmark.py` has two kinds of tests: an **unmarked correctness snapshot** (`test_dfs_correctness_snapshot_baseline`, runs in the default suite — hand-computed token-id sets pinning DFS behavior) and **`@pytest.mark.benchmark` tests** (excluded by default via `addopts = "-m 'not benchmark'"`; run with `pytest -m benchmark -s`). The benchmarks measure clone count per `find_allowed_tokens` call — the C1 metric.
- **C1 (DFS clone storm) — FIXED 2026-07-25.** Pre-fix baseline: 331,073 deepcopies / ~11.5s per call inside a string literal with the real vocab. Post-fix (mask via `accepts_char` + cheap `clone()`): structural states collapse to ~10¹ clones; inside strings the ~150k-token enumeration is inherent but the call now takes ~0.7s (~16x faster). Real-vocab numbers: S1=2, S2=7, S4=17 clones; S3=330,620 clones / 0.72s.

## Known Limitations

- **W3 (no KV cache):** the model has no KV cache, so `generate()` re-runs the whole sequence every step (O(n²) attention) via `get_logits_from_input_ids`, which also allocates two vocab-sized lists per call. Only the single-candidate fast-forward (`len(allowed_ids) == 1`) skips the model. Fixing this properly requires a stateful SDK API; deliberately left as-is (`llm_sdk/` untouched).
- **W8 (byte-level BPE partial sequences):** the trie is built from `tokenizer.decode([id])` per token. Tokens that are partial UTF-8 sequences decode in isolation to `U+FFFD`; multiple ids then collide and `decoded_vocab` keeps only the last (logged at bootstrap). Output JSON stays valid, but exact handling of split multi-byte characters needs a byte-aware trie (raw bytes + partial-UTF-8 state) — out of scope.
- **W10 (prompt injection):** the user request and function catalog are framed as untrusted data in both prompt builders, and grammar constraints prevent malformed output/unknown functions, but injected text can still bias function *selection*.
- **W7 (`trust_remote_code`):** `llm_sdk.Small_LLM_Model` defaults to `trust_remote_code=True` (required for some HF models including the default Qwen). Callers who care about supply-chain risk should pass `trust_remote_code=False` when the model supports it. No change planned in `llm_sdk/`.
- **`FunctionDefinition.returns`** is required for schema completeness only; it is not used during generation.
