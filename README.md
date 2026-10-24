*This project has been created as part of the 42 curriculum by ohernan.*

# call-me-maybe

## Description

call-me-maybe is a from-scratch function-calling engine for Small Language Models (SLMs). It uses constrained decoding with a pushdown automaton (PDA) and token tries to guarantee 100% parseable JSON output from a 0.6B parameter model (Qwen/Qwen3-0.6B), with every function call constrained to its schema.

Given a natural-language request like *"What is the sum of 2 and 3?"*, the system returns a structured function call:

```json
{"name": "fn_add_numbers", "parameters": {"a": 2.0, "b": 3.0}}
```

The engine operates in two phases:
1. **Routing** — selects which function to call based on the user prompt
2. **Extraction** — extracts typed arguments matching the function's parameter schema

Both phases use the same constrained-decoding core: a prefix trie over the model's 150k-token vocabulary is pruned token-by-token via the PDA's `accepts_char` mask, guaranteeing that every generated token contributes to valid JSON.

## Instructions

### Installation

```bash
make install
```

This runs `uv sync --no-cache`, which installs dependencies and configures the local `llm_sdk` package as an editable dependency.

### Running
```bash
make run
```
On a CPU-only machine, disable CUDA first:

```bash
make run CPU=1
```

`CPU=1` sets `CUDA_VISIBLE_DEVICES=""` for `run`, `debug`, and `trace`.
The same can be done directly:

```bash
CUDA_VISIBLE_DEVICES="" uv run python -m src
```

Both invocations share the repo-local Hugging Face cache in `.hf_cache/`:
`make` exports `HF_HOME`, and the entry point sets it (only when unset)
before importing the model. Set `HF_HOME` yourself to use a different cache.

Or with custom paths:

```bash
CUDA_VISIBLE_DEVICES="" uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calling_results.json
```

The results file is the **only** file written by a run.

### Error Handling & Exit Codes

The CLI is best-effort per prompt: every prompt produces exactly one object
in the results file. If a prompt cannot be routed or processed, the error is
logged and recorded as:

```json
{"prompt": "...", "name": "none", "parameters": {}}
```

`none` is a failure sentinel — it is **not** a function from
`functions_definition.json`. Normal runs (including ambiguous prompts) route
to one of the available functions. Exit codes are:

| Code | Meaning |
|---|---|
| 0 | Ran to completion; the results file was written |
| 2 | Startup failure: missing/invalid input, schema error, bootstrap or write error |
| 130 | Interrupted (Ctrl-C); partial results were saved |

Inputs are loaded **before** the model is bootstrapped, so invalid input
fails fast without paying for model startup.

Directory or non-UTF-8 input files, and output paths whose parent is a
regular file, are reported as exit 2 rather than crashing. Ctrl-C saves
partial results before exiting 130.

### Trace Mode

```bash
make trace
# or: uv run python -m src --trace
```

Enables per-token PDA and FSM state tracing to the terminal, showing every generation step and the remaining schema keys.

### Linting

```bash
make lint          # flake8 . + mypy . with the mandatory flags
make lint-strict   # flake8 . + mypy . --strict
```

`flake8 .` and `mypy .` are the commands required by the subject. Tests,
scripts, and the vendored `llm_sdk` are excluded through `.flake8` and
`[tool.mypy]` because they are not part of the implementation under test.

### Cleaning

```bash
make clean
```

Removes `__pycache__`, `.pyc`, `.mypy_cache`, and `.pytest_cache`.

## Algorithm

### Constrained Decoding

At each generation step, the model produces logits over its 150k-token vocabulary. Instead of picking the highest-probability token unconditionally, we:

1. **Build** a prefix trie from the decoded vocabulary (character-paths, not raw tokens)
2. **Mask** each trie edge through the PDA's `accepts_char()` method (non-mutating, cheap)
3. **Clone** the PDA for each surviving edge and feed the character via `advance()` to confirm validity
4. **Collect** all token IDs that survive the full DFS traversal
5. **Drop optional structural whitespace** (spaces/newlines between JSON tokens) so a model cannot stall in a state such as `EXPECTING_OBJECT_START`; whitespace inside strings is preserved
6. **Select** the max-logit token from the allowed set, or fast-forward when only one token is valid

This guarantees that every output token contributes to valid, schema-compliant JSON.

### Pushdown Automaton (PDA)

The PDA enforces JSON structure using a `Scope` stack (OBJECT/ARRAY) and a state machine with states like `EXPECTING_KEY`, `EXPECTING_COLON`, `EXPECTING_VALUE`, and `EXPECTING_COMMA_OR_END`. It tracks `remaining_keys` per nesting level to enforce key completeness and reject unknown keys. The `accepts_char()` method mirrors `advance()` without mutation, enabling the DFS to pre-filter trie edges cheaply.

### Prefix Trie

The trie stores the model's entire vocabulary as character-level paths. It is built once at bootstrap from the tokenizer's decoded token strings. The DFS walks the trie root, checking each edge's character against the PDA mask, and collects terminal token IDs.

## Design Decisions

### Two-Phase Pipeline

Routing and extraction are separated into two generation calls with distinct PDAs. This keeps each grammar tight: the router PDA only accepts `{"name": "fn_x"}` for a function in the current catalog, while the extractor PDA only accepts the chosen function's parameter schema. A single-shot approach would require a more complex PDA supporting nested objects. The `none` failure sentinel is produced by the CLI, never by the router grammar.

### Recursive Values (Objects and Arrays)

Parameters may be nested objects and arrays, declared as a JSON Schema subset:

```json
{
  "config": {
    "type": "object",
    "properties": {
      "retries": { "type": "integer" },
      "labels": { "type": "array", "items": { "type": "string" } }
    }
  },
  "items": {
    "type": "array",
    "items": { "type": "object", "properties": { "id": { "type": "number" } } }
  }
}
```

Values are fully recursive: objects in objects, arrays of objects, arrays of
arrays. Empty objects (`{}`) and empty arrays (`[]`) are allowed; trailing
commas and unknown or duplicate keys are rejected at every level. All object
properties are required (a `required` list, if present, must name every
property).

The PDA tracks nesting with a stack of frames — one per open object or array —
each carrying its own remaining-required keys, current field spec, or element
spec. `EXPECTING_OBJECT_START`/`EXPECTING_ARRAY_START` open new scopes, and
`clone()` copies the frame stack so the DFS mask stays independent per branch.

### Pydantic Validation

Every data model crossing the program's boundaries (`ParameterField`,
`FunctionDefinition`, `PromptItem`, `FunctionCallResult`) is a Pydantic
`BaseModel` with `extra='forbid'` and custom validators. The FSMs, the PDA,
the trie, and the generator are **stateful algorithms**, not data containers;
keeping them as plain classes avoids validation overhead on the DFS hot path
(especially `_Frame.clone()`) without losing any safety at the I/O boundary.

### Numeric Output Coercion

The grammar accepts both `6` and `6.0` for a `number` parameter. To match the
expected output format, `number`/`float` parameters are coerced to `float`
after extraction (recursing through nested arrays and objects), so `2` is
emitted as `2.0`. `integer` parameters remain ints.

### PDA over Regex

A pushdown automaton was chosen over regex-based generation because regex cannot enforce balanced brackets, key ordering, or schema-level constraints (e.g., required keys must appear before closing `}`). The PDA's stack naturally handles nested objects and maintains `remaining_keys` for completeness checking.

### Hand-Written Clone

The PDA implements a manual `clone()` method (~30x faster than `copy.deepcopy`) that shares immutable fields (schema, state enum) and copies only mutable state (stack, remaining_keys, active_fsm). This was critical for the DFS performance: the C1 clone-storm fix reduced a single call from 331k deepcopies / ~11.5s to ~150k clones / ~0.72s inside string literals.

### FSM Primitives

Three finite-state machines handle leaf values:
- **NumberFSM** — validates integers and floats with proper state transitions for minus signs, leading zeros, decimal points, and terminators. Digits are matched against the ASCII set `0-9` only, so Unicode "digits" that `str.isdigit()` accepts can never leak into the output. Exponents (`1e3`) are intentionally unsupported.
- **StringLiteralFSM** — enforces JSON string rules including escape sequences and `\u` hex digits (ASCII hex only)
- **ExactMatchFSM** — filters candidates char-by-char for fixed vocabularies (function names, booleans, null, JSON keys)

## Performance

On the 11-prompt test suite with Qwen/Qwen3-0.6B:

| Metric | Value |
|---|---|
| Accuracy | 100% (11/11 prompts correct) |
| Valid JSON | 100% |
| Routing correctness | 11/11 |
| Number extraction (add + sqrt) | 4/4 |
| String extraction (greet + reverse) | 4/4 |
| Regex extraction | 3/3 |
| Total time | ~3 minutes on the campus evaluation machine (GPU; no KV cache) |

All 11 prompts are correct. The `fn_reverse_string` case is handled by a
scoped extraction instruction and example that are added only for functions
with a single string parameter, so the model extracts the argument instead of
computing the result. The official moulinette validates both exercise sets at
100%: **11/11 public** and **11/11 private**. CPU-only runs are slower because
the model has no KV cache. This table covers the main suite; the adversarial
edge fixtures show lower semantic accuracy while remaining 100% schema-valid
(see Testing Strategy).

## Challenges Faced

### C1 DFS Clone Storm

The original DFS implementation used `copy.deepcopy` for every PDA clone during trie traversal, resulting in 331k deepcopies (~11.5s per call) inside string literals. Fixed by:
- Pre-filtering trie edges with the non-mutating `accepts_char()` mask before cloning
- Implementing a hand-written `clone()` method (~30x faster than deepcopy)

### Token Trie Decoding

The trie must be built from the tokenizer's **decoded** token strings (with `Ġ` decoded to space, `Ċ` to newline). Originally built from raw token strings, causing a character-level mismatch between the trie and PDA. Fixed by using `tokenizer.decode()` to obtain the decoded vocabulary.

### Regex Overgeneralization

The model was generating `[0-9]+` for every regex field, even for literal word substitutions (e.g., replacing `'cat'` → `[0-9]+`). Fixed by adding explicit generic regex rules plus catalog-derived class/literal examples (numbers, vowels, and a literal word) to the extraction prompt.

### Reverse/Sqrt Computation

The model computes answers instead of extracting arguments for
`fn_reverse_string` (reverses the string) and `fn_get_square_root` (computes
the square root). A generic verbatim-extraction instruction plus a scoped
example — added only for functions with a single string parameter — fixes the
reverse case without hardcoding function names, while the same reminder keeps
the numeric cases extracting rather than computing.

## Testing Strategy

### Unit Tests

- **PDA tests**: structural states, key completeness, type enforcement, nested objects
- **FSM tests**: NumberFSM (integer/number modes, edge cases), StringLiteralFSM (escapes, hex), ExactMatchFSM (candidate filtering)
- **Trie tests**: insertion, lookup, build from vocab
- **DFS tests**: token masking correctness, clone count benchmarks
- **Schema compiler tests**: router/extractor table compilation, type mapping, array `ArraySpec` mapping
- **Array tests** (`test_arrays.py`): array schema validation, compiler mapping, PDA accept/reject (`[]`, `[1,2]`, `[1,]`, type mismatch)
- **Object tests** (`test_objects.py`): nested objects, arrays of objects, arrays of arrays, empty objects, per-scope key/type enforcement
- **Engine tests**: generation loop, fast-forward, token selection
- **CLI error-handling tests** (`test_main_errors.py`): load errors exit 2 before bootstrap, per-prompt failures become `none` placeholders while the run still exits 0, and startup exits 2

### End-to-End Tests

- Synthetic vocab e2e: scripted model with deterministic token ids to verify the full pipeline (trie → DFS → PDA → generator → decode)
- Real vocab benchmark: same pipeline with the actual 151k-token Qwen vocab (marked as benchmark, excluded from default suite)

### Adversarial Fixtures

Beyond pytest, the `data/input/edge_*.json` fixtures were run manually against
the real model:

| Fixture | Schema valid | Semantically correct |
|---|---|---|
| `edge_arrays` | 5/5 | 3/5 |
| `edge_objects` | 2/2 | 2/2 |
| `edge_regex_sandbox` | 5/5 | 2/5 |
| `edge_special_chars` | 5/5 | 3/5 |

Every output remained schema-valid JSON. The failures are model-level:
omitted regex brackets (`0-9+` instead of `[0-9]+`), complex
whitespace/newline prompts, a double-quoted reverse prompt, and literal text
containing regex metacharacters. The main 11-prompt suite remains 11/11.

### Moulinette Validation

The provided moulinette was used as an external end-to-end check
(`prepare_exercises` → run the CLI on the generated inputs →
`grade_student_answers`):

| Set | Score |
|---|---|
| public | 11/11 (100%) |
| private | 11/11 (100%) |

The private set exercises integer and boolean parameters, decimal compound
interest, SQL strings, backslash paths, and template braces/quotes.

### Invariant Tests

`test_automata_predicates.py` verifies the core invariant: `pda.accepts_char(c) == pda.clone().advance(c)` for every FSM and the PDA, ensuring the DFS mask stays in sync with the advance logic.

### Running Tests

```bash
pytest                        # run all tests (benchmarks excluded)
pytest tests/test_pda.py      # single file
pytest -k "pda"               # filter by keyword
pytest -m benchmark -s        # run benchmarks opt-in
```

## Known Limitations

- **No KV cache.** Each generation step re-runs the whole sequence through
  the model, so attention cost grows with the generated prefix (O(n²) per
  prompt) and the SDK returns two full vocab-sized Python lists per call.
  Only the single-candidate fast-forward skips the model entirely. This is
  why a full run takes minutes; the deterministic engine trades speed for
  the validity guarantee.
- **Prompt injection is mitigated, not eliminated.** The user request and
  function catalog are marked as untrusted data in the prompts, and output
  is grammar-constrained, so injected text cannot corrupt the JSON or force
  an unknown function. It can still bias *which* function is selected.
- **Byte-level BPE partial sequences.** The trie is built from
  `tokenizer.decode([id])` per token. For byte-level BPE, tokens that are
  partial UTF-8 sequences decode in isolation to the replacement character
  `U+FFFD`; distinct ids that decode to the same text collide and only one
  survives (`decoded_vocab` is last-wins). This does not affect JSON
  validity, but a fully correct treatment of multi-byte characters requires
  a byte-aware trie that tracks partial UTF-8 state, which is out of scope.
- **Regex extraction on adversarial prompts is best-effort.** The model
  sometimes drops the brackets in a character class (`0-9+` instead of
  `[0-9]+`), mishandles whitespace/newline literals, or does not escape
  regex metacharacters in a literal string. The 11-prompt suite passes, but
  these edge cases can produce semantically wrong (yet valid) parameters.
- **Out-of-catalog prompts are best-guess routed.** Because the router only
  offers real functions, a prompt that matches no available function is still
  routed to the closest one (and may then extract nonsense). `none` is
  reserved for prompts that could not be processed at all.
- **Double-quoted reverse prompts can still be reversed.** The
  transform-aware extraction guidance is a prompt-level mitigation. It
  reliably extracts verbatim for the main suite (single quotes) but the model
  can still compute the reversal for some double-quoted phrasings.

## Resources

- [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) — the underlying language model
- [Constrained Language Generation](https://arxiv.org/abs/2103.10343) — background on logit masking approaches
- [Outlines](https://github.com/outlines-dev/outlines) — reference implementation of structured generation (not used, studied for concepts)
- [Pydantic](https://docs.pydantic.dev/) — schema validation for function definitions and output

### AI Usage

AI was used throughout this project for:
- Prompt engineering for `src/engine/prompt_builder.py` and analysis of
  model outputs
- Debugging the PDA state transitions in `src/automata/pda.py` and the DFS
  clone-storm in `src/dfs/dfs.py`
- Reviewing the constrained-decoding design (`src/automata`, `src/trie`,
  `src/dfs`) and the error-handling code in `src/app.py` / `src/io_manager.py`
- Writing and polishing this README and `AGENTS.md`
- Generating edge-case tests under `tests/`

All AI-generated code was reviewed, tested, and understood before being committed. Peer review was conducted regularly to validate design decisions and catch blind spots.
