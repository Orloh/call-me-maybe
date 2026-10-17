*This project has been created as part of the 42 curriculum by ohernan.*

# call-me-maybe

## Description

call-me-maybe is a from-scratch function-calling engine for Small Language Models (SLMs). It uses constrained decoding with a pushdown automaton (PDA) and token tries to guarantee 100% schema-valid JSON output from a 0.6B parameter model (Qwen/Qwen3-0.6B).

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

This runs `uv sync --link-mode copy`, which installs dependencies and configures the local `llm_sdk` package as an editable dependency.

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

Or with custom paths:

```bash
CUDA_VISIBLE_DEVICES="" uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calling_results.json \
    --errors_output data/output/function_calling_errors.json
```

`--errors_output` is optional: when omitted it defaults to a sibling of
`--output` (e.g. `results.json` → `results_errors.json`).

### Error Handling & Exit Codes

The CLI is best-effort per prompt: a failure on one prompt is recorded to
the errors file and the run continues. Exit codes are:

| Code | Meaning |
|---|---|
| 0 | All prompts produced a result |
| 1 | Ran, but at least one prompt failed (see errors file) |
| 2 | Startup failure: missing/invalid input, schema error, or bootstrap/IO error |
| 130 | Interrupted (Ctrl-C); partial results were saved |

Inputs are loaded **before** the model is bootstrapped, so invalid input
fails fast without paying for model startup. The results file keeps the
same shape as always (successful calls only); failures are written to a
separate `function_calling_errors.json`.

Directory or non-UTF-8 input files, and output paths whose parent is a
regular file, are reported as exit 2 rather than crashing. Ctrl-C saves
partial results (and any recorded failures) before exiting 130.

### Debug Mode

```bash
make debug
```

Enables per-token PDA and FSM state tracing to the terminal, showing every generation step and the remaining schema keys.

### Linting

```bash
make lint
```

Runs flake8 and mypy with strict settings.

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
5. **Select** the max-logit token from the allowed set, or fast-forward when only one token is valid

This guarantees that every output token contributes to valid, schema-compliant JSON.

### Pushdown Automaton (PDA)

The PDA enforces JSON structure using a `Scope` stack (OBJECT/ARRAY) and a state machine with states like `EXPECTING_KEY`, `EXPECTING_COLON`, `EXPECTING_VALUE`, and `EXPECTING_COMMA_OR_END`. It tracks `remaining_keys` per nesting level to enforce key completeness and reject unknown keys. The `accepts_char()` method mirrors `advance()` without mutation, enabling the DFS to pre-filter trie edges cheaply.

### Prefix Trie

The trie stores the model's entire vocabulary as character-level paths. It is built once at bootstrap from the tokenizer's decoded token strings. The DFS walks the trie root, checking each edge's character against the PDA mask, and collects terminal token IDs.

## Design Decisions

### Two-Phase Pipeline

Routing and extraction are separated into two generation calls with distinct PDAs. This keeps each grammar tight: the router PDA only accepts `{"name": "fn_x"}` or `{"name": "none"}`, while the extractor PDA only accepts the chosen function's parameter schema. A single-shot approach would require a more complex PDA supporting nested objects.

### Arrays of Scalars

Parameters may be arrays of scalars, declared JSON-Schema style:

```json
{
  "numbers": { "type": "array", "items": { "type": "number" } }
}
```

The extractor PDA accepts `[]`, `[v]`, and `[v,v,...]` for any scalar element
type (`number`, `string`, `boolean`, `null`). Empty arrays are allowed and
there is no item-count bound. Nested arrays are intentionally rejected at
schema validation, so the PDA never needs a stack of element types. A new
`EXPECTING_ARRAY_START` state and the previously dead `Scope.ARRAY` frame now
carry the element FSM; `array_just_opened` distinguishes an empty `[]` from a
rejected trailing comma (`[1,]`).

### Numeric Output Coercion

The grammar accepts both `6` and `6.0` for a `number` parameter. To match the
expected output format, `number`/`float` parameters are coerced to `float`
after extraction (recursing into array elements), so `2` is emitted as `2.0`.
`integer` parameters remain ints.

### PDA over Regex

A pushdown automaton was chosen over regex-based generation because regex cannot enforce balanced brackets, key ordering, or schema-level constraints (e.g., required keys must appear before closing `}`). The PDA's stack naturally handles nested objects and maintains `remaining_keys` for completeness checking.

### Hand-Written Clone

The PDA implements a manual `clone()` method (~30x faster than `copy.deepcopy`) that shares immutable fields (schema, state enum) and copies only mutable state (stack, remaining_keys, active_fsm). This was critical for the DFS performance: the C1 clone-storm fix reduced a single call from 331k deepcopies / ~11.5s to ~150k clones / ~0.72s inside string literals.

### FSM Primitives

Three finite-state machines handle leaf values:
- **NumberFSM** — validates integers and floats with proper state transitions for minus signs, leading zeros, decimal points, and terminators
- **StringLiteralFSM** — enforces JSON string rules including escape sequences and `\u` hex digits
- **ExactMatchFSM** — filters candidates char-by-char for fixed vocabularies (function names, booleans, null, JSON keys)

## Performance

On the 11-prompt test suite with Qwen/Qwen3-0.6B:

| Metric | Value |
|---|---|
| Accuracy | 91% (10/11 prompts correct) |
| Valid JSON | 100% |
| Routing correctness | 11/11 |
| Regex extraction | 3/3 |
| Number extraction | 4/4 |
| String extraction | 2/4 (model limitation: reverses instead of extracting verbatim) |
| Sqrt extraction | 2/2 (fixed via few-shot examples) |
| Total time | ~5 minutes (no KV cache) |

The 91% accuracy meets the 90%+ target. The two remaining failures are both `fn_reverse_string` prompts where the 0.6B model cannot separate the function's purpose ("reverse a string") from the extraction task ("extract the input verbatim"). This is a known model-scale limitation.

## Challenges Faced

### C1 DFS Clone Storm

The original DFS implementation used `copy.deepcopy` for every PDA clone during trie traversal, resulting in 331k deepcopies (~11.5s per call) inside string literals. Fixed by:
- Pre-filtering trie edges with the non-mutating `accepts_char()` mask before cloning
- Implementing a hand-written `clone()` method (~30x faster than deepcopy)

### Token Trie Decoding

The trie must be built from the tokenizer's **decoded** token strings (with `Ġ` decoded to space, `Ċ` to newline). Originally built from raw token strings, causing a character-level mismatch between the trie and PDA. Fixed by using `tokenizer.decode()` to obtain the decoded vocabulary.

### Regex Overgeneralization

The model was generating `[0-9]+` for every regex field, even for literal word substitutions (e.g., replacing `'cat'` → `[0-9]+`). Fixed by adding explicit regex rules in the extraction prompt with few-shot examples covering both class-based patterns and literal words.

### Reverse/Sqrt Computation

The model computes answers instead of extracting arguments for `fn_reverse_string` (reverses the string) and `fn_get_square_root` (computes the square root). Partially mitigated with function-specific counter-examples and a verbatim extraction instruction. The reverse case remains at the model's 0.6B ceiling.

## Testing Strategy

### Unit Tests

- **PDA tests**: structural states, key completeness, type enforcement, nested objects
- **FSM tests**: NumberFSM (integer/number modes, edge cases), StringLiteralFSM (escapes, hex), ExactMatchFSM (candidate filtering)
- **Trie tests**: insertion, lookup, build from vocab
- **DFS tests**: token masking correctness, clone count benchmarks
- **Schema compiler tests**: router/extractor table compilation, type mapping, array `ArraySpec` mapping
- **Array tests** (`test_arrays.py`): array schema validation, compiler mapping, PDA accept/reject (`[]`, `[1,2]`, `[1,]`, type mismatch)
- **Engine tests**: generation loop, fast-forward, token selection
- **CLI error-handling tests** (`test_main_errors.py`): load errors exit 2 before bootstrap, per-prompt failures are isolated and recorded, exit codes 0/1/2

### End-to-End Tests

- Synthetic vocab e2e: scripted model with deterministic token ids to verify the full pipeline (trie → DFS → PDA → generator → decode)
- Real vocab benchmark: same pipeline with the actual 151k-token Qwen vocab (marked as benchmark, excluded from default suite)

### Invariant Tests

`test_automata_predicates.py` verifies the core invariant: `pda.accepts_char(c) == pda.clone().advance(c)` for every FSM and the PDA, ensuring the DFS mask stays in sync with the advance logic.

### Running Tests

```bash
pytest                        # run all tests (benchmarks excluded)
pytest tests/test_pda.py      # single file
pytest -k "pda"               # filter by keyword
pytest -m benchmark -s        # run benchmarks opt-in
```

## Resources

- [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) — the underlying language model
- [Constrained Language Generation](https://arxiv.org/abs/2103.10343) — background on logit masking approaches
- [Outlines](https://github.com/outlines-dev/outlines) — reference implementation of structured generation (not used, studied for concepts)
- [Pydantic](https://docs.pydantic.dev/) — schema validation for function definitions and output

### AI Usage

AI was used throughout this project for:
- Prompt engineering iteration and analysis of model outputs
- Debugging PDA state transition and DFS clone-storm issues
- Documentation and README writing
- Test generation for edge cases

All AI-generated code was reviewed, tested, and understood before being committed. Peer review was conducted regularly to validate design decisions and catch blind spots.
