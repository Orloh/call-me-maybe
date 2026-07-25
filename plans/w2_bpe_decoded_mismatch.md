# W2 Fix: BPE/Decoded Charset Mismatch

## Problem Summary

The trie and PDA operate on different character representations:

| Component | Representation | Example |
|---|---|---|
| **Trie** (built from `vocab.json`) | Raw BPE strings | `Ġ` (U+0120) for space |
| **PDA** (decoded text) | Decoded Unicode | `␣` (U+0020) for space |
| **DFS** (walks trie) | Raw BPE chars (`Ġ`) | feeds `pda.accepts_char('Ġ')` |
| **Generator** (`model.decode()`) | Decoded Unicode (` `) | feeds `pda.advance(' ')` |

This mismatch causes three concrete problems:

### Problem 1: Structural Whitespace Impossible (Silent)

`pda.py:63` WHITESPACE = `" \n\t\r"` — the PDA accepts literal space, newline, tab, carriage return as structural whitespace between JSON tokens. But the trie has **zero tokens containing literal space** (verified empirically: Qwen3-0.6B has 53,131 `Ġ`-prefixed tokens, **0** with literal space). The generator never emits whitespace between JSON tokens — output is always compact JSON like `{"a":5}`.

### Problem 2: Byte-Mapped Control Chars Crash Mid-Generation

`generator.py:140-142` — `_advance_pda` calls `self.pda.advance(char)` for each decoded character of a token. A token like `Ċ` (raw BPE for newline) passes the DFS clone check (inside string: `Ċ` U+010A ≥ 32, accepted by `accepts_char`). But `model.decode([Ċ_id])` returns `"\n"` (U+000A). When fed to `StringLiteralFSM.advance('\n')`, it hits `ILLEGAL_RAW_CHARS` (set of `chr(i)` for `i` in `range(32)`) and returns `False` → `RuntimeError("PDA rejected safely char")` → **generation crashes**.

### Problem 3: `allowed_characters()` Mask Disagrees with `advance()` (Latent)

Before the W1 fix, `StringLiteralFSM.allowed_characters()` for `INSIDE_STRING` returned ASCII-only `string.printable`, but `advance()` accepted any non-control character. A mask-based DFS pruner built on `allowed_characters()` would silently prune all `Ġ`-prefixed tokens inside strings, making spaces impossible.

---

## Solution: Rebuild Trie with Decoded Strings

Align the trie and PDA on the same character representation by building the trie from **decoded** token strings obtained via the HuggingFace tokenizer's `decode()` method.

The tokenizer (`Small_LLM_Model._tokenizer`) is the authoritative decoder — it knows that `Ġ` → ` `, `Ċ` → `\n`, etc. Using it to produce decoded strings for every token ID gives us a mapping that the PDA can process directly.

### Impact Summary

| Aspect | Before (Raw BPE Trie) | After (Decoded Trie) |
|---|---|---|
| Char in trie for space | `Ġ` (U+0120) | ` ` (U+0020) |
| Char in trie for newline | `Ċ` (U+010A) | `\n` (U+000A) |
| Structural whitespace | Impossible | Enabled (pretty JSON) |
| Control-char crash | Possible (inside strings) | Impossible (caugh at DFS time) |
| `model.decode()` per token | Required (for each step) | Eliminated (pre-built mapping) |

---

## Implementation Plan

### Phase 1: Build Decoded Vocabulary Mapping in `bootstrap.py`

**Goal**: Produce `token_id → decoded_string` mapping at startup, build trie with decoded strings.

```python
def initialize_system_dependencies() -> tuple[Small_LLM_Model, PrefixTrie, dict[int, str]]:
    model = Small_LLM_Model()

    # Load raw vocabulary
    vocab_path = model.get_path_to_vocab_file()
    with open(vocab_path, encoding="utf-8") as file:
        raw_vocab = json.load(file)

    # Build decoded vocabulary: skip empty/special tokens
    decoded_vocab: dict[str, int] = {}
    token_to_decoded: dict[int, str] = {}
    skip_ids = {
        model._tokenizer.bos_token_id,
        model._tokenizer.eos_token_id,
        model._tokenizer.pad_token_id,
    }

    for token_str, token_id in raw_vocab.items():
        if token_id in skip_ids:
            continue
        decoded = model.decode([token_id])
        if not decoded:
            continue
        decoded_vocab[decoded] = token_id
        token_to_decoded[token_id] = decoded

    # Build trie from decoded strings
    trie = PrefixTrie()
    trie.build_from_vocab(decoded_vocab)

    return model, trie, token_to_decoded
```

**Return signature changes**: `tuple[Small_LLM_Model, PrefixTrie]` → `tuple[Small_LLM_Model, PrefixTrie, dict[int, str]]`.

**Notes**:
- `model.decode` uses `self._tokenizer.decode(ids, skip_special_tokens=True)` from the existing SDK.
- The decoded text for byte-mapped control chars (`Ċ` → `\n`) is now part of the trie. The DFS will test `pda.accepts_char('\n')` — inside a string, `StringLiteralFSM` will correctly reject it (`\n` is in `ILLEGAL_RAW_CHARS`). The token is pruned at DFS time, not at decode time. No more crash.
- The decoded text for `Ġ` → ` ` is now in the trie. Structural whitespace becomes possible because ` ` is in the PDA's `WHITESPACE` set. Tokens that are purely whitespace (like a space-only BPE token) can now appear between structural elements.

### Phase 2: Trie Remains Unchanged

**File**: `src/trie/trie.py`

No changes needed. The `PrefixTrie` already works with any string representation. The `.insert()` method and `.build_from_vocab()` method are string-agnostic.

### Phase 3: Generator Uses Pre-Built Mapping

**File**: `src/engine/generator.py`

**Changes**:
1. Accept `token_to_decoded: dict[int, str]` in `__init__`
2. Use it instead of `self.model.decode([next_token_id])` in the generate loop

```python
class ConstrainedGenerator:
    def __init__(
        self,
        model: Small_LLM_Model,
        pda: JSONPushdownAutomaton,
        trie: PrefixTrie,
        token_to_decoded: dict[int, str],
        debug: bool = False
    ) -> None:
        self.model = model
        self.pda = pda
        self.trie = trie
        self.token_to_decoded = token_to_decoded
        self.tracer = GenerationTracer(enabled=debug)

    def generate(self, prompt: str, max_new_tokens: int = 500) -> str:
        # ... existing encode/tensor setup ...
        generated_text = ""
        for step in range(max_new_tokens):
            if self.pda.state == PDAState.TERMINAL:
                break
            allowed_ids = self._get_allowed_tokens()
            next_token_id = self._select_next_token(current_tokens, allowed_ids)
            current_tokens.append(next_token_id)

            new_text_chunk = self.token_to_decoded[next_token_id]   # was model.decode(...)
            generated_text += new_text_chunk

            # ... record pre-state, advance PDA, log step ...
```

**Efficiency gain**: The per-tensor `model.decode()` call is eliminated. A dict lookup replaces it.

### Phase 4: Pipeline Passes Mapping

**File**: `src/engine/pipeline.py`

**Changes**:
1. Accept `token_to_decoded` in `__init__`
2. Pass it to every `ConstrainedGenerator` constructor

```python
class FunctionCallingPipeline:
    def __init__(self, model, trie, token_to_decoded, available_functions, debug=False):
        self.model = model
        self.trie = trie
        self.token_to_decoded = token_to_decoded
        ...

    def process_prompt(self, user_prompt, available_functions):
        # Phase 1
        router_gen = ConstrainedGenerator(
            self.model, router_pda, self.trie,
            self.token_to_decoded, debug=self.debug
        )
        ...
        # Phase 2
        extractor_gen = ConstrainedGenerator(
            self.model, extractor_pda, self.trie,
            self.token_to_decoded, debug=self.debug
        )
```

### Phase 5: Update Entry Point

**File**: `src/__main__.py`

```python
def main() -> None:
    model, trie, token_to_decoded = initialize_system_dependencies()

    pipeline = FunctionCallingPipeline(
        model=model,
        trie=trie,
        token_to_decoded=token_to_decoded,
        available_functions=available_functions,
        debug=args.debug,
    )
```

### Phase 6: Update Tests

**Files affected**:

| File | Change |
|---|---|
| `tests/test_bootstrap.py` | Mock `model.decode` to return fixed strings; assert `token_to_decoded` is built; update return type |
| `tests/test_engine.py` | Pass mock `token_to_decoded` dict to `ConstrainedGenerator` |
| `tests/test_pipeline.py` | Pass mock `token_to_decoded` to `FunctionCallingPipeline` |
| `tests/test_dfs_benchmark.py` | **No change** — benchmark builds trie from raw vocab for reality anyway |
| `tests/test_e2e_constrained_json.py` | Update `_ScriptedModel.decode` to match new trie (decoded strings) |

**Key test**: Add a new test verifying that the decoded trie no longer contains `Ġ` as a character:

```python
def test_decoded_trie_no_raw_bpe_chars() -> None:
    """After W2 fix, the trie must contain decoded chars, not raw BPE."""
    trie, _ = build_decoded_trie()
    # Walk trie: no node should be Ġ
    assert b"\xc4\xa0".decode("utf-8") not in _collect_trie_chars(trie)
```

### Phase 7: Update AGENTS.md

**Changes**:
1. Remove W2 from "Known Limitations"
2. Add architecture note: "Trie built from decoded token strings using the tokenizer (not raw BPE)"
3. Add note about pretty-printed JSON becoming available
4. Update testing section if relevant

---

## Performance Considerations

### Startup Cost
- Building the decoded mapping: iterating 151,643 tokens and calling `model.decode([id])` for each
- Estimated: ~15–30 seconds for 150k tokens (each decode is a tokenizer lookup)
- **Mitigation**: Cache the decoded mapping to a file (e.g., `decoded_vocab.json`) so subsequent runs load it in milliseconds. Only regenerate when `vocab.json` changes.

### Generation Speedup
- Eliminates `model.decode()` call per generation step (dict lookup replaces it)
- At ~300 tokens per prompt, that's ~300 fewer tokenizer calls

---

## Edge Cases

### Case 1: Multiple Tokens Decode to the Same String

Two different token IDs may decode to the same string (common in BPE — e.g., `a` and `Ġa` both decode to `a` after the space is stripped? No — `Ġa` decodes to ` a` with a leading space, which is different from `a`). But there may be true duplicates. The trie's `insert()` method keeps only the first `token_id` for each string, which means the second ID's version is lost. The model can never generate the second version.

**Impact**: Acceptable — if two tokens produce identical decoded text, either one is fine for the grammar. The model's logit for the surviving token represents both.

### Case 2: Empty Decoded Strings

Some tokens (especially padding or control tokens) may decode to `""`. These are skipped in bootstrap (no trie insertion). The model can never generate them, which is correct — an empty-string token adds nothing to the generated text.

### Case 3: Special Token IDs

BOS (`<s>`), EOS (`</s>`), PAD tokens are skipped. The model shouldn't be generating these in the middle of JSON anyway — the PDA grammar wouldn't allow them (they don't match structural or string characters). The skip is a safety layer.

### Case 4: Tokens with Only Control Characters

A token that decodes to, say, `"\t"` alone (a single tab). Inside a JSON string, `\t` is in `ILLEGAL_RAW_CHARS` → rejected by DFS. In structural position, `\t` is in `WHITESPACE` → accepted (structural whitespace). So this token can now appear as "pretty-printing" whitespace between JSON tokens — a new capability enabled by the fix.

---

## Rollback Strategy

If the fix introduces unexpected behavior:
1. Revert `bootstrap.py` to building trie from raw vocab
2. Remove `token_to_decoded` parameter from generator/pipeline
3. Restore `model.decode()` in the generator loop
4. Revert AGENTS.md changes

This only takes effect after a restart (the trie is built once at startup), so there's no risk of corrupting state mid-run.

---

## Concrete Work Order

1. `src/engine/generator.py` — add `token_to_decoded` param, swap lookup
2. `src/engine/pipeline.py` — accept and pass-through the new param
3. `src/engine/bootstrap.py` — build decoded mapping, build decoded trie
4. `src/__main__.py` — adapt to new return signature
5. Update tests
6. Remove W2 from AGENTS.md Known Limitations
7. Run full lint + test suite
