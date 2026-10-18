import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path

from llm_sdk import Small_LLM_Model
from src.trie import PrefixTrie

logger = logging.getLogger(__name__)

# Decoded token maps are expensive to rebuild (one tokenizer.decode call per
# vocab entry, ~150k for Qwen). They are deterministic for a given vocab file,
# so cache them under the project root, keyed by path + mtime + size.
_DEFAULT_CACHE_DIR = (
    Path(__file__).resolve().parents[2] / ".cache" / "decoded_vocab"
)


def _vocab_cache_path(cache_dir: Path, vocab_path: str) -> Path | None:
    """
    Returns the cache file for `vocab_path`, or None when it cannot be
    stat-ed (e.g. a fake path in tests), in which case caching is skipped.
    """
    try:
        stat = Path(vocab_path).stat()
    except OSError:
        return None

    token = f"{vocab_path}:{stat.st_mtime_ns}:{stat.st_size}"
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()[:32]
    return cache_dir / f"{digest}.json"


def _build_decoded_vocab(
    model: Small_LLM_Model, raw_vocab: dict[str, int]
) -> tuple[dict[str, int], dict[int, str]]:
    decoded_vocab: dict[str, int] = {}
    token_to_decoded: dict[int, str] = {}
    for token_id in raw_vocab.values():
        decoded = model.decode([token_id])
        if not decoded:
            continue
        decoded_vocab[decoded] = token_id
        token_to_decoded[token_id] = decoded
    return decoded_vocab, token_to_decoded


def _load_cached_vocab(
    cache_file: Path,
) -> tuple[dict[str, int], dict[int, str]] | None:
    try:
        data = json.loads(cache_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None

    if not isinstance(data, dict):
        return None
    raw_decoded = data.get("decoded_vocab")
    raw_by_id = data.get("token_to_decoded")
    if not isinstance(raw_decoded, dict) or not isinstance(raw_by_id, dict):
        return None

    decoded_vocab: dict[str, int] = {}
    token_to_decoded: dict[int, str] = {}
    try:
        for text, token_id in raw_decoded.items():
            if not isinstance(text, str) or not isinstance(token_id, int):
                return None
            decoded_vocab[text] = token_id
        for token_id, text in raw_by_id.items():
            if not isinstance(token_id, str) or not isinstance(text, str):
                return None
            token_to_decoded[int(token_id)] = text
    except (TypeError, ValueError):
        return None

    return decoded_vocab, token_to_decoded


def _write_cached_vocab(
    cache_file: Path,
    decoded_vocab: dict[str, int],
    token_to_decoded: dict[int, str],
) -> None:
    """Best-effort atomic write; caching must never break startup."""
    tmp_name: str | None = None
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "decoded_vocab": decoded_vocab,
            "token_to_decoded": {
                str(token_id): text
                for token_id, text in token_to_decoded.items()
            },
        }
        fd, tmp_name = tempfile.mkstemp(
            dir=str(cache_file.parent),
            prefix=f".{cache_file.name}.",
            suffix=".tmp",
        )
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
        os.replace(tmp_name, cache_file)
        tmp_name = None
    except OSError as error:
        logger.warning(
            "Could not write decoded-vocab cache %s: %s", cache_file, error
        )
    finally:
        if tmp_name is not None:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass


def initialize_system_dependencies(
    cache_dir: Path | None = None,
) -> tuple[Small_LLM_Model, PrefixTrie, dict[int, str]]:
    """
    Handles all the heavy lifting of booting up the model and
    the in-memory Trie.

    Returns (model, trie, token_to_decoded) where token_to_decoded
    is a mapping from integer token ids to their decoded Unicode strings,
    used by the generator to feed decoded text to the PDA.

    The decoded vocabulary (decoded) is cached on disk (see
    `_DEFAULT_CACHE_DIR`) and reused across runs while the vocab file is
    unchanged; pass `cache_dir` to relocate or isolate the cache.
    """
    logger.info("Initializing language model...")
    model = Small_LLM_Model()

    logger.info("Loading and parsing model vocabulary...")
    vocab_path = model.get_path_to_vocab_file()

    resolved_cache_dir = (
        cache_dir if cache_dir is not None else _DEFAULT_CACHE_DIR
    )
    cache_file = _vocab_cache_path(resolved_cache_dir, vocab_path)

    cached = _load_cached_vocab(cache_file) if cache_file else None
    if cached is not None:
        decoded_vocab, token_to_decoded = cached
        logger.info("Loaded decoded vocabulary from cache: %s", cache_file)
    else:
        with open(vocab_path, "r", encoding="utf-8") as file:
            raw_vocab: dict[str, int] = json.load(file)

        logger.info("Building decoded vocabulary mapping...")
        decoded_vocab, token_to_decoded = _build_decoded_vocab(
            model, raw_vocab
        )
        if cache_file is not None:
            _write_cached_vocab(cache_file, decoded_vocab, token_to_decoded)

    shadowed = len(token_to_decoded) - len(decoded_vocab)
    if shadowed:
        logger.info(
            "%d token id(s) decode to text already mapped to another id; "
            "the last id wins and the earlier ids are unreachable.",
            shadowed,
        )

    logger.info(
        f"Building Prefix Trie from {len(decoded_vocab)} decoded tokens..."
    )
    trie = PrefixTrie()
    trie.build_from_vocab(decoded_vocab)
    logger.info(
        f"Prefix Trie built succesfully with {trie.size} tokens."
    )

    return model, trie, token_to_decoded
