import json
import logging
from llm_sdk import Small_LLM_Model
from src.trie import PrefixTrie

logger = logging.getLogger(__name__)


def initialize_system_dependencies() -> tuple[
    Small_LLM_Model, PrefixTrie, dict[int, str]
]:
    """
    Handles all the heavy lifting of bootin up the model and
    the in-memory Trie

    Returns (model, trie, token_to_decoded) where token_to_decoded
    is a mapping from integer token ids to their decoded Unicode strings,
    used by the generator to feed decoded text to the PDA.
    """
    logger.info("Initializing language model...")
    model = Small_LLM_Model()

    logger.info("Loading and parsing model vocabulary...")
    vocab_path = model.get_path_to_vocab_file()
    with open(vocab_path, "r", encoding="utf-8") as file:
        raw_vocab: dict[str, int] = json.load(file)

    logger.info("Building decoded vocabulary mapping...")
    decoded_vocab: dict[str, int] = {}
    token_to_decoded: dict[int, str] = {}

    for token_str, token_id in raw_vocab.items():
        decoded = model.decode([token_id])
        if not decoded:
            continue
        decoded_vocab[decoded] = token_id
        token_to_decoded[token_id] = decoded

    logger.info(
        f"Building Prefix Trie from {len(decoded_vocab)} decoded tokens..."
    )
    trie = PrefixTrie()
    trie.build_from_vocab(decoded_vocab)
    logger.info(
        f"Prefix Trie built succesfully with {trie.size} tokens."
    )

    return model, trie, token_to_decoded
