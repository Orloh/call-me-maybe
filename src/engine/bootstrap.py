import json
import logging
from pathlib import Path
from llm_sdk import Small_LLM_Model
from src.trie import PrefixTrie

logger = logging.getLogger(__name__)

def initialize_system_dependencies() -> tuple[Small_LLM_Model, PrefixTrie]:
    """
    Handles all the heavy lifting of bootin up the model and 
    the in-memory Trie
    """
    logger.info("Initializing language model...")
    model = Small_LLM_Model()

    logger.info("Loading and parsing model vocabulary...")
    vocab_path = model.get_path_to_vocab_file()
    with open(vocab_path, "r", encoding="utf-8") as file:
        vocabulary = json.load(file)

    logger.info("Building Prefix Trie from vocabulary...")
    trie = PrefixTrie()
    trie.build_from_vocab(vocabulary)
    logger.info(
        f"Prefix Trie built succesfully with {trie.size} tokens."
    )

    return model, trie
