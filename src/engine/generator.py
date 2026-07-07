import math
from src.automata.pda import JSONPushdownAutomaton, PDAState
from src.trie.trie import PrefixTrie
from src.dfs.dfs import find_allowed_tokens
from llm_sdk import Small_LLM_Model

def generate_constrained_json(
    prompt: str,
    model: Small_LLM_Model,
    pda: JSONPushdownAutomaton,
    trie: PrefixTrie,
    max_new_tokens: int = 500
) -> str:
    """
    Manages the custom decoding loop, enforcing grammar contraints via the PDA
    and fast-forwarding when only a single token is mathematically possible.
    """
    generated_text = ""
    # TODO
    return generated_text
