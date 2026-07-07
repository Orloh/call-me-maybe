import math
from src.automata.pda import JSONPushdownAutomaton, PDAState
from src.trie.trie import PrefixTrie
from src.dfs.dfs import find_allowed_tokens
from llm_sdk import Small_LLM_Model

class ConstraindeGenerator:
    """
    Orchestrates the determinsitic generation of a JSON by coupling a
    Language Model with a gramatical PDA and a Prefix Trie.
    """
    def __init__(
        self,
        model: Small_LLM_Model,
        pda: JSONPushdownAutomaton,
        trie: PrefixTrie,
    ):
        self.model = model
        self.pda = pda
        self.trie = trie

    def generate(self, prompt: str, max_new_tokens: int = 500) -> str:
        generate_text = ""
        #TODO
        return generated_text
