import math
from src.automata.pda import JSONPushdownAutomaton, PDAState
from src.trie.trie import PrefixTrie
from src.dfs.dfs import find_allowed_tokens
from llm_sdk import Small_LLM_Model

class ConstrainedGenerator:
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
        """
        Main execution loop for constrained decoding
        """
        current_tokens = self.model.tokenize(prompt)
        generated_text = ""

        for _ in range(max_new_tokens):
            if self.pda.state == PDAState.TERMINAL:
                break
            
            allowed_ids = self._get_allowed_tokens()
            next_token_id = self._select_next_token(current_tokens, allowed_ids)

            current_tokens.append(next_token_id)
            new_text_chunk = self.model.decode([next_token_id])
            generated_text += new_text_chunk

            self._advance_pda(new_text_chunk)

        return generated_text

    def _get_allowed_tokens(self) -> list[int]:
        """Queries the DFS to find gramatically valid next tokens."""
        allowed_ids = find_allowed_tokens(self.trie.root, self.pda)
        if not allowed_ids:
            raise RuntimeError("Grammar deadlock: The PDA rejected all possible next tokens.")
        return allowed_ids
    
    def _apply_logits_mask(
            self,
            logits: list[float],
            allowed_ids: list[int]
    ) -> list[float]:
        """
        Sets the probability of all disallowed tokens to negative infinity.
        """
        allowed_set = set(allowed_ids)
        for vocab_id in range(len(logits)):
            if vocab_id not in allowed_set:
                logits[vocab_id] = -math.inf
        return logits
