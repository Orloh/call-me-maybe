import math
from src.automata.pda import JSONPushdownAutomaton, PDAState
from src.trie import PrefixTrie
from src.dfs import find_allowed_tokens
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
        encoded_tensor = self.model.encode(prompt)
        
        if encoded_tensor.dim() == 2:
            current_tokens = encoded_tensor[0].tolist()
        else:
            current_tokens = encoded_tensor.tolist()

        generated_text = ""

        for _ in range(max_new_tokens):
            if self.pda.state == PDAState.TERMINAL:
                break
            
            allowed_ids = self._get_allowed_tokens()
            next_token_id = self._select_next_token(current_tokens, allowed_ids)

            current_tokens.append(next_token_id)
            new_text_chunk = self.model.decode([next_token_id])
            generated_text += new_text_chunk

            print(f"'{new_text_chunk}'")
            
            self._advance_pda(new_text_chunk)

        print(f"{generated_text}")
        return generated_text

    def _get_allowed_tokens(self) -> list[int]:
        """Queries the DFS to find gramatically valid next tokens."""
        allowed_ids = find_allowed_tokens(self.trie.root, self.pda)
        if not allowed_ids:
            raise RuntimeError("Grammar deadlock: The PDA rejected all possible next tokens.")
        return allowed_ids
    
    def _select_next_token(
            self,
            current_tokens: list[int],
            allowed_ids: list[int]
    ) -> int:
        """
        Selects the next token using Fast-Forwarding if possible,
        otherwise falls back to masked LLM generation.
        """
        if len(allowed_ids) == 1:
            return allowed_ids[0]

        raw_logits = self.model.get_logits_from_input_ids(current_tokens)

        best_token_id = max(allowed_ids, key=lambda vocab_id: raw_logits[vocab_id])

        return best_token_id

    def _advance_pda(self, text_chunk: str) -> None:
        """
        Advances the internal state machine with the newly generated characters.
        """
        for char in text_chunk:
            if not self.pda.advance(char):
                raise RuntimeError(f"PDA rejected safely char '{char}'. Check DFS logic!")
