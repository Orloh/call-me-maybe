from src.automata.pda import JSONPushdownAutomaton, PDAState
from src.trie import PrefixTrie
from src.dfs import find_allowed_tokens
from src.utils import GenerationTracer
from llm_sdk import Small_LLM_Model


class MaxTokensExceededError(RuntimeError):
    """
    Raised when the generator exhausts `max_new_tokens` without the PDA
    reaching a terminal state. The partial text is attached as `partial_text`.
    """
    def __init__(
        self,
        partial_text: str,
        max_new_tokens: int,
        final_state: PDAState
    ) -> None:
        self.partial_text = partial_text
        self.max_new_tokens = max_new_tokens
        self.final_state = final_state
        super().__init__(
            "Max tokens ({max_new_tokens}) exceeded without reaching "
            "TERMINAL. Final PDA state: {final_state.name}. "
            "Partial text: {partial_text!r}".format(
                max_new_tokens=max_new_tokens,
                final_state=final_state,
                partial_text=partial_text,
            )
        )


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
        token_to_decoded: dict[int, str],
        stop_tokens: set[int] | None = None,
        debug: bool = False
    ) -> None:
        self.model = model
        self.pda = pda
        self.trie = trie
        self.token_to_decoded = token_to_decoded
        self.stop_tokens = (
            frozenset(stop_tokens) if stop_tokens else frozenset()
        )
        self.tracer = GenerationTracer(enabled=debug)

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

        self.tracer.start_trace(prompt)

        for step in range(max_new_tokens):
            if self.pda.state == PDAState.TERMINAL:
                break

            allowed_ids = self._get_allowed_tokens()
            next_token_id = self._select_next_token(
                current_tokens,
                allowed_ids
            )

            if next_token_id in self.stop_tokens:
                break

            current_tokens.append(next_token_id)
            new_text_chunk = self.token_to_decoded[next_token_id]
            generated_text += new_text_chunk

            pda_before = self.pda.state
            fsm_before = self.pda.active_fsm

            self._advance_pda(new_text_chunk)

            self.tracer.log_step(
                step=step + 1,
                token=new_text_chunk,
                pda_before=pda_before,
                pda_after=self.pda.state,
                fsm_before=fsm_before,
                fsm_after=self.pda.active_fsm,
                keys_left=len(self.pda.remaining_keys)
            )

        self.tracer.end_trace()
        if self.pda.state != PDAState.TERMINAL:
            raise MaxTokensExceededError(
                partial_text=generated_text,
                max_new_tokens=max_new_tokens,
                final_state=self.pda.state,
            )
        return generated_text

    def _get_allowed_tokens(self) -> list[int]:
        """Queries the DFS to find gramatically valid next tokens."""
        allowed_ids = find_allowed_tokens(self.trie.root, self.pda)
        if not allowed_ids:
            raise RuntimeError(
                "Grammar deadlock: "
                "The PDA rejected all possible next tokens."
            )
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

        best_token_id = max(
            allowed_ids,
            key=lambda vocab_id: raw_logits[vocab_id]
        )

        return best_token_id

    def _advance_pda(self, text_chunk: str) -> None:
        """
        Advances the internal state machine with the newly generated
        characters.
        """
        for char in text_chunk:
            if not self.pda.advance(char):
                raise RuntimeError(
                    f"PDA rejected safely char '{char}'. Check DFS logic!"
                )
