from typing import Any


class GenerationTracer:
    """Prints per-token PDA/FSM state transitions when tracing is enabled."""

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled

    @staticmethod
    def _get_name(obj: Any) -> str:
        if obj is None:
            return "None"

        if hasattr(obj, "name") and isinstance(obj.name, str):
            return obj.name

        if hasattr(obj, "__class__"):
            return str(obj.__class__.__name__)

        return str(obj)

    def _emit(self, text: str) -> None:
        try:
            print(text)
        except BrokenPipeError:
            # A closed downstream reader (e.g. `... | head`) must not be
            # mistaken for a generation failure; disable tracing instead.
            self.enabled = False

    def start_trace(self, prompt: str) -> None:
        """Emits the trace header for a generation run."""
        if not self.enabled:
            return
        self._emit(
            f"\n{'-'*90}\n"
            f"🔍 TRACE START"
            f"\n{'-'*90}\n"
            f"{prompt}"
            f"\n{'-'*90}\n"
        )

    def log_step(
        self,
        step: int,
        token: str,
        token_id: int,
        pda_before: Any,
        pda_after: Any,
        fsm_before: Any,
        fsm_after: Any,
        keys_left: int
    ) -> None:
        """Emits one generation-step trace line."""
        if not self.enabled:
            return
        p_before = self._get_name(pda_before)
        p_after = self._get_name(pda_after)
        f_before = self._get_name(fsm_before)
        f_after = self._get_name(fsm_after)

        self._emit(
            f"[{step:02d} Token{repr(token):<8} ID:{token_id:<6} | "
            f"PDA: {p_before:<24} -> {p_after:<24} | "
            f"FSM: {f_before:<16} -> {f_after:<16} | "
            f"Keys Left: {keys_left}"
        )

    def end_trace(self) -> None:
        """Emits the trace footer for a generation run."""
        if not self.enabled:
            return
        self._emit(
            f"\n{'-'*90}\n"
            "✅ TRACE COMPLETE"
            f"\n{'-'*90}\n"
        )
