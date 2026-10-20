from os import environ
from pathlib import Path


def _configure_hf_cache() -> None:
    """Use the repo-local HF cache when run without `make`.

    `make` exports HF_HOME; a direct `python -m src` does not. This must run
    before transformers/huggingface_hub are imported, since they resolve
    HF_HOME at import time. An explicitly-set HF_HOME still wins.
    """
    default = Path(__file__).resolve().parent.parent / ".hf_cache"
    environ.setdefault("HF_HOME", str(default))


if __name__ == "__main__":
    _configure_hf_cache()
    from src.app import main

    main()
