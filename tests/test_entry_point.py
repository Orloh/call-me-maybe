"""
Tests for the `python -m src` entry point's HF cache configuration.

`make` exports HF_HOME; a direct run must fall back to the repo-local
`.hf_cache`, and it must do so before huggingface_hub resolves its constants
at import time.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from src import __main__ as entry

_ROOT = Path(__file__).resolve().parent.parent
_HF_CACHE = _ROOT / ".hf_cache"


def test_configure_hf_cache_sets_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HF_HOME", raising=False)

    entry._configure_hf_cache()

    assert os.environ["HF_HOME"] == str(_HF_CACHE)


def test_configure_hf_cache_respects_existing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_HOME", "/custom/hf-cache")

    entry._configure_hf_cache()

    assert os.environ["HF_HOME"] == "/custom/hf-cache"


def test_configure_hf_cache_applies_before_huggingface_hub_import() -> None:
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE")
    }
    env["PYTHONPATH"] = str(_ROOT)

    code = (
        "import src.__main__ as m;"
        "m._configure_hf_cache();"
        "import huggingface_hub.constants as c;"
        "print(c.HF_HOME);"
        "print(c.HUGGINGFACE_HUB_CACHE)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    lines = result.stdout.strip().splitlines()
    assert lines[0] == str(_HF_CACHE)
    assert lines[1] == str(_HF_CACHE / "hub")
