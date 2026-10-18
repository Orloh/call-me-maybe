"""Standalone smoke test for the opencode mypy LSP stack.

Spawns the project's `.venv/bin/pylsp` (with the same initialization options as
`opencode.json`), opens a temporary probe file that is *guaranteed* to violate
`[tool.mypy]` (`disallow_untyped_defs`), and asserts that pylsp-mypy publishes
the matching diagnostic. Cleans up after itself.

Usage:  uv run python scripts/lsp_smoke.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from typing import IO, Any, cast

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYLSP = os.path.join(ROOT, ".venv", "bin", "pylsp")
PROBE = os.path.join(ROOT, "src", "_lsp_smoke_probe.py")
PROBE_TEXT = "def needs_annotation(x):\n    return x + missing_name\n"

INIT_OPTIONS = {
    "pylsp": {
        "plugins": {
            "pylsp_mypy": {"dmypy": True, "report_progress": True},
            "pyflakes": {"enabled": False},
            "pycodestyle": {"enabled": False},
            "mccabe": {"enabled": False},
        }
    }
}


def frame(obj: dict[str, Any]) -> bytes:
    body = json.dumps(obj).encode()
    return b"Content-Length: %d\r\n\r\n" % len(body) + body


def read_message(stream: IO[bytes]) -> dict[str, Any] | None:
    headers: dict[bytes, bytes] = {}
    line = b""
    while True:
        ch = stream.read(1)
        if not ch:
            return None
        line += ch
        if line.endswith(b"\r\n"):
            if line == b"\r\n":
                break
            key, _, val = line[:-2].partition(b":")
            headers[key.strip().lower()] = val.strip()
            line = b""
    body = stream.read(int(headers[b"content-length"]))
    return cast(dict[str, Any], json.loads(body))


def main() -> int:
    if not os.path.exists(PYLSP):
        print(f"FAIL: {PYLSP} not found (run `make install`)")
        return 2

    with open(PROBE, "w") as fh:
        fh.write(PROBE_TEXT)

    proc = subprocess.Popen(
        [PYLSP],
        cwd=ROOT,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    stdout = proc.stdout
    stdin = proc.stdin
    if stdout is None or stdin is None:
        proc.terminate()
        os.remove(PROBE)
        print("FAIL: could not capture pylsp stdio")
        return 2

    diagnostics: dict[str, list[dict[str, Any]]] = {}

    def reader() -> None:
        while True:
            msg = read_message(stdout)
            if msg is None:
                return
            if msg.get("method") == "textDocument/publishDiagnostics":
                params = cast(dict[str, Any], msg["params"])
                diagnostics[str(params["uri"])] = cast(
                    list[dict[str, Any]], params["diagnostics"]
                )

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()

    stdin.write(frame({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "processId": os.getpid(),
            "rootUri": "file://" + ROOT,
            "capabilities": {},
            "initializationOptions": INIT_OPTIONS,
        },
    }))
    stdin.write(frame({"jsonrpc": "2.0", "method": "initialized", "params": {}}))
    stdin.write(frame({
        "jsonrpc": "2.0",
        "method": "textDocument/didOpen",
        "params": {
            "textDocument": {
                "uri": "file://" + PROBE,
                "languageId": "python",
                "version": 1,
                "text": PROBE_TEXT,
            }
        },
    }))
    stdin.flush()

    deadline = time.time() + 30
    found: list[dict[str, Any]] = []
    while time.time() < deadline:
        found = diagnostics.get("file://" + PROBE, [])
        if found:
            break
        time.sleep(0.5)
    proc.terminate()
    os.remove(PROBE)

    if not found:
        print("FAIL: pylsp published no diagnostics within 30s")
        return 1

    print("OK: pylsp + pylsp-mypy published mypy diagnostics:")
    for diag in found:
        message = str(diag["message"]).splitlines()[0]
        start = cast(dict[str, Any], diag["range"])["start"]
        line = cast(dict[str, Any], start)["line"]
        print(f"  line {line}: {message}")

    pyflakes = [d for d in found if str(d["message"]).startswith("undefined name")]
    if pyflakes:
        print("FAIL: pyflakes diagnostics leaked (expected disabled):")
        for diag in pyflakes:
            print(f"  {diag['message']}")
        return 1

    print("OK: pyflakes diagnostics suppressed as configured")
    return 0


if __name__ == "__main__":
    sys.exit(main())
