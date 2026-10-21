"""
Error-handling tests for the testable CLI entry logic in src/app.py.

Heavy dependencies (model bootstrap, pipeline) are monkeypatched away so
these run without torch/transformers.
"""
import json
import signal
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src import app
from src.app import (
    EXIT_STARTUP_ERROR,
    EXIT_SUCCESS,
    RunConfig,
    parse_args,
    run,
)
from src.schema import FunctionCallResult

_DATA_INPUT = Path(__file__).resolve().parent.parent / "data" / "input"
_ARRAY_PROMPTS = _DATA_INPUT / "edge_arrays_prompts.json"
_VALID_FUNCTIONS = _DATA_INPUT / "functions_definition.json"


def _write_invalid_array_functions(path: Path) -> None:
    path.write_text(
        json.dumps([{
            "name": "fn_bad",
            "description": "array without items",
            "parameters": {"xs": {"type": "array"}},
            "returns": {"type": "number"},
        }]),
        encoding="utf-8",
    )


def _fake_bootstrap() -> tuple[MagicMock, MagicMock, dict[int, str]]:
    return MagicMock(), MagicMock(), {}


def _fail_if_bootstrapped() -> tuple[MagicMock, MagicMock, dict[int, str]]:
    pytest.fail("bootstrap should not run before inputs are validated")
    raise AssertionError("unreachable")


def _install_fake_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    outcomes: list[object],
) -> None:
    class FakePipeline:
        def __init__(self, **kwargs: object) -> None:
            pass

        def process_prompt(self, user_prompt: str) -> object:
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

    monkeypatch.setattr(app, "FunctionCallingPipeline", FakePipeline)


def _write_prompts(path: Path, prompts: list[str]) -> None:
    path.write_text(
        json.dumps([{"prompt": p} for p in prompts]), encoding="utf-8"
    )


def test_invalid_schema_fails_at_startup_without_bootstrap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(app, "initialize_system_dependencies",
                        _fail_if_bootstrapped)

    bad_functions = tmp_path / "bad_functions.json"
    _write_invalid_array_functions(bad_functions)

    output = tmp_path / "results.json"
    config = RunConfig(bad_functions, _ARRAY_PROMPTS, output)

    assert run(config, handle_signals=False) == EXIT_STARTUP_ERROR
    assert not output.exists()


def test_missing_input_file_returns_startup_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(app, "initialize_system_dependencies",
                        _fail_if_bootstrapped)

    output = tmp_path / "results.json"
    config = RunConfig(tmp_path / "nope.json", _ARRAY_PROMPTS, output)

    assert run(config) == EXIT_STARTUP_ERROR
    assert not output.exists()


def test_prompt_failure_produces_none_placeholder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A failed prompt still yields one result object; exit code stays 0."""
    prompts = tmp_path / "prompts.json"
    _write_prompts(prompts, ["first", "second"])

    good = FunctionCallResult(
        prompt="second", name="fn_greet", parameters={"name": "x"}
    )
    outcomes: list[object] = [RuntimeError("Grammar deadlock"), good]

    monkeypatch.setattr(app, "initialize_system_dependencies", _fake_bootstrap)
    _install_fake_pipeline(monkeypatch, outcomes)

    output = tmp_path / "results.json"
    config = RunConfig(_VALID_FUNCTIONS, prompts, output)

    assert run(config) == EXIT_SUCCESS
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "prompts.json", "results.json"
    ]

    results = json.loads(output.read_text(encoding="utf-8"))
    assert len(results) == 2
    assert results[0] == {
        "prompt": "first", "name": "none", "parameters": {}
    }
    assert results[1]["name"] == "fn_greet"


def test_all_prompts_succeed_writes_every_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prompts = tmp_path / "prompts.json"
    _write_prompts(prompts, ["one", "two"])

    outcomes: list[object] = [
        FunctionCallResult(prompt="one", name="fn_greet", parameters={}),
        FunctionCallResult(prompt="two", name="fn_greet", parameters={}),
    ]

    monkeypatch.setattr(app, "initialize_system_dependencies", _fake_bootstrap)
    _install_fake_pipeline(monkeypatch, outcomes)

    output = tmp_path / "results.json"
    config = RunConfig(_VALID_FUNCTIONS, prompts, output)

    assert run(config) == EXIT_SUCCESS
    assert len(json.loads(output.read_text(encoding="utf-8"))) == 2


def test_unexpected_exception_yields_placeholder_and_success_exit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prompts = tmp_path / "prompts.json"
    _write_prompts(prompts, ["boom"])

    outcomes: list[object] = [ValueError("something odd")]

    monkeypatch.setattr(app, "initialize_system_dependencies", _fake_bootstrap)
    _install_fake_pipeline(monkeypatch, outcomes)

    output = tmp_path / "results.json"
    config = RunConfig(_VALID_FUNCTIONS, prompts, output)

    assert run(config) == EXIT_SUCCESS
    assert json.loads(output.read_text(encoding="utf-8")) == [
        {"prompt": "boom", "name": "none", "parameters": {}}
    ]


def test_parse_args_defaults_use_project_paths() -> None:
    config = parse_args([])

    assert config.functions_path.name == "functions_definition.json"
    assert config.prompts_path.name == "function_calling_tests.json"
    assert config.output_path.name == "function_calling_results.json"
    assert config.trace is False


def test_parse_args_rejects_removed_errors_output(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        parse_args(["--errors_output", str(tmp_path / "errors.json")])


def test_directory_input_returns_startup_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(app, "initialize_system_dependencies",
                        _fail_if_bootstrapped)

    output = tmp_path / "results.json"
    config = RunConfig(tmp_path, _ARRAY_PROMPTS, output)

    assert run(config, handle_signals=False) == EXIT_STARTUP_ERROR
    assert not output.exists()


def test_non_utf8_input_returns_startup_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(app, "initialize_system_dependencies",
                        _fail_if_bootstrapped)

    bad = tmp_path / "bad.json"
    bad.write_bytes(b"\xff\xfe\x00\x01")

    output = tmp_path / "results.json"
    config = RunConfig(bad, _ARRAY_PROMPTS, output)

    assert run(config, handle_signals=False) == EXIT_STARTUP_ERROR
    assert not output.exists()


def test_output_parent_is_a_file_returns_startup_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prompts = tmp_path / "prompts.json"
    _write_prompts(prompts, ["one"])
    outcomes: list[object] = [
        FunctionCallResult(prompt="one", name="fn_greet", parameters={})
    ]
    monkeypatch.setattr(app, "initialize_system_dependencies", _fake_bootstrap)
    _install_fake_pipeline(monkeypatch, outcomes)

    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    output = blocker / "results.json"
    config = RunConfig(_VALID_FUNCTIONS, prompts, output)

    assert run(config, handle_signals=False) == EXIT_STARTUP_ERROR


def test_save_partial_skips_empty_output(tmp_path: Path) -> None:
    config = RunConfig(
        _VALID_FUNCTIONS, _ARRAY_PROMPTS, tmp_path / "results.json"
    )

    app._save_partial([], config)

    assert not config.output_path.exists()


def test_save_partial_writes_when_there_is_content(tmp_path: Path) -> None:
    result = FunctionCallResult(prompt="x", name="none", parameters={})
    config = RunConfig(
        _VALID_FUNCTIONS, _ARRAY_PROMPTS, tmp_path / "results.json"
    )

    app._save_partial([result], config)

    assert len(json.loads(config.output_path.read_text(encoding="utf-8"))) == 1


def test_run_restores_previous_sigint_handler(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prompts = tmp_path / "prompts.json"
    _write_prompts(prompts, [])
    monkeypatch.setattr(app, "initialize_system_dependencies", _fake_bootstrap)
    _install_fake_pipeline(monkeypatch, [])

    before = signal.getsignal(signal.SIGINT)
    config = RunConfig(_VALID_FUNCTIONS, prompts, tmp_path / "results.json")

    assert run(config) == EXIT_SUCCESS
    assert signal.getsignal(signal.SIGINT) == before


@pytest.mark.skipif(
    not hasattr(signal, "pthread_sigmask"), reason="POSIX only"
)
def test_defer_sigint_blocks_and_restores_mask() -> None:
    before = signal.pthread_sigmask(signal.SIG_BLOCK, set())

    with app._defer_sigint():
        during = signal.pthread_sigmask(signal.SIG_BLOCK, set())
        assert signal.SIGINT in during

    after = signal.pthread_sigmask(signal.SIG_BLOCK, set())
    assert after == before


def test_run_can_skip_signal_handling(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prompts = tmp_path / "prompts.json"
    _write_prompts(prompts, [])
    monkeypatch.setattr(app, "initialize_system_dependencies", _fake_bootstrap)
    _install_fake_pipeline(monkeypatch, [])

    before = signal.getsignal(signal.SIGINT)
    config = RunConfig(_VALID_FUNCTIONS, prompts, tmp_path / "results.json")

    assert run(config, handle_signals=False) == EXIT_SUCCESS
    assert signal.getsignal(signal.SIGINT) is before
