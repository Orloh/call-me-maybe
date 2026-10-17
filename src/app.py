import argparse
import logging
import signal
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from src.automata import UnsupportedSchemaTypeError
from src.engine import (
    FunctionCallingPipeline,
    MaxTokensExceededError,
    initialize_system_dependencies,
)
from src.io_manager import (
    InputFileNotFoundError,
    JSONParsingError,
    PermissionDeniedError,
    SchemaValidationError,
    WriteOutputError,
    load_function_definitions,
    load_prompts,
    write_errors,
    write_output,
)
from src.schema import (
    FunctionCallResult,
    FunctionDefinition,
    PromptFailure,
    PromptItem,
)

logger = logging.getLogger(__name__)

EXIT_SUCCESS = 0
EXIT_PROMPT_FAILURES = 1
EXIT_STARTUP_ERROR = 2
EXIT_INTERRUPTED = 130

_INPUT_ERRORS = (
    InputFileNotFoundError,
    PermissionDeniedError,
    JSONParsingError,
    SchemaValidationError,
)

_INPUT_READ_ERRORS = (OSError, UnicodeError, RecursionError)

_WRITE_ERRORS = (WriteOutputError, PermissionDeniedError)

_PROMPT_ERRORS = (
    MaxTokensExceededError,
    RuntimeError,
    UnsupportedSchemaTypeError,
)


@dataclass
class RunConfig:
    """Resolved paths and options for a single pipeline run."""
    functions_path: Path
    prompts_path: Path
    output_path: Path
    errors_path: Path
    trace: bool = False


def _derive_errors_path(output_path: Path) -> Path:
    return output_path.with_name(
        f"{output_path.stem}_errors{output_path.suffix}"
    )


def _project_defaults() -> tuple[Path, Path, Path, Path]:
    root = Path(__file__).resolve().parent.parent
    input_dir = root / "data" / "input"
    output_dir = root / "data" / "output"
    return (
        input_dir / "functions_definition.json",
        input_dir / "function_calling_tests.json",
        output_dir / "function_calling_results.json",
        output_dir / "function_calling_errors.json",
    )


def parse_args(argv: Sequence[str] | None = None) -> RunConfig:
    """
    Parses CLI arguments into a resolved RunConfig.

    Explicit paths override the project defaults. When `--output` is given
    without `--errors_output`, the errors path is derived from it
    (e.g. `results.json` -> `results_errors.json`).
    """
    default_functions, default_prompts, default_output, default_errors = (
        _project_defaults()
    )

    parser = argparse.ArgumentParser(
        description="Run the Function Calling Pipeline."
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Enable step-by-step PDA/FSM tracing in the terminal."
    )
    parser.add_argument(
        "--functions_definition",
        type=str,
        default=None,
        help="Path to function definitions JSON "
             "(default: data/input/functions_definition.json)."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to prompts JSON "
             "(default: data/input/function_calling_tests.json)."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write results JSON "
             "(default: data/output/function_calling_results.json)."
    )
    parser.add_argument(
        "--errors_output",
        type=str,
        default=None,
        help="Path to write per-prompt failures JSON "
             "(default: derived from --output)."
    )
    args = parser.parse_args(argv)

    functions_path = (
        Path(args.functions_definition).resolve()
        if args.functions_definition
        else default_functions
    )
    prompts_path = (
        Path(args.input).resolve()
        if args.input
        else default_prompts
    )
    output_path = (
        Path(args.output).resolve()
        if args.output
        else default_output
    )

    if args.errors_output:
        errors_path = Path(args.errors_output).resolve()
    elif args.output:
        errors_path = _derive_errors_path(output_path)
    else:
        errors_path = default_errors

    return RunConfig(
        functions_path=functions_path,
        prompts_path=prompts_path,
        output_path=output_path,
        errors_path=errors_path,
        trace=args.trace,
    )


def _load_inputs(
    config: RunConfig
) -> tuple[list[FunctionDefinition], list[PromptItem]]:
    logger.info("Loading function definitions and prompts...")
    available_functions = load_function_definitions(config.functions_path)
    prompt_items = load_prompts(config.prompts_path)
    return available_functions, prompt_items


def _make_failure(prompt: str, error: Exception) -> PromptFailure:
    return PromptFailure(
        prompt=prompt,
        error_type=type(error).__name__,
        error=str(error),
    )


def _save(
    results: list[FunctionCallResult],
    failures: list[PromptFailure],
    config: RunConfig,
) -> bool:
    """
    Writes results and (only when present) failures. Returns True on
    full success; a write failure is logged and reported as False.
    """
    ok = True

    try:
        write_output(results, config.output_path)
    except _WRITE_ERRORS as error:
        logger.error("Failed to write results to %s: %s",
                     config.output_path, error)
        ok = False
    except Exception:
        logger.exception("Unexpected error writing results to %s",
                         config.output_path)
        ok = False

    if failures:
        try:
            write_errors(failures, config.errors_path)
        except _WRITE_ERRORS as error:
            logger.error("Failed to write errors to %s: %s",
                         config.errors_path, error)
            ok = False
        except Exception:
            logger.exception("Unexpected error writing errors to %s",
                             config.errors_path)
            ok = False

    return ok


def _save_partial(
    results: list[FunctionCallResult],
    failures: list[PromptFailure],
    config: RunConfig,
) -> None:
    """
    Persists partial output on interrupt, but never clobbers existing
    output files with an empty result set.
    """
    if results or failures:
        _save(results, failures, config)


def _run_body(
    config: RunConfig,
    results: list[FunctionCallResult],
    failures: list[PromptFailure],
) -> int:
    """
    Core run: load inputs, bootstrap, then process every prompt.

    Inputs are loaded before the model is bootstrapped so that invalid
    input fails fast (exit 2) without paying for model startup. A failure
    on one prompt is recorded and does not abort the run.
    """
    try:
        available_functions, prompt_items = _load_inputs(config)
    except _INPUT_ERRORS as error:
        logger.error("Input error: %s", error)
        return EXIT_STARTUP_ERROR
    except _INPUT_READ_ERRORS as error:
        logger.error("Input read error (%s): %s",
                     type(error).__name__, error)
        return EXIT_STARTUP_ERROR
    except Exception:
        logger.exception("Unexpected error while loading inputs")
        return EXIT_STARTUP_ERROR

    logger.info("Bootstrapping dependencies (Model & Prefix Trie)...")
    try:
        model, trie, token_to_decoded = initialize_system_dependencies()
    except Exception as error:
        logger.error("Bootstrap error: %s", error)
        return EXIT_STARTUP_ERROR

    logger.info("Initializing pipeline with %d function(s)...",
                len(available_functions))
    try:
        pipeline = FunctionCallingPipeline(
            model=model,
            trie=trie,
            token_to_decoded=token_to_decoded,
            available_functions=available_functions,
            trace=config.trace,
        )
    except Exception as error:
        logger.error("Pipeline initialization error: %s", error)
        return EXIT_STARTUP_ERROR

    logger.info("Starting generation loop for %d prompt(s)...",
                len(prompt_items))

    for index, item in enumerate(prompt_items, 1):
        logger.info("Processing [%d/%d]: %r",
                    index, len(prompt_items), item.prompt)
        try:
            result = pipeline.process_prompt(
                user_prompt=item.prompt,
                available_functions=available_functions,
            )
            results.append(result)
            logger.info("-> Success! Routed to: %s()", result.name)

        except _PROMPT_ERRORS as error:
            logger.warning("-> Failed on prompt [%d]: %s", index, error)
            failures.append(_make_failure(item.prompt, error))

        except Exception as error:
            logger.exception("-> Unexpected failure on prompt [%d]", index)
            failures.append(_make_failure(item.prompt, error))

    if not _save(results, failures, config):
        return EXIT_STARTUP_ERROR

    logger.info(
        "Pipeline complete: %d succeeded, %d failed.",
        len(results), len(failures)
    )
    return EXIT_PROMPT_FAILURES if failures else EXIT_SUCCESS


def run(config: RunConfig, *, handle_signals: bool = True) -> int:
    """
    Executes the pipeline for every prompt in `config`.

    When `handle_signals` is set and we are on the main thread, a SIGINT
    handler is installed for the duration of the run (so Ctrl-C saves
    partial output and exits 130) and the previous handler is restored
    afterwards. Off the main thread, signal registration is skipped.
    """
    results: list[FunctionCallResult] = []
    failures: list[PromptFailure] = []

    def _handle_interrupt(sig: int, frame: object) -> None:
        logger.info("Interrupted by user. Saving partial results...")
        try:
            _save_partial(results, failures, config)
        except Exception:  # pragma: no cover - best-effort save
            logger.exception("Failed to save partial results")
        sys.exit(EXIT_INTERRUPTED)

    main_thread = threading.current_thread() is threading.main_thread()
    if not (handle_signals and main_thread):
        return _run_body(config, results, failures)

    previous_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, _handle_interrupt)
    try:
        return _run_body(config, results, failures)
    finally:
        signal.signal(signal.SIGINT, previous_handler)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    try:
        config = parse_args()
    except SystemExit:
        raise
    except Exception as error:
        logger.error("Failed to parse arguments: %s", error)
        sys.exit(EXIT_STARTUP_ERROR)

    try:
        code = run(config)
    except SystemExit:
        raise
    except BrokenPipeError:
        code = EXIT_SUCCESS
    except Exception:
        logger.exception("Unexpected top-level error")
        code = EXIT_STARTUP_ERROR

    sys.exit(code)
