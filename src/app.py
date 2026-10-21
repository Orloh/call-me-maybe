import argparse
import logging
import signal
import sys
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

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
    write_output,
)
from src.schema import (
    FunctionCallResult,
    FunctionDefinition,
    PromptItem,
)

logger = logging.getLogger(__name__)

EXIT_SUCCESS = 0
EXIT_STARTUP_ERROR = 2
EXIT_INTERRUPTED = 130

# Sentinel used when a prompt cannot be routed or processed. It is not a
# function from functions_definition.json; it keeps the output to exactly one
# object per prompt while signalling "no call could be produced".
NO_FUNCTION = "none"

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
    trace: bool = False


_HAS_PTHREAD_SIGMASK = hasattr(signal, "pthread_sigmask")


@contextmanager
def _defer_sigint() -> Iterator[None]:
    """
    Defers SIGINT delivery for the duration of the block (POSIX only).

    The SIGINT handler runs on the main thread and itself saves partial
    output, so it can otherwise interrupt an in-flight serialization.
    Masking the signal (rather than a threading.Lock, which the handler
    would deadlock on) lets the current write finish; the deferred signal
    is then delivered and the handler re-saves the now-complete data.
    """
    if not _HAS_PTHREAD_SIGMASK:
        yield
        return

    previous = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT})
    try:
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)


def _project_defaults() -> tuple[Path, Path, Path]:
    """Returns the default functions, prompts, and results paths."""
    root = Path(__file__).resolve().parent.parent
    input_dir = root / "data" / "input"
    output_dir = root / "data" / "output"
    return (
        input_dir / "functions_definition.json",
        input_dir / "function_calling_tests.json",
        output_dir / "function_calling_results.json",
    )


def parse_args(argv: Sequence[str] | None = None) -> RunConfig:
    """
    Parses CLI arguments into a resolved RunConfig.

    Explicit paths override the project defaults. The results file is the
    only file written by a run.
    """
    default_functions, default_prompts, default_output = _project_defaults()

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

    return RunConfig(
        functions_path=functions_path,
        prompts_path=prompts_path,
        output_path=output_path,
        trace=args.trace,
    )


def _load_inputs(
    config: RunConfig
) -> tuple[list[FunctionDefinition], list[PromptItem]]:
    """Loads and validates the function definitions and prompt files."""
    logger.info("Loading function definitions and prompts...")
    available_functions = load_function_definitions(config.functions_path)
    prompt_items = load_prompts(config.prompts_path)
    return available_functions, prompt_items


def _placeholder(prompt: str) -> FunctionCallResult:
    """
    Builds the ``none`` result recorded when a prompt cannot be processed.

    The errors file was removed so that a run produces exactly one JSON file;
    a failure is still represented by one object in that file.
    """
    return FunctionCallResult(prompt=prompt, name=NO_FUNCTION, parameters={})


def _save(results: list[FunctionCallResult], config: RunConfig) -> bool:
    """
    Writes the results file atomically.

    Returns True on success; a write failure is logged and reported as False.
    """
    with _defer_sigint():
        try:
            write_output(results, config.output_path)
        except _WRITE_ERRORS as error:
            logger.error("Failed to write results to %s: %s",
                         config.output_path, error)
            return False
        except Exception:
            logger.exception("Unexpected error writing results to %s",
                             config.output_path)
            return False
        return True


def _save_partial(
    results: list[FunctionCallResult],
    config: RunConfig,
) -> None:
    """
    Persists partial output on interrupt.

    Never clobbers an existing output file with an empty result set.
    """
    if results:
        _save(results, config)


def _run_body(
    config: RunConfig,
    results: list[FunctionCallResult],
) -> int:
    """
    Core run: load inputs, bootstrap, then process every prompt.

    Inputs are loaded before the model is bootstrapped so that invalid
    input fails fast (exit 2) without paying for model startup. A failure
    on one prompt is logged and recorded as a ``none`` placeholder result,
    so the output always contains exactly one object per prompt.
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

    failed = 0
    for index, item in enumerate(prompt_items, 1):
        logger.info("Processing [%d/%d]: %r",
                    index, len(prompt_items), item.prompt)
        try:
            result = pipeline.process_prompt(user_prompt=item.prompt)
            results.append(result)
            logger.info("-> Success! Routed to: %s()", result.name)

        except _PROMPT_ERRORS as error:
            logger.warning("-> Failed on prompt [%d]: %s", index, error)
            results.append(_placeholder(item.prompt))
            failed += 1

        except Exception:
            logger.exception("-> Unexpected failure on prompt [%d]", index)
            results.append(_placeholder(item.prompt))
            failed += 1

    if not _save(results, config):
        return EXIT_STARTUP_ERROR

    logger.info(
        "Pipeline complete: %d succeeded, %d failed.",
        len(results) - failed, failed
    )
    return EXIT_SUCCESS


def run(config: RunConfig, *, handle_signals: bool = True) -> int:
    """
    Executes the pipeline for every prompt in `config`.

    When `handle_signals` is set and we are on the main thread, a SIGINT
    handler is installed for the duration of the run (so Ctrl-C saves
    partial output and exits 130) and the previous handler is restored
    afterwards. Off the main thread, signal registration is skipped.
    """
    results: list[FunctionCallResult] = []

    def _handle_interrupt(sig: int, frame: object) -> None:
        """Saves partial results on SIGINT and exits with status 130."""
        logger.info("Interrupted by user. Saving partial results...")
        try:
            _save_partial(results, config)
        except Exception:  # pragma: no cover - best-effort save
            logger.exception("Failed to save partial results")
        sys.exit(EXIT_INTERRUPTED)

    main_thread = threading.current_thread() is threading.main_thread()
    if not (handle_signals and main_thread):
        return _run_body(config, results)

    previous_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, _handle_interrupt)
    try:
        return _run_body(config, results)
    finally:
        signal.signal(signal.SIGINT, previous_handler)


def main() -> None:
    """Configures logging, parses arguments, runs the pipeline, and exits."""
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
