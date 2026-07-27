import argparse
import logging
from pathlib import Path

from src.io_manager import (
    load_function_definitions,
    load_prompts,
    write_output
)

from src.engine import initialize_system_dependencies
from src.engine import FunctionCallingPipeline

logging_format = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=logging_format)
logger = logging.getLogger(__name__)


def main() -> None:
    # Parse Arguments
    parser = argparse.ArgumentParser(
        description="Run the Function Calling Pipeline."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable step-by-step PDA/FSM tracing in the terminal."
    )
    parser.add_argument(
        "--functions",
        type=str,
        default=None,
        help="Path to function definitions JSON "
             "(default: .../function_definitions.json)."
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default=None,
        help="Path to prompts JSON "
             "(default: .../function_calling_tests.json)."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write output JSON "
             "(default: .../function_calling_results.json)."
    )
    args = parser.parse_args()

    # Path Setup — CLI args override project defaults
    project_root = Path(__file__).resolve().parent.parent
    default_input = project_root / "data" / "input"
    default_output = project_root / "data" / "output"

    functions_path = (
        Path(args.functions).resolve()
        if args.functions
        else (
            default_input
            / "function_definitions"
            / "function_definitions.json"
        )
    )
    prompts_path = (
        Path(args.prompts).resolve()
        if args.prompts
        else (
            default_input
            / "function_call_prompts"
            / "function_calling_tests.json"
        )
    )
    output_path = (
        Path(args.output).resolve()
        if args.output
        else default_output / "function_calling_results.json"
    )

    logger.info(f"{project_root} :{project_root.exists()}")

    # Bootstrapping
    logger.info("Bootstrapping dependencies (Model & Prefix Trie)...")
    model, trie, token_to_decoded = initialize_system_dependencies()

    # Load Data Using io_manager
    logger.info("Loading function definitions and prompts...")
    available_functions = load_function_definitions(functions_path)
    prompt_items = load_prompts(prompts_path)

    # Execute pipline
    logger.info(f"Initializing pipeline with {len(available_functions)}")
    pipeline = FunctionCallingPipeline(
        model=model,
        trie=trie,
        token_to_decoded=token_to_decoded,
        available_functions=available_functions,
        stop_tokens={151643, 151645},
        debug=args.debug
    )

    results = []
    logger.info(f"Strating generating loop for {len(prompt_items)} prompts...")

    for i, item in enumerate(prompt_items, 1):
        logger.info(
            f"Processing [{i}/{len(prompt_items)}]: '{item.prompt[:40]}...'"
        )

        try:
            result = pipeline.process_prompt(
                user_prompt=item.prompt,
                available_functions=available_functions
            )
            results.append(result)
            logger.info(f"-> Success! Routed to: {result.name}()")

        except Exception as e:
            logger.error(f"-> Failed on prompt [{i}]: {e}")

    # Save Output using io_manager
    logger.info(
        f"Writing {len(results)} successful results to {output_path}..."
    )
    write_output(results, output_path)
    logger.info("Pipeline execution complete! 🎉")


if __name__ == "__main__":
    main()
