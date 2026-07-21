import logging
from pathlib import Path

from src.io_manager import load_function_definitions, load_prompts, write_output

from src.engine import initialize_system_dependencies
from src.engine import FunctionCallingPipeline

logging_format = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=logging_format)
logger = logging.getLogger(__name__)

def main() -> None:
    # Path Setup
    project_root = Path(__file__).resolve().parent.parent
    input_dir = project_root / "data" / "input"
    output_dir = project_root / "data" / "output"

    functions_path = input_dir / "function_definitions" / "function_definitions.json"
    prompts_path = input_dir / "function_call_prompts" / "function_calling_tests.json"
    output_path = output_dir / "function_calling_results.json"

    logger.info(f"{project_root} :{project_root.exists()}")

    # Bootstrapping
    logger.info("Bootstrapping dependencies (Model & Prefix Trie)...")
    model, trie = initialize_system_dependencies()

    # Load Data Using io_manager
    logger.info("Loading function definitions and prompts...")
    available_functions = load_function_definitions(functions_path)
    prompt_items = load_prompts(prompts_path)

    # Execute pipline
    logger.info(f"Initializing pipeline with {len(available_functions)}")
    pipeline = FunctionCallingPipeline(
        model=model,
        trie=trie,
        available_functions=available_functions
    )

    results = []
    logger.info(f"Strating generating loop for {len(prompt_items)} prompts...")

    for i, item in enumerate(prompt_items, 1):
        logger.info(f"Processing [{i}/{len(prompt_items)}]: '{item.prompt[:40]}...'")
        
        try:
            result = pipeline.process_prompt(
                user_prompt=item.prompt,
                available_functions=available_functions
            )
            results.append(result)
            logger.info(f"-> Success! Routed to: {result.name}()")

        except Exception as e:
            logger.error(f"-> Failed on prompt [{i}]: {e}")

    #Save Output using io_manager
    logger.info(f"Writing {len(results)} successful results to {output_path}...")
    write_output(results, output_path)
    logger.info("Pipeline execution complete! 🎉")

if __name__ == "__main__":
    main()
