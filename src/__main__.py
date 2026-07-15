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
