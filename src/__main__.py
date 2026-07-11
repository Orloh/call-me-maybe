import logging
from pathlib import Path

from src.io_manager import load_function_definitions, load_prompts, write_output

from src.engine import initialize_system_dependencies
from src.engine import FunctionCallingPipeline

logging_format = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=logging_format)
logger = logging.getLogger(__name__)

def main():
    # Path Setup
    # Bootstrapping
    # Load Data Using io_manager
    # Execute pipline
    # Save Output using io_manager
    pass

if __name__ == "__main__":
    main()
