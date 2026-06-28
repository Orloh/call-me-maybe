import json
import logging
from pathlib import Path
from pydantic import ValidationError
from src.schema import FunctionCallResult, FunctionDefinition, PromptItem


class SchemaValidationError(Exception):
    pass

class InputFileNotFoundError(Exception):
    pass

class JSONParsingError(Exception):
    pass

class PermissionDeniedError(Exception):
    """Raised when the application lacks read access to the target file."""
    pass

logger = logging.getLogger(__name__)


def load_function_definitions(filepath: Path) -> list[FunctionDefinition]:
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            data = json.load(file)

    except FileNotFoundError as error:
        raise InputFileNotFoundError(f"File not found: {filepath}") from error

    except PermissionError as error:
        raise PermissionDeniedError(f"Permission denied accessing: {filepath}") from error

    except json.JSONDecodeError as error:
        raise JSONParsingError(f"Invalid JSON syntax in {filepath}: {error}") from error
    
    except Exception as error:
        logger.exception(f"An unexpected error ocurred while loading {filepath}")
        raise Exception
    
    if not isinstance(data, list):
        raise SchemaValidationError(f"Root of {filepath} must be a JSON list")

    validated_functions: list[FunctionDefinition] = []
    for idx, item in enumerate(data):
        try:
            validation = FunctionDefinition.model_validate(item)
            validated_functions.append(validation)
        
        except ValidationError as error:
            raise SchemaValidationError(
                    f"Validation failed for function at index {idx} in {filepath}.\n"
                    f"Details: {error}"
                    ) from error
    
    return validated_functions


def load_prompts(filepath: Path) -> list[PromptItem]:
    # 1. Open file using context manager [cite: 96]
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            data = json.load(file)

    except FileNotFoundError as error:
        raise InputFileNotFoundError(f"File not found: {filepath}") from error

    except PermissionError as error:
        raise PermissionDeniedError(f"Permission denied accessing: {filepath}") from error

    except json.JSONDecodeError as error:
        raise JSONParsingError(f"Invalid JSON syntax in {filepath}: {error}") from error
    
    except Exception as error:
        logger.exception(f"An unexpected error ocurred while loading {filepath}")
        raise Exception
   
    if not isinstance(data, list):
        raise SchemaValidationError(f"Root of {filepath} must be a JSON list")
    
    validated_prompts: list[PromptItem] = []
    for idx, item in enumerate(data):
        try:
            validation = PromptItem.model_validate(item)
            validated_prompts.append(validation)
        
        except ValidationError as error:
            raise SchemaValidationError(
                    f"Validation failed for prompt at index {idx} in {filepath}.\n"
                    f"Details: {error}"
                    ) from error
    
    return validated_prompts


def write_output(results: list[FunctionCallResult], filepath: Path) -> None:
    """
    Serializes a list of LLM function call results to a strictly fromatted JSON file
    """
    serialized_data = [result.model_dump() for result in results]

    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)

    except PermissionError as error:
        raise PermissionDeniedError(f"Permission denied creating directories for {filepath}") from error

    try:
        with open(filepath, "w", encoding="utf-8") as file:
            json.dump(serialized_data, file, indent=4)

    except PermissionError as error:
        raise PermissionDeniedError(f"Permission denied writing to: {filepath}") from error
    
    except Exception as error:
        logger.exception(f"An unexpected error ocurred while writing to {filepath}")
        raise Exception(f"Failed to write output: {error}") from error
