import json
from pathlib import Path
from pydantic import ValidationError
from src.schema import FunctionDefinition


class SchemaValidationError(Exception):
    pass

class FunctionFileNotFoundError(Exception):
    pass

class JSONParsingError(Exception):
    pass

def load_function_definitions(filepath: Path) -> list[FunctionDefinition]:
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            data = json.load(file)

    except FileNotFoundError as error:
        raise FunctionFileNotFoundError(f"File not found: {filepath}") from error

    except json.JSONDecodeError as error:
        raise JSONParsingError(f"Invalid JSON syntax in {filepath}: {error}") from error
    
    except Exception as e:
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
