import json
import pytest
from pathlib import Path
from src.io_manager import JSONParsingError, SchemaValidationError, load_function_definitions, FunctionFileNotFoundError
from src.schema import FunctionDefinition


VALID_MOCK_FUNCTIONS = [
    {
        "name": "fn_test",
        "description": "A test function",
        "parameters": {},
        "returns": {"type": "string"}
    },
    {
        "name": "fn_add_numbers",
        "description": "Add two numbers together.",
        "parameters": {
            "a": {"type": "number"},
            "b": {"type": "number"}
        },
        "returns": {"type": "number"}
    }
]

INVALID_MOCK_FUNCTIONS = [
    {
        "name": "fn_test",
        "description": "A test function",
        "parameters": {},
        "returns": {"type": "string"}
    },
    {
        "description": "Add two numbers together.",
        "parameters": {
            "a": {"type": "number"},
            "b": {"type": "number"}
        },
        "returns": {"type": "number"}
    }
]

INVALID_EXTRA_FIELD_MOCK_FUNCTION = [
    {
        "name": "fn_greet",
        "description": "Greet a person.",
        "parameters": {
            "name": {
                "type": "string",
                "forbidden_extra_key": "not allowed"  # Extra key not defined in ParameterField!
            }
        },
        "returns": {"type": "string"}
    }
]


def test_load_function_definitions_success(tmp_path: Path) -> None:
    """ Test succesful loading of the functions definitions JSON file"""
    test_file = tmp_path / "valid_function_definitions.json"
    
    with open(test_file, "w", encoding="utf-8") as file:
        json.dump(VALID_MOCK_FUNCTIONS, file)

    result = load_function_definitions(test_file)

    assert result is not None
    assert len(result) == 2
    assert isinstance(result, list)
    assert isinstance(result[0], FunctionDefinition)
    assert isinstance(result[1], FunctionDefinition)
    assert result[0].name == "fn_test"
    assert result[1].name == "fn_add_numbers"


def test_load_function_definitions_file_not_found() -> None:
    """Test that a non-existing filepath raises a ConfigFileNotFoundError."""
    fake_path = Path("fake_directory/fake_file.json")
    with pytest.raises(FunctionFileNotFoundError) as exc_info:
        result = load_function_definitions(fake_path)

    print(exc_info.value)


def test_load_function_definitions_bad_json_syntax(tmp_path: Path) -> None:
    """Test that a file with broken JSON syntax raises a JSONParsingError."""
    test_file = tmp_path / "broken_syntax.json"

    with open(test_file, "w", encoding="utf-8") as file:
        file.write('{"invalid json"; "value"}')

    with pytest.raises(JSONParsingError):
        load_function_definitions(test_file)


def test_load_function_definitions_schema_validation_error(tmp_path: Path) -> None:
    """Test that valid JSON with missing required fields is rejected and raises a ShemaValidationError."""
    test_file = tmp_path / "invalid_function_definitions.json"

    with open(test_file, "w", encoding="utf-8") as file:
        json.dump(INVALID_MOCK_FUNCTIONS, file)

    with pytest.raises(SchemaValidationError) as exc_info:
        load_function_definitions(test_file)

    assert "Validation failed" in str(exc_info.value)

def test_load_function_definitions_extra_field(tmp_path: Path) -> None:
    test_file = tmp_path / "extra_field.json"

    with open(test_file, "w", encoding="utf-8") as file:
        json.dump(INVALID_EXTRA_FIELD_MOCK_FUNCTION, file)

    with pytest.raises(SchemaValidationError) as exc_info:
        load_function_definitions(test_file)

    assert "Extra inputs are not permitted" in str(exc_info.value)
