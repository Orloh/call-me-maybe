"""
Array-support limitation fixtures (characterization tests).

These tests pin how the CURRENT, unmodified engine handles array-typed
function arguments. They are intentionally "failing-because-unsupported"
documentation: if array support is ever added to AllowedTypes, the
CompiledSchema and the PDA, several of these tests are expected to flip.

Fixtures:
    data/input/edge_arrays_functions.json  -- functions whose params are arrays
    data/input/edge_arrays_prompts.json     -- prompts that need those params

No production code is modified by this module.
"""
from pathlib import Path

import pytest

from src.automata import (
    JSONPushdownAutomaton,
    NumberFSM,
    PDAState,
    Scope,
    SchemaCompiler,
    StringLiteralFSM,
    UnsupportedSchemaTypeError,
)
from src.io_manager import (
    SchemaValidationError,
    load_function_definitions,
    load_prompts,
)

_DATA_INPUT = Path(__file__).resolve().parent.parent / "data" / "input"
_ARRAY_FUNCTIONS = _DATA_INPUT / "edge_arrays_functions.json"
_ARRAY_PROMPTS = _DATA_INPUT / "edge_arrays_prompts.json"


def _pda_at_value_position(key: str) -> JSONPushdownAutomaton:
    """Drive a scalar PDA up to EXPECTING_VALUE for `key`."""
    schema = {key: lambda: NumberFSM()}
    pda = JSONPushdownAutomaton(schema)
    for char in '{"' + key + '":':
        assert pda.advance(char), char
    return pda


def test_array_prompts_are_loadable() -> None:
    """The prompts themselves are valid; only the array schema is the blocker."""
    prompts = load_prompts(_ARRAY_PROMPTS)
    assert len(prompts) == 5
    assert any("[" in item.prompt for item in prompts)


def test_array_typed_parameters_fail_schema_validation() -> None:
    """`AllowedTypes` has no `array`, so loading the file dies before generation."""
    with pytest.raises(SchemaValidationError) as exc_info:
        load_function_definitions(_ARRAY_FUNCTIONS)

    message = str(exc_info.value)
    assert "Validation failed" in message
    assert "array" in message


def test_compiler_has_no_array_mapping() -> None:
    """Even if pydantic allowed it, the compiler cannot translate `array`."""
    with pytest.raises(UnsupportedSchemaTypeError):
        SchemaCompiler._map_type_to_fsm("array")


def test_pda_rejects_array_open_bracket_at_value() -> None:
    """A scalar value position rejects `[`, so an array can never be opened."""
    pda = _pda_at_value_position("numbers")

    assert not pda.accepts_char("[")
    assert not pda.clone().advance("[")
    assert not pda._on_value("[")


def test_array_scope_is_unreachable_for_scalar_schemas() -> None:
    """No scalar-schema program ever pushes Scope.ARRAY or hits its state."""
    schema = {
        "a": lambda: NumberFSM(mode="integer"),
        "b": lambda: StringLiteralFSM(),
    }
    programs = [
        '{"a":1,"b":"x"}',
        '{ "b":"y" , "a":2 }',
        '{"a":0,"b":""}',
    ]

    for program in programs:
        pda = JSONPushdownAutomaton(schema)
        seen_states: list[PDAState] = []
        for char in program:
            seen_states.append(pda.state)
            assert pda.advance(char), (program, char)

        assert Scope.ARRAY not in pda.stack
        assert PDAState.EXPECTING_ARRAY_START not in seen_states
        assert pda.state == PDAState.TERMINAL
