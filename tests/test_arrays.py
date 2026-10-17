"""
Array-support tests: schema declaration, compiler mapping, and PDA grammar.

Scope: arrays of scalar leaves, declared JSON-Schema style
(``{"type": "array", "items": {"type": "number"}}``), empty allowed,
unbounded, no nesting.
"""
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.automata import (
    ArraySpec,
    CompiledSchema,
    ExactMatchFSM,
    JSONPushdownAutomaton,
    NumberFSM,
    PDAState,
    SchemaCompiler,
    Scope,
    StringLiteralFSM,
)
from src.io_manager import load_function_definitions
from src.schema import FunctionDefinition, ParameterField

_DATA_INPUT = Path(__file__).resolve().parent.parent / "data" / "input"


def _number_array_schema() -> CompiledSchema:
    return {"numbers": ArraySpec(item=lambda: NumberFSM())}


def _string_array_schema() -> CompiledSchema:
    return {"names": ArraySpec(item=lambda: StringLiteralFSM())}


def _accepts(schema: CompiledSchema, text: str) -> bool:
    pda = JSONPushdownAutomaton(schema)
    for char in text:
        if not pda.advance(char):
            return False
    return pda.state == PDAState.TERMINAL


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

def test_array_requires_items() -> None:
    with pytest.raises(ValidationError):
        ParameterField(type="array")


def test_items_forbidden_on_scalar() -> None:
    with pytest.raises(ValidationError):
        ParameterField(
            type="string", items=ParameterField(type="string")
        )


def test_nested_arrays_allowed() -> None:
    field = ParameterField(
        type="array",
        items=ParameterField(
            type="array", items=ParameterField(type="number")
        ),
    )
    assert field.items is not None
    assert field.items.type == "array"
    assert field.items.items is not None
    assert field.items.items.type == "number"


def test_valid_array_field() -> None:
    field = ParameterField(
        type="array", items=ParameterField(type="number")
    )
    assert field.type == "array"
    assert field.items is not None
    assert field.items.type == "number"


def test_edge_array_fixtures_load() -> None:
    funcs = load_function_definitions(
        _DATA_INPUT / "edge_arrays_functions.json"
    )
    assert len(funcs) == 3
    numbers = funcs[0].parameters["numbers"]
    assert numbers.type == "array"
    assert numbers.items is not None
    assert numbers.items.type == "number"


# ---------------------------------------------------------------------------
# Compiler mapping
# ---------------------------------------------------------------------------

def test_compile_extractor_array_spec() -> None:
    fn = FunctionDefinition(
        name="fn_sum_list",
        description=".",
        parameters={
            "numbers": ParameterField(
                type="array", items=ParameterField(type="number")
            ),
            "label": ParameterField(type="string"),
        },
        returns=ParameterField(type="number"),
    )

    table = SchemaCompiler.compile_extractor_table(fn)

    assert isinstance(table["numbers"], ArraySpec)
    assert isinstance(table["numbers"].item(), NumberFSM)
    assert callable(table["label"])


# ---------------------------------------------------------------------------
# PDA accepts
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    '{"numbers":[]}',
    '{"numbers":[0]}',
    '{"numbers":[1,2,3]}',
    '{"numbers":[-1.5,2]}',
    '{"numbers":[ 1 , 2 ]}',
    '{"numbers":[\n 1,\n 2\n]}',
])
def test_number_array_accepts(text: str) -> None:
    assert _accepts(_number_array_schema(), text)


@pytest.mark.parametrize("text", [
    '{"names":[]}',
    '{"names":["a","b"]}',
    '{"names":[""]}',
    '{"names":["hello world"]}',
])
def test_string_array_accepts(text: str) -> None:
    assert _accepts(_string_array_schema(), text)


def test_boolean_array_accepts() -> None:
    schema: CompiledSchema = {
        "flags": ArraySpec(item=lambda: ExactMatchFSM(["true", "false"]))
    }
    assert _accepts(schema, '{"flags":[true,false]}')


def test_array_scope_is_pushed_and_popped() -> None:
    pda = JSONPushdownAutomaton(_number_array_schema())
    for char in '{"numbers":[':
        assert pda.advance(char)

    assert pda.stack[-1].scope == Scope.ARRAY

    for char in '1,2]':
        assert pda.advance(char)

    assert all(frame.scope != Scope.ARRAY for frame in pda.stack)


# ---------------------------------------------------------------------------
# PDA rejects
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    '{"numbers":[1,]}',      # trailing comma
    '{"numbers":[,1]}',      # leading comma
    '{"numbers":["a"]}',     # element type mismatch
    '{"numbers":1}',         # scalar where array expected
    '{"numbers":"[1]"}',     # string where array expected
    '{"numbers":[}',         # unclosed, wrong closer
    '{"numbers":[1}',        # unclosed
    '{"numbers":[1,2}',      # unclosed after elements
])
def test_number_array_rejects(text: str) -> None:
    assert not _accepts(_number_array_schema(), text)


def test_string_array_rejects_wrong_element_type() -> None:
    assert not _accepts(_string_array_schema(), '{"names":[1,2]}')


def test_empty_array_then_more_elements_is_rejected() -> None:
    # `[]` closes the array; the object must then close, not continue.
    assert not _accepts(_number_array_schema(), '{"numbers":[],1]}')


# ---------------------------------------------------------------------------
# JSON round-trip of a generated array
# ---------------------------------------------------------------------------

def test_array_output_round_trips() -> None:
    output = json.loads('{"numbers": [1, 2, 3]}')
    assert output == {"numbers": [1, 2, 3]}
    assert isinstance(output["numbers"], list)
