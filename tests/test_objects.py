"""
Recursive object tests: schema declaration, compiler mapping, and the
frame-stack PDA grammar (objects in objects, arrays of objects, arrays of
arrays, empty objects). All properties are required.
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
    ObjectSpec,
    PDAState,
    SchemaCompiler,
    StringLiteralFSM,
)
from src.io_manager import load_function_definitions
from src.schema import FunctionDefinition, ParameterField

_DATA_INPUT = Path(__file__).resolve().parent.parent / "data" / "input"


def _field(type_: str, **kwargs: object) -> ParameterField:
    return ParameterField(type=type_, **kwargs)  # type: ignore[arg-type]


def _config_tool() -> FunctionDefinition:
    return FunctionDefinition(
        name="fn_config",
        description="Configure a job.",
        parameters={
            "name": _field("string"),
            "config": _field("object", properties={
                "retries": _field("integer"),
                "labels": _field("array", items=_field("string")),
                "nested": _field("object", properties={
                    "flag": _field("boolean"),
                }),
            }),
            "items": _field("array", items=_field("object", properties={
                "id": _field("number"),
            })),
            "matrix": _field("array", items=_field(
                "array", items=_field("number")
            )),
        },
        returns=_field("number"),
    )


def _accepts(schema: CompiledSchema, text: str) -> bool:
    pda = JSONPushdownAutomaton(schema)
    for char in text:
        if not pda.advance(char):
            return False
    return pda.state == PDAState.TERMINAL


_VALID_JSON = (
    '{"name":"x","config":{"retries":3,"labels":["a","b"],'
    '"nested":{"flag":true}},"items":[{"id":1},{"id":2}],'
    '"matrix":[[1,2],[3]]}'
)


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

def test_object_requires_properties() -> None:
    with pytest.raises(ValidationError):
        _field("object")


def test_object_allows_empty_properties() -> None:
    field = _field("object", properties={})
    assert field.properties == {}


def test_object_forbids_items() -> None:
    with pytest.raises(ValidationError):
        _field("object", properties={}, items=_field("number"))


def test_required_must_list_every_property() -> None:
    with pytest.raises(ValidationError):
        _field(
            "object",
            properties={"a": _field("number"), "b": _field("number")},
            required=["a"],
        )


def test_required_may_list_all_properties() -> None:
    field = _field(
        "object",
        properties={"a": _field("number"), "b": _field("number")},
        required=["a", "b"],
    )
    assert field.required == ["a", "b"]


def test_scalar_forbids_properties() -> None:
    with pytest.raises(ValidationError):
        _field("string", properties={})


# ---------------------------------------------------------------------------
# Compiler mapping
# ---------------------------------------------------------------------------

def test_compile_object_specs_recursively() -> None:
    table = SchemaCompiler.compile_extractor_table(_config_tool())

    config = table["config"]
    assert isinstance(config, ObjectSpec)
    assert set(config.fields) == {"retries", "labels", "nested"}
    assert isinstance(config.fields["labels"], ArraySpec)
    assert isinstance(config.fields["nested"], ObjectSpec)

    items = table["items"]
    assert isinstance(items, ArraySpec)
    assert isinstance(items.item, ObjectSpec)

    matrix = table["matrix"]
    assert isinstance(matrix, ArraySpec)
    assert isinstance(matrix.item, ArraySpec)


# ---------------------------------------------------------------------------
# PDA accepts
# ---------------------------------------------------------------------------

def test_full_nested_document_accepts() -> None:
    assert _accepts(
        SchemaCompiler.compile_extractor_table(_config_tool()),
        _VALID_JSON,
    )


def test_nested_object_accepts() -> None:
    schema: CompiledSchema = {
        "o": ObjectSpec(fields={
            "p": ObjectSpec(fields={"q": NumberFSM}),
        })
    }
    assert _accepts(schema, '{"o":{"p":{"q":1}}}')


def test_empty_object_accepts() -> None:
    schema: CompiledSchema = {"o": ObjectSpec(fields={})}
    assert _accepts(schema, '{"o":{}}')


def test_object_inside_array_accepts() -> None:
    schema: CompiledSchema = {
        "items": ArraySpec(item=ObjectSpec(fields={"d": NumberFSM}))
    }
    assert _accepts(schema, '{"items":[{"d":1},{"d":2}]}')
    assert _accepts(schema, '{"items":[]}')


def test_array_of_arrays_accepts() -> None:
    schema: CompiledSchema = {
        "m": ArraySpec(item=ArraySpec(item=NumberFSM))
    }
    assert _accepts(schema, '{"m":[[1,2],[3]]}')
    assert _accepts(schema, '{"m":[[]]}')


def test_boolean_object_field() -> None:
    schema: CompiledSchema = {
        "cfg": ObjectSpec(fields={
            "on": lambda: ExactMatchFSM(["true", "false"]),
        })
    }
    assert _accepts(schema, '{"cfg":{"on":true}}')


# ---------------------------------------------------------------------------
# PDA rejects
# ---------------------------------------------------------------------------

def test_nested_missing_required_rejected() -> None:
    table = SchemaCompiler.compile_extractor_table(_config_tool())
    bad = (
        '{"name":"x","config":{"retries":3,"labels":[]},'
        '"items":[],"matrix":[]}'
    )
    assert not _accepts(table, bad)


def test_nested_unknown_key_rejected() -> None:
    table = SchemaCompiler.compile_extractor_table(_config_tool())
    bad = (
        '{"name":"x","config":{"retries":3,"labels":[],'
        '"nested":{"flag":true,"oops":1}},"items":[],"matrix":[]}'
    )
    assert not _accepts(table, bad)


def test_nested_duplicate_key_rejected() -> None:
    schema: CompiledSchema = {"o": ObjectSpec(fields={"a": NumberFSM})}
    assert not _accepts(schema, '{"o":{"a":1,"a":2}}')


def test_nested_trailing_comma_rejected() -> None:
    schema: CompiledSchema = {"o": ObjectSpec(fields={"a": NumberFSM})}
    assert not _accepts(schema, '{"o":{"a":1,}}')


def test_array_of_objects_missing_key_rejected() -> None:
    table = SchemaCompiler.compile_extractor_table(_config_tool())
    bad = (
        '{"name":"x","config":{"retries":3,"labels":[],'
        '"nested":{"flag":true}},"items":[{"id":1},{}],"matrix":[]}'
    )
    assert not _accepts(table, bad)


def test_object_key_with_quote_accepts() -> None:
    """Property names needing JSON escaping must be representable."""
    schema: CompiledSchema = {"o": ObjectSpec(fields={'a"b': NumberFSM})}
    doc = '{"o":' + json.dumps({'a"b': 1}) + '}'
    assert _accepts(schema, doc)


def test_object_key_with_backslash_accepts() -> None:
    schema: CompiledSchema = {"o": ObjectSpec(fields={"a\\b": NumberFSM})}
    doc = '{"o":' + json.dumps({"a\\b": 1}) + '}'
    assert _accepts(schema, doc)


def test_object_key_with_unicode_accepts() -> None:
    schema: CompiledSchema = {"o": ObjectSpec(fields={"é": NumberFSM})}
    doc = '{"o":' + json.dumps({"é": 1}, ensure_ascii=False) + '}'
    assert _accepts(schema, doc)


def test_object_wrong_field_type_rejected() -> None:
    table = SchemaCompiler.compile_extractor_table(_config_tool())
    bad = (
        '{"name":"x","config":{"retries":"three","labels":[],'
        '"nested":{"flag":true}},"items":[],"matrix":[]}'
    )
    assert not _accepts(table, bad)


# ---------------------------------------------------------------------------
# Fixture + coercion
# ---------------------------------------------------------------------------

def test_edge_object_fixtures_load() -> None:
    funcs = load_function_definitions(
        _DATA_INPUT / "edge_objects_functions.json"
    )
    assert len(funcs) == 2
    config = funcs[0].parameters["config"]
    assert config.type == "object"
    assert config.properties is not None


def test_coercion_recurses_through_objects() -> None:
    from src.engine.pipeline import _coerce_parameters

    tool = FunctionDefinition(
        name="fn_config",
        description=".",
        parameters={
            "cfg": _field("object", properties={
                "score": _field("number"),
                "ids": _field("array", items=_field("number")),
            }),
        },
        returns=_field("number"),
    )

    coerced = _coerce_parameters(
        {"cfg": {"score": 3, "ids": [1, 2.5]}}, tool
    )

    assert coerced == {"cfg": {"score": 3.0, "ids": [1.0, 2.5]}}
