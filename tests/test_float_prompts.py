"""
Float-focused prompt tests.

Exercises prompts whose extraction involves decimal numbers: the fixtures
live in data/input/edge_floats.json. The pipeline is driven with a scripted
generator so no model is needed; the router/extractor tables and prompt
builder run for real.
"""
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.automata import (
    JSONPushdownAutomaton,
    NumberFSM,
    PDAState,
    SchemaCompiler,
)
from src.engine.pipeline import FunctionCallingPipeline
from src.io_manager import load_prompts
from src.schema import FunctionDefinition, ParameterField

_FLOAT_PROMPTS = (
    Path(__file__).resolve().parent.parent
    / "data" / "input" / "edge_floats.json"
)

_FUNCTIONS = [
    FunctionDefinition(
        name="fn_add_numbers",
        description="Add two numbers together and return their sum.",
        parameters={
            "a": ParameterField(type="number"),
            "b": ParameterField(type="number"),
        },
        returns=ParameterField(type="number"),
    ),
    FunctionDefinition(
        name="fn_get_square_root",
        description="Calculate the square root of a number.",
        parameters={"a": ParameterField(type="number")},
        returns=ParameterField(type="number"),
    ),
]

# (prompt, routed function, raw extractor JSON, expected coerced params)
_FLOAT_CASES = [
    ("Add 1.5 and 2.5", "fn_add_numbers",
     {"a": 1.5, "b": 2.5}, {"a": 1.5, "b": 2.5}),
    ("Add 0.1 and 0.2", "fn_add_numbers",
     {"a": 0.1, "b": 0.2}, {"a": 0.1, "b": 0.2}),
    ("Add 2 and 3", "fn_add_numbers",
     {"a": 2, "b": 3}, {"a": 2.0, "b": 3.0}),
    ("Add -1.5 and 4.75", "fn_add_numbers",
     {"a": -1.5, "b": 4.75}, {"a": -1.5, "b": 4.75}),
    ("What is the square root of 2.25?", "fn_get_square_root",
     {"a": 2.25}, {"a": 2.25}),
    ("Calculate the square root of 0.25", "fn_get_square_root",
     {"a": 0.25}, {"a": 0.25}),
    ("What is the square root of 16?", "fn_get_square_root",
     {"a": 16}, {"a": 16.0}),
]


def test_float_prompts_fixture_loads() -> None:
    prompts = load_prompts(_FLOAT_PROMPTS)
    assert len(prompts) == 7
    assert any("." in item.prompt for item in prompts)


def test_fixture_prompts_are_covered_by_cases() -> None:
    fixture = {item.prompt for item in load_prompts(_FLOAT_PROMPTS)}
    covered = {case[0] for case in _FLOAT_CASES}
    assert fixture == covered


@pytest.mark.parametrize("prompt,func,raw,expected", _FLOAT_CASES)
def test_float_prompt_parameters_are_floats(
    monkeypatch: pytest.MonkeyPatch,
    prompt: str,
    func: str,
    raw: dict[str, float],
    expected: dict[str, float],
) -> None:
    mock_generator = MagicMock()
    mock_generator.return_value.generate.side_effect = [
        json.dumps({"name": func}),
        json.dumps(raw),
    ]
    monkeypatch.setattr(
        "src.engine.pipeline.ConstrainedGenerator", mock_generator
    )

    pipeline = FunctionCallingPipeline(MagicMock(), MagicMock(), {}, _FUNCTIONS)
    result = pipeline.process_prompt(prompt)

    assert result.name == func
    assert result.parameters == expected
    assert all(
        isinstance(value, float) for value in result.parameters.values()
    )


@pytest.mark.parametrize("literal", [
    "0.1", "1.5", "2.25", "-1.5", "4.75", "0.25", "-0.5", "0.0",
])
def test_number_fsm_accepts_decimal_literals(literal: str) -> None:
    fsm = NumberFSM(mode="number")
    for char in literal:
        assert fsm.advance(char), (literal, char)

    probe = fsm.clone()
    probe.advance(",")
    assert probe.is_terminal()


@pytest.mark.parametrize("literal", ["1.5", "-0.25"])
def test_integer_fsm_rejects_decimals(literal: str) -> None:
    fsm = NumberFSM(mode="integer")
    for char in literal:
        if char == ".":
            assert not fsm.advance(char)
            break
        assert fsm.advance(char)


def test_pda_accepts_decimal_parameters() -> None:
    tool = _FUNCTIONS[0]
    table = SchemaCompiler.compile_extractor_table(tool)

    pda = JSONPushdownAutomaton(table)
    for char in '{"a": 1.5, "b": -2.25}':
        assert pda.advance(char)
    assert pda.state == PDAState.TERMINAL


def test_pda_rejects_leading_dot_decimal() -> None:
    """.5 is not valid JSON; the NumberFSM rejects it at the first char."""
    tool = _FUNCTIONS[0]
    table = SchemaCompiler.compile_extractor_table(tool)

    pda = JSONPushdownAutomaton(table)
    for char in '{"a": .5':
        if not pda.advance(char):
            return
    pytest.fail("PDA accepted an invalid leading-dot number")
