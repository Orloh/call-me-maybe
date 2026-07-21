import pytest
from src.schema import FunctionDefinition, ParameterField
from src.automata import SchemaCompiler, NumberFSM, StringLiteralFSM, ExactMatchFSM, UnsupportedSchemaTypeError


def test_compile_extractor_table():
    """
    Proves that the compiler maps Pydantic schema types
    to their corresponding FSM lambda factories for ONE specific target function.
    """
    mock_tool = FunctionDefinition(
        name="get_weather",
        description="Fetches the weather.",
        parameters={
            "location": ParameterField(type="string"),
            "days": ParameterField(type="integer"),
            "is_metric": ParameterField(type="boolean")
        },
        returns=ParameterField(type="string")
    )

    routing_table = SchemaCompiler.compile_extractor_table(mock_tool)

    assert "name" not in routing_table

    assert "location" in routing_table
    assert "days" in routing_table
    assert "is_metric" in routing_table

    assert isinstance(routing_table["location"](), StringLiteralFSM)
    assert isinstance(routing_table["days"](), NumberFSM)
    assert isinstance(routing_table["is_metric"](), ExactMatchFSM)


def test_compile_router_table():
    """
    Proves that the compiler builds the router table 
    strictly containing the ExactMatchFSM for the available function names.
    """
    tool_1 = FunctionDefinition(
        name="fn_add",
        description=".",
        parameters={},
        returns=ParameterField(type="number")
    )

    tool_2 = FunctionDefinition(
        name="fn_subtract",
        description=".",
        parameters={},
        returns=ParameterField(type="number")
    )

    routing_table = SchemaCompiler.compile_router_table([tool_1, tool_2])

    assert "name" in routing_table
    assert len(routing_table) == 1 # ONLY contains 'name'

    name_fsm = routing_table["name"]()
    assert isinstance(name_fsm, ExactMatchFSM)
    assert '"fn_add"' in name_fsm.active_candidates
    assert '"fn_subtract"' in name_fsm.active_candidates


def test_schema_compiler_raises_on_invalid_type():
    """
    Proves the internal fail-safe raises the custom error exception.
    (Pydantic prevents this at the file-loading stage, testing the
     private method ensures the fail-safe works).
    """
    with pytest.raises(UnsupportedSchemaTypeError) as exc_info:
        SchemaCompiler._map_type_to_fsm("some_hallucinated_type")

    assert "Fatal internal error: Pydantic allowed an unsupported type" in str(exc_info.value)
