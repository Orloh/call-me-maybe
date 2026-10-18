import pytest
import json
from unittest.mock import MagicMock, patch

from src.engine.pipeline import FunctionCallingPipeline
from src.schema import FunctionDefinition, ParameterField, FunctionCallResult


@pytest.fixture
def mock_model():
    """Provides a dummy LLM model."""
    return MagicMock()


@pytest.fixture
def mock_trie():
    """Provides a dummy Trie."""
    return MagicMock()


@pytest.fixture
def mock_token_to_decoded():
    """Provides a dummy token-to-decoded mapping."""
    return {}


@pytest.fixture
def sample_functions():
    """Provides a standard list of FunctionDefinitions for testing."""
    return [
        FunctionDefinition(
            name="get_weather",
            description="Get the weather for a location",
            parameters={
                "location": ParameterField(type="string")
            },
            returns=ParameterField(type="string")
        )
    ]


@patch("src.engine.pipeline.SchemaCompiler.compile_router_table")
def test_pipeline_initialization(
    mock_compile,
    mock_model,
    mock_trie,
    mock_token_to_decoded,
    sample_functions
) -> None:
    """
    Proves that the pipeline correctly compiles ONLY the router table during init.
    """
    mock_compile.return_value = {"name": "mock_name_fsm"}

    pipeline = FunctionCallingPipeline(
        mock_model, mock_trie, mock_token_to_decoded, sample_functions
    )

    mock_compile.assert_called_once_with(sample_functions)

    assert pipeline.router_table == {"name": "mock_name_fsm"}


@patch("src.engine.pipeline.PromptBuilder")
@patch("src.engine.pipeline.ConstrainedGenerator")
@patch("src.engine.pipeline.SchemaCompiler.compile_extractor_table")
@patch("src.engine.pipeline.SchemaCompiler.compile_router_table")
def test_process_prompt_success(
    mock_compile_router,
    mock_compile_extractor,
    mock_generator_class,
    mock_prompt_builder,
    mock_model,
    mock_trie,
    mock_token_to_decoded,
    sample_functions
):
    """
    Proves the "Happy Path": Phase 1 outputs a valid name, the pipeline finds the 
    matching function, and Phase 2 successfully extracts the parameters.
    """
    mock_compile_router.return_value = {"name": "mock_name_fsm"}
    mock_compile_extractor.return_value ={"location": "mock_location_fsm"}

    pipeline = FunctionCallingPipeline(
        mock_model, mock_trie, mock_token_to_decoded, sample_functions
    )

    mock_prompt_builder.build_function_name_prompt.return_value = "Phase 1 Prompt"
    mock_prompt_builder.build_parameters_prompt.return_value = "Phase 2 Prompt"

    mock_gen_instance = mock_generator_class.return_value
    mock_gen_instance.generate.side_effect = [
        '{"name": "get_weather"}',
        '{"location": "Madrid"}'
    ]

    user_prompt = "What is the weather in Madrid?"
    result = pipeline.process_prompt(user_prompt)

    assert isinstance(result, FunctionCallResult)
    assert result.prompt == user_prompt
    assert result.name == "get_weather"
    assert result.parameters == {"location": "Madrid"}

    assert mock_gen_instance.generate.call_count == 2


@patch("src.engine.pipeline.ConstrainedGenerator")
@patch("src.engine.pipeline.SchemaCompiler.compile_extractor_table")
@patch("src.engine.pipeline.SchemaCompiler.compile_router_table")
def test_process_prompt_hallucination_trap(
    mock_compile_router,
    mock_compile_extractor,
    mock_generator_class,
    mock_model,
    mock_trie,
    mock_token_to_decoded,
    sample_functions
):
    """
    Proves that if Phase 1 generates a function name that does not exist,
    the pipeline returns a 'none' result instead of raising an error.
    """
    mock_compile_router.return_value = {"name": "mock_router_fm"}

    pipeline = FunctionCallingPipeline(
        mock_model, mock_trie, mock_token_to_decoded, sample_functions
    )

    mock_gen_instance = mock_generator_class.return_value
    mock_gen_instance.generate.return_value = '{"name": "get_sports_scores"}'

    result = pipeline.process_prompt("Who won the game?")

    assert isinstance(result, FunctionCallResult)
    assert result.name == "none"
    assert result.parameters == {}

    # Extractor tables are compiled eagerly in __init__, once per function;
    # Phase 2 itself must not generate for an unroutable prompt.
    assert mock_compile_extractor.call_count == len(sample_functions)
    assert mock_gen_instance.generate.call_count == 1


@patch("src.engine.pipeline.ConstrainedGenerator")
@patch("src.engine.pipeline.SchemaCompiler.compile_router_table")
def test_process_prompt_propagates_max_tokens_error(
    mock_compile_router,
    mock_generator_class,
    mock_model,
    mock_trie,
    mock_token_to_decoded,
    sample_functions,
):
    """No retry: a MaxTokensExceededError propagates after a single call."""
    from src.automata import PDAState
    from src.engine import MaxTokensExceededError

    mock_compile_router.return_value = {"name": "mock_name_fsm"}

    pipeline = FunctionCallingPipeline(
        mock_model, mock_trie, mock_token_to_decoded, sample_functions
    )

    mock_gen_instance = mock_generator_class.return_value
    mock_gen_instance.generate.side_effect = MaxTokensExceededError(
        partial_text="", max_new_tokens=50,
        final_state=PDAState.EXPECTING_OBJECT_START,
    )

    with pytest.raises(MaxTokensExceededError):
        pipeline.process_prompt("hi")

    assert mock_gen_instance.generate.call_count == 1


@patch("src.engine.pipeline.ConstrainedGenerator")
@patch("src.engine.pipeline.SchemaCompiler.compile_extractor_table")
@patch("src.engine.pipeline.SchemaCompiler.compile_router_table")
def test_process_prompt_coerces_numbers_to_float(
    mock_compile_router,
    mock_compile_extractor,
    mock_generator_class,
    mock_model,
    mock_trie,
    mock_token_to_decoded
):
    """`number` parameters must surface as floats, even if emitted as ints."""
    tool = FunctionDefinition(
        name="fn_add_numbers",
        description="Add two numbers.",
        parameters={
            "a": ParameterField(type="number"),
            "b": ParameterField(type="number")
        },
        returns=ParameterField(type="number")
    )
    mock_compile_router.return_value = {"name": "mock_name_fsm"}
    mock_compile_extractor.return_value = {"a": "mock", "b": "mock"}

    pipeline = FunctionCallingPipeline(
        mock_model, mock_trie, mock_token_to_decoded, [tool]
    )

    mock_gen_instance = mock_generator_class.return_value
    mock_gen_instance.generate.side_effect = [
        '{"name": "fn_add_numbers"}',
        '{"a": 6, "b": 3.5}'
    ]

    result = pipeline.process_prompt("Add 6 and 3.5")

    assert result.parameters == {"a": 6.0, "b": 3.5}
    assert isinstance(result.parameters["a"], float)


@patch("src.engine.pipeline.ConstrainedGenerator")
@patch("src.engine.pipeline.SchemaCompiler.compile_extractor_table")
@patch("src.engine.pipeline.SchemaCompiler.compile_router_table")
def test_process_prompt_keeps_integers_as_int(
    mock_compile_router,
    mock_compile_extractor,
    mock_generator_class,
    mock_model,
    mock_trie,
    mock_token_to_decoded
):
    """`integer` parameters stay ints."""
    tool = FunctionDefinition(
        name="fn_count",
        description="Count things.",
        parameters={"n": ParameterField(type="integer")},
        returns=ParameterField(type="integer")
    )
    mock_compile_router.return_value = {"name": "mock_name_fsm"}
    mock_compile_extractor.return_value = {"n": "mock"}

    pipeline = FunctionCallingPipeline(
        mock_model, mock_trie, mock_token_to_decoded, [tool]
    )

    mock_gen_instance = mock_generator_class.return_value
    mock_gen_instance.generate.side_effect = [
        '{"name": "fn_count"}',
        '{"n": 5}'
    ]

    result = pipeline.process_prompt("Count 5")

    assert result.parameters == {"n": 5}
    assert isinstance(result.parameters["n"], int)


def test_coerce_parameters_rejects_key_mismatch():
    """Unknown/missing parameter keys must surface, not be dropped."""
    from src.engine.pipeline import _coerce_parameters

    tool = FunctionDefinition(
        name="fn_one",
        description=".",
        parameters={"a": ParameterField(type="number")},
        returns=ParameterField(type="number"),
    )

    with pytest.raises(ValueError, match="do not match schema"):
        _coerce_parameters({"a": 1, "extra": 2}, tool)

    with pytest.raises(ValueError, match="do not match schema"):
        _coerce_parameters({}, tool)


def test_coerce_parameters_recurses_into_arrays():
    """Array-of-number parameters get float elements."""
    from src.engine.pipeline import _coerce_parameters

    tool = FunctionDefinition(
        name="fn_sum_list",
        description="Sum a list.",
        parameters={
            "numbers": ParameterField(
                type="array", items=ParameterField(type="number")
            )
        },
        returns=ParameterField(type="number")
    )

    coerced = _coerce_parameters({"numbers": [1, 2.5, 3]}, tool)

    assert coerced == {"numbers": [1.0, 2.5, 3.0]}
