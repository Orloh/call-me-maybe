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
    sample_functions
) -> None:
    """
    Proves that the pipeline correctly compiles ONLY the router table during init.
    """
    mock_compile.return_value = {"name": "mock_name_fsm"}

    pipeline = FunctionCallingPipeline(mock_model, mock_trie, sample_functions)

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
    sample_functions
):
    """
    Proves the "Happy Path": Phase 1 outputs a valid name, the pipeline finds the 
    matching function, and Phase 2 successfully extracts the parameters.
    """
    mock_compile_router.return_value = {"name": "mock_name_fsm"}
    mock_compile_extractor.return_value ={"location": "mock_location_fsm"}

    pipeline = FunctionCallingPipeline(mock_model, mock_trie, sample_functions)

    mock_prompt_builder.build_function_name_prompt.return_value = "Phase 1 Prompt"
    mock_prompt_builder.build_parameters_prompt.return_value = "Phase 2 Prompt"

    mock_gen_instance = mock_generator_class.return_value
    mock_gen_instance.generate.side_effect = [
        '{"name": "get_weather"}',
        '{"location": "Madrid"}'
    ]

    user_prompt = "What is the weather in Madrid?"
    result = pipeline.process_prompt(user_prompt, sample_functions)

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
    sample_functions
):
    """
    Proves that if Phase 1 somehow generates a function name that does not exist 
    in available_functions, the pipeline intercepts it and raises a ValueError.
    """
    mock_compile_router.return_value = {"name": "mock_router_fm"}

    pipeline = FunctionCallingPipeline(mock_model, mock_trie, sample_functions)

    mock_gen_instance = mock_generator_class.return_value
    mock_gen_instance.generate.return_value = '{"name": "get_sports_scores"}'

    with pytest.raises(ValueError, match="LLM hallucinated function: get_sports_scores"):
        pipeline.process_prompt("Who won the game?", sample_functions)

    mock_compile_extractor.assert_not_called()
    assert mock_gen_instance.generate.call_count == 1
