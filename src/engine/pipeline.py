import json
import logging
from typing import Any

from llm_sdk import Small_LLM_Model
from src.trie import PrefixTrie
from src.automata import JSONPushdownAutomaton, SchemaCompiler
from src.schema import FunctionDefinition, FunctionCallResult, ParameterField
from src.engine import (
    ConstrainedGenerator,
    PromptBuilder,
)

logger = logging.getLogger(__name__)


def _coerce_number(value: Any, field: ParameterField) -> Any:
    """
    Coerces a numeric schema field to float, recursing into array elements.

    The grammar accepts integer-looking JSON numbers, but `number`/`float`
    parameters must surface as floats (e.g. 6 -> 6.0). Booleans are left
    untouched (bool is an int subclass). `integer` stays an int.
    """
    if field.type in ("number", "float"):
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return float(value)
        return value

    if field.type == "array" and field.items is not None:
        if isinstance(value, list):
            return [_coerce_number(item, field.items) for item in value]

    if field.type == "object" and field.properties is not None:
        if isinstance(value, dict):
            return {
                key: _coerce_number(value.get(key), sub_field)
                for key, sub_field in field.properties.items()
            }

    return value


def _coerce_parameters(
    parameters: dict[str, Any], tool: FunctionDefinition
) -> dict[str, Any]:
    """
    Applies schema-driven numeric coercion to extracted parameters.

    The extracted key set must match the tool's schema exactly; a mismatch
    indicates a grammar/compiler regression and must surface rather than
    silently dropping unknown keys.
    """
    expected = set(tool.parameters)
    actual = set(parameters)
    if actual != expected:
        raise ValueError(
            f"Extracted parameters for {tool.name!r} do not match schema: "
            f"expected {sorted(expected)}, got {sorted(actual)}."
        )

    return {
        key: _coerce_number(parameters[key], field)
        for key, field in tool.parameters.items()
    }


class FunctionCallingPipeline:
    """
    Orchestrates the two-phase generation process for single prompt.
    """
    def __init__(
        self,
        model: Small_LLM_Model,
        trie: PrefixTrie,
        token_to_decoded: dict[int, str],
        available_functions: list[FunctionDefinition],
        trace: bool = False
    ):
        self.model = model
        self.trie = trie
        self.token_to_decoded = token_to_decoded
        self.available_functions = available_functions
        self.router_table = SchemaCompiler.compile_router_table(
            available_functions
        )
        # Compile each function's extractor table once, not per prompt.
        self.functions_by_name = {
            func.name: func for func in available_functions
        }
        self.extractor_tables = {
            func.name: SchemaCompiler.compile_extractor_table(func)
            for func in available_functions
        }
        self.trace = trace

    def process_prompt(self, user_prompt: str) -> FunctionCallResult:
        """
        Run the two-phase generation pipeline for a single user prompt.

        Phase 1 (routing) generates the function name.
        Phase 2 (extraction) generates the parameter JSON for the chosen
        function.

        Args:
            user_prompt: The natural-language user request.

        Returns:
            A FunctionCallResult with the selected function name
            and extracted parameters.
        """
        # PHASE 1: Function Routing
        router_prompt = PromptBuilder.build_function_name_prompt(
            user_prompt, self.available_functions
        )
        router_pda = JSONPushdownAutomaton(self.router_table)
        router_gen = ConstrainedGenerator(
            self.model,
            router_pda,
            self.trie,
            self.token_to_decoded,
            trace=self.trace
        )

        router_json_str = (
            router_gen.generate(router_prompt, max_new_tokens=50).strip()
        )
        logger.info("Router output: %s", router_json_str)
        selected_func_name = json.loads(router_json_str).get("name")

        target_function = self.functions_by_name.get(selected_func_name)

        if not target_function:
            # Defensive: the router grammar only offers catalog names, so this
            # is reachable only with an empty catalog or an internal mismatch.
            logger.warning(
                "No matching function for '%s'. Returning 'none'.",
                selected_func_name
            )
            return FunctionCallResult(
                prompt=user_prompt,
                name="none",
                parameters={}
            )
        # PHASE 2: Parameter Extraction
        extractor_prompt = PromptBuilder.build_parameters_prompt(
            user_prompt,
            target_function
        )

        extractor_table = self.extractor_tables[target_function.name]

        extractor_pda = JSONPushdownAutomaton(extractor_table)
        extractor_gen = ConstrainedGenerator(
            self.model,
            extractor_pda,
            self.trie,
            self.token_to_decoded,
            trace=self.trace
        )

        extractor_json_str = (
            extractor_gen.generate(extractor_prompt, max_new_tokens=300)
            .strip()
        )
        logger.info("Extractor output: %s", extractor_json_str)
        extracted_parameters = _coerce_parameters(
            json.loads(extractor_json_str), target_function
        )

        return FunctionCallResult(
            prompt=user_prompt,
            name=target_function.name,
            parameters=extracted_parameters
        )
