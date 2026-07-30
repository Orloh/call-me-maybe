import json
import logging

from llm_sdk import Small_LLM_Model
from src.trie import PrefixTrie
from src.automata import JSONPushdownAutomaton, SchemaCompiler
from src.schema import FunctionDefinition, FunctionCallResult
from src.engine import ConstrainedGenerator, PromptBuilder

logger = logging.getLogger(__name__)


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
        debug: bool = False
    ):
        self.model = model
        self.trie = trie
        self.token_to_decoded = token_to_decoded
        self.router_table = SchemaCompiler.compile_router_table(
            available_functions
        )
        self.debug = debug

    def process_prompt(
        self,
        user_prompt: str,
        available_functions: list[FunctionDefinition]
    ) -> FunctionCallResult:
        """
        Run the two-phase generation pipeline for a single user prompt.

        Phase 1 (routing) generates the function name.
        Phase 2 (extraction) generates the parameter JSON for the chosen
        function.

        Args:
            user_prompt: The natural-language user request.
            available_functions: The list of callable function definitions.

        Returns:
            A FunctionCallResult with the selected function name
            and extracted parameters.
        """
        # PHASE 1: Function Routing
        router_prompt = (
            PromptBuilder
            .build_function_name_prompt(user_prompt, available_functions)
        )
        router_pda = JSONPushdownAutomaton(self.router_table)
        router_gen = ConstrainedGenerator(
            self.model,
            router_pda,
            self.trie,
            self.token_to_decoded,
            debug=self.debug
        )

        router_json_str = (
            router_gen.generate(router_prompt, max_new_tokens=50)
            .strip()
        )
        logger.info("Router output: %s", router_json_str)
        selected_func_name = json.loads(router_json_str).get("name")

        target_function = next(
            (func for func in available_functions
             if func.name == selected_func_name),
            None)

        if not target_function:
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

        extractor_table = SchemaCompiler.compile_extractor_table(
            target_function
        )

        extractor_pda = JSONPushdownAutomaton(extractor_table)
        extractor_gen = ConstrainedGenerator(
            self.model,
            extractor_pda,
            self.trie,
            self.token_to_decoded,
            debug=self.debug
        )

        extractor_json_str = extractor_gen.generate(
            extractor_prompt,
            max_new_tokens=300
        ).strip()
        logger.info("Extractor output: %s", extractor_json_str)
        extracted_parameters = json.loads(extractor_json_str)

        return FunctionCallResult(
            prompt=user_prompt,
            name=target_function.name,
            parameters=extracted_parameters
        )
