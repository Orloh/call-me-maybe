import json
import logging

from llm_sdk import Small_LLM_Model
from src.trie import PrefixTrie
from src.automata import JSONPushdownAutomaton, CompiledSchema, compile_tools
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
        available_functions: list[FunctionDefinition]
    ):
        self.model = model
        self.trie = trie
        self.extractor_table = compile_tools(available_functions)
        self.router_table = {
            "name": self.extractor_table["name"]
        }
    
    def process_prompt(
        self,
        user_prompt: str,
        available_functions: list[FunctionDefinition]
    ) -> FunctionCallResult:
        router_prompt = (
            PromptBuilder
            .build_function_name_prompt(user_prompt, available_functions)
        )

        router_pda = JSONPushdownAutomaton(self.router_table)
        router_gen = ConstrainedGenerator(self.model, router_pda, self.trie)

        router_json_str = router_gen.generate(router_prompt, max_new_tokens=50)
        selected_func_name = json.loads(router_json_str).get("name")

        target_function = next(
            (func for func in available_functions 
             if func.name == selected_func_name),
            None)
