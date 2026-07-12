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
