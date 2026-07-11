import json
import logging
from src.schema import FunctionDefinition, FunctionCallResult
from src.automata import JSONPushdownAutomaton, SchemaCompiler
from src.engine import ConstrainedGenerator, PromptBuilder
from llm_sdk import Small_LLM_Model
from src.trie import PrefixTrie

logger = logging.getLogger(__name__)

class FunctionCallingPipeline:
    """
    Orchestrates the two-phase generation process for single prompt.
    """

