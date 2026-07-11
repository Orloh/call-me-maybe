import json
from pathlib import Path
from llm_sdk import Small_LLM_Model
from src.trie import PrefixTrie
from src.automata import SchemaCompiler

def initialize_system_dependencies() -> tuple[Small_LLM_Model, PrefixTrie, SchemaCompiler]:
    """
    Handles all the heavy lifting of bootin up the model and 
    the in-memory Trie
    """
