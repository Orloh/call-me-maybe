import json
from src.schema import FunctionDefinition

class PromptBuilder:
    """
    Constructs a flat-text completion prompts optimized for raw next-token
    prediction models, priming themm for constrained JSON generation.
    """
    @staticmethod
    def build_function_name_prompt(
        user_prompt: str,
        available_functions: list[FunctionDefinition]
    ) -> str:
        """
        Formats the prompt to determine which funtion the user wants to call.
        """
        pass

    @staticmethod
    def build_parameters_prompt(
        user_prompt: str,
        target_function: FunctionDefinition,
    ) -> str:
        """
        Formats the prompt to extract the specific parameter for the funtion.
        """
        pass
