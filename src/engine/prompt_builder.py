import json
from src.schema import FunctionDefinition


class PromptBuilder:
    def __init__(self) -> None:
        """
        Prevents instantiation of utility class.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} cannot be instantiated."
        )

    @staticmethod
    def build_function_name_prompt(
        user_prompt: str,
        available_functions: list[FunctionDefinition]
    ) -> str:
        """
        Formats the prompt to determine which funtion the user wants to call.
        """
        function_catalog = ""

        for func in available_functions:
            function_catalog += f"- {func.name}: {func.description}\n"

        return (
            "You are a function router. "
            "Select the single best function that matches "
            "the user request.\n\n"
            "Available Functions:\n"
            f"{function_catalog}\n"
            f"User Request: \"{user_prompt}\"\n\n"
            "Output ONLY a single JSON object with exactly one key 'name'.\n"
            "The value must be one of the function names listed above.\n"
            "Do NOT include any explanation, markdown, or extra whitespace.\n"
            'Example: {"name": "fn_add_numbers"}\n'
            "JSON Output:\n"
        )

    @staticmethod
    def build_parameters_prompt(
        user_prompt: str,
        target_function: FunctionDefinition,
    ) -> str:
        """
        Formats the prompt to extract the specific parameter for the funtion.
        """
        schema_dict = {
            key: field.model_dump()
            for key, field in target_function.parameters.items()
        }
        schema_str = json.dumps(schema_dict, indent=2)

        return (
            "You are a parameter extractor.\n"
            "Extract the required parameters for the target function "
            "from the user request.\n\n"
            f"Target Function: {target_function.name}\n"
            f"Function Description: {target_function.description}\n"
            "When a parameter expects a regex pattern, generate a generic "
            "regex (e.g. \\d+ for numbers, [0-9]+ for digits, "
            "[aeiouAEIOU] for vowels) instead of a literal value from "
            "the input. The regex must match all relevant items.\n\n"
            f"Parameters Schema:\n{schema_str}\n\n"
            f"User Request: \"{user_prompt}\"\n\n"
            "Output ONLY a single JSON object containing the extracted "
            "parameters.\n"
            "Use exactly the key names from the schema above.\n"
            "If a parameter value is not found in the request, output null.\n"
            "Do NOT include any explanation, markdown, or extra whitespace.\n"
            'Example: {"location": "Madrid", "days": 5}\n'
            "JSON Output:\n"
        )
