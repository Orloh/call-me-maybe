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

        system = (
            "You are a function router. "
            "Select the single best function that matches "
            "the user request.\n\n"
            "Available Functions:\n"
            f"{function_catalog}\n"
            "Examples:\n"
            'User: "Add 2 and 3"\n'
            'Output: {"name": "fn_add_numbers"}\n\n'
            'User: "Greet Alice"\n'
            'Output: {"name": "fn_greet"}\n\n'
            "User: \"Reverse 'hello'\"\n"
            'Output: {"name": "fn_reverse_string"}\n\n'
            'User: "Square root of 16"\n'
            'Output: {"name": "fn_get_square_root"}\n\n'
            "User: \"Replace 'a' with 'b'\"\n"
            'Output: {"name": "fn_substitute_string_with_regex"}\n\n'
            'User: "What\'s the weather?"\n'
            'Output: {"name": "none"}\n'
        )
        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
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

        system = (
            "You are a parameter extractor.\n"
            "Extract the required parameters for the target function "
            "from the user request.\n\n"
            f"Target Function: {target_function.name}\n"
            f"Function Description: {target_function.description}\n"
            "Extract all parameters literally except regex patterns.\n"
            "When a parameter expects a regex pattern, generate a "
            "character class like [0-9]+ for digits instead of "
            "extracting a literal value from the input.\n"
            "Example: 'replace all numbers' → regex: [0-9]+\n\n"
            f"Parameters Schema:\n{schema_str}\n"
        )
        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
