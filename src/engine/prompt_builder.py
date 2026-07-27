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
            "Examples:\n"
            'User: "Add 2 and 3"\n'
            'Output: {"name": "fn_add_numbers"}\n\n'
            'User: "Greet Alice"\n'
            'Output: {"name": "fn_greet"}\n\n'
            'User: "Reverse \'hello\'"\n'
            'Output: {"name": "fn_reverse_string"}\n\n'
            'User: "Square root of 16"\n'
            'Output: {"name": "fn_get_square_root"}\n\n'
            'User: "Replace \'a\' with \'b\'"\n'
            'Output: {"name": "fn_substitute_string_with_regex"}\n\n'
            'User: "What\'s the weather?"\n'
            'Output: {"name": "none"}\n\n'
            f'User: "{user_prompt}"\n'
            "Output:"
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
            "from the user request.\n"
            "Extract all string values verbatim without transforming them "
            "(e.g. do not reverse, do not strip spaces).\n"
            "This rule applies to all parameters except regex parameters "
            "covered below.\n\n"
            f"Target Function: {target_function.name}\n"
            f"Function Description: {target_function.description}\n\n"
            "Regex rules:\n"
            "- For class-based replacements (numbers, digits, vowels, "
            "whitespace), generate a generic pattern (e.g. [0-9]+ for "
            "digits, [aeiouAEIOU] for vowels, \\s+ for whitespace).\n"
            "- For a specific quoted word (e.g. 'cat'), use that word "
            "literally as the regex.\n\n"
            "Examples:\n"
            "User: sum of 2 and 3\n"
            "Function: fn_add_numbers\n"
            'Output: {"a": 2, "b": 3}\n'
            "\n"
            "User: Replace all numbers in 'A1 B2 C3' with X\n"
            "Function: fn_substitute_string_with_regex\n"
            'Output: {"source_string": "A1 B2 C3", "regex": "[0-9]+", '
            '"replacement": "X"}\n'
            "\n"
            "User: Replace all vowels in 'hello' with *\n"
            "Function: fn_substitute_string_with_regex\n"
            'Output: {"source_string": "hello", "regex": "[aeiouAEIOU]", '
            '"replacement": "*"}\n'
            "\n"
            "User: Substitute 'cat' with 'dog' in 'The cat sat'\n"
            "Function: fn_substitute_string_with_regex\n"
            'Output: {"source_string": "The cat sat", "regex": "cat", '
            '"replacement": "dog"}\n'
            "\n"
            f"Parameters Schema:\n{schema_str}\n"
            "\n"
            f"User: \"{user_prompt}\"\n"
            f"Function: {target_function.name}\n"
            "Output:\n"
        )
