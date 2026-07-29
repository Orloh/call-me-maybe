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
            "Task: You are a function selector. Given a user request, "
            "output the name of the best matching function "
            "as a JSON object.\n\n"
            "Available functions:\n"
            f"{function_catalog}"
            "\n"
            "Examples:\n"
            'User: "Add 2 and 3"\n'
            'Output: {"name": "fn_add_numbers"}\n\n'
            'User: "Greet Alice"\n'
            'Output: {"name": "fn_greet"}\n\n'
            'User: "Reverse \'opencode\'"\n'
            'Output: {"name": "fn_reverse_string"}\n\n'
            'User: "Square root of 25"\n'
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
        schema_lines = "\n".join(
            f"  {key} ({field.type})"
            for key, field in target_function.parameters.items()
        )

        generic_examples = (
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
            "User: Substitute 'apple' with 'orange' in 'The apple fell'\n"
            "Function: fn_substitute_string_with_regex\n"
            'Output: {"source_string": "The apple fell", "regex": "apple", '
            '"replacement": "orange"}\n'
        )

        if target_function.name == "fn_reverse_string":
            contradiction = (
                "Despite the function name, "
                "do NOT reverse the string — extract it verbatim.\n"
            )
            counter_examples = (
                "\n"
                "User: Reverse 'Testing'\n"
                "Function: fn_reverse_string\n"
                'Output: {"s": "Testing"}\n'
                "\n"
                "User: Reverse 'abc'\n"
                "Function: fn_reverse_string\n"
                'Output: {"s": "abc"}\n'
                "\n"
                "User: Reverse 'xyz'\n"
                "Function: fn_reverse_string\n"
                'Output: {"s": "xyz"}\n'
            )
        elif target_function.name == "fn_get_square_root":
            contradiction = ""
            counter_examples = (
                "\n"
                "User: Square root of 25\n"
                "Function: fn_get_square_root\n"
                'Output: {"a": 25}\n'
                "\n"
                "User: What is the square root of 49\n"
                "Function: fn_get_square_root\n"
                'Output: {"a": 49}\n'
                "\n"
                "User: Square root of 9\n"
                "Function: fn_get_square_root\n"
                'Output: {"a": 9}\n'
                "\n"
                "User: Square root of 100\n"
                "Function: fn_get_square_root\n"
                'Output: {"a": 100}\n'
            )
        else:
            contradiction = ""
            counter_examples = ""

        return (
            "Task: You are a parameter extractor. "
            "Do NOT solve the problem or calculate the answer. "
            "Only extract the arguments from the user request.\n\n"
            f"Function: {target_function.name}\n"
            f"{contradiction}"
            f"Parameters:\n{schema_lines}\n\n"
            "For regex parameters, generate a generic pattern:\n"
            "- Class-based (numbers, digits, vowels, whitespace) → "
            "[0-9]+, [aeiouAEIOU], \\s+\n"
            "- Specific quoted word (e.g. 'cat') → use that word "
            "literally\n\n"
            f"{generic_examples}"
            f"{counter_examples}\n"
            "Reminder: Extract the value exactly as written "
            "— do NOT transform it.\n\n"
            f"User: \"{user_prompt}\"\n"
            f"Function: {target_function.name}\n"
            "Output:\n"
        )
