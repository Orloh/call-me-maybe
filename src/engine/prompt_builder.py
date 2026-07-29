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
            "If no function matches, output: {\"name\": \"none\"}\n\n"
            f"User: \"{user_prompt}\"\n"
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

        return (
            "Task: You are a parameter extractor. "
            "Do NOT solve the problem or calculate the answer. "
            "Only extract the arguments from the user request.\n\n"
            f"Function: {target_function.name}\n"
            f"Description: {target_function.description}\n"
            f"Parameters:\n{schema_lines}\n\n"
            "For regex parameters, generate a pattern instead of a literal:\n"
            "- Class-based (numbers, digits, vowels, whitespace) → "
            "[0-9]+, [aeiouAEIOU], \\s+\n"
            "- Specific quoted word (e.g. 'cat') → use that word "
            "literally\n\n"
            f"User: \"{user_prompt}\"\n"
            f"Function: {target_function.name}\n"
            "Output:\n"
        )
