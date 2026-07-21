import json
from src.schema import FunctionDefinition

class PromptBuilder:
    def __init__(self):
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

        return(
            "You are an expert routing assistant.\n"
            "Determine which function best matches the user's request.\n\n"
            "Available Functions:\n"
            f"{function_catalog}\n"
            f"User Request: \"{user_prompt}\"\n\n"
            "Output exactly one JSON object witht the key 'name'"
            "containing the selected function.\n"
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
        schema_str = json.dumps(schema_dict, indent = 2)

        return(
            "You are an expert data extraction assitant.\n"
            "Extract the required parameters from the user's request based on the schema.\n"
            f"Target Function: {target_function.name}\n"
            f"Function Description: {target_function.description}\n"
            f"Parameters Schema:\n{schema_str}\n\n"
            f"User Request: \"{user_prompt}\"\n\n"
            "Output exatly one JSON object containing the extracted parameters.\n"
            "JSON Output:\n"
        )
