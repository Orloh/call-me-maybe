from typing import Callable, Any
from .primitives import BaseFSM, NumberFSM, StringLiteralFSM, ExactMatchFSM
from src.schema import FunctionDefinition

class UnsupportedSchemaTypeError(Exception):
    """
    Raised when the SchemaCompiler encouters an unsuported a type error.
    """
    pass


class SchemaCompiler:
    """
    Translates a list of FunctionDefinitions into a high-speed FSM routing table.
    """

    @classmethod
    def compile_tools(cls, tools: list[FunctionDefinition]) -> dict[str, Callable[[], BaseFSM]]:
        """
        Parses a list of FunctionDefinitions and creates a universal
        routing dictionary mappin JSON keys to FSM generators.
        """
        routing_table: dict[str, Callable[[], BaseFSM]] = {}
        valid_function_names: list[str] = []

        for tool in tools:
            valid_function_names.append(f'"{tool.name}"')
            
            for param_key,  param_field in tool.parameters.items():
                routing_table[param_key] = cls._map_type_to_fsm(param_field.type)

        if valid_function_names:
            routing_table["name"] = lambda names=valid_function_names: ExactMatchFSM(names)

        return routing_table

    @staticmethod
    def _map_type_to_fsm(data_type: str) -> Callable[[], BaseFSM]:
        """
        Translates specific ParameterField string types into lambda factories.
        """
        match data_type.lower():
            case "string":
                return lambda: StringLiteralFSM()

            case "number" | "integer" | "float":
                return lambda: NumberFSM()

            case "boolean":
                return lambda: ExactMatchFSM(["true", "false"])

            case "null":
                return lambda: ExactMatchFSM(["null"])

            case _:
                raise UnsupportedSchemaTypeError(
                    f"Fatal internal error: Pydantic allowed an unsupported type '{data_type}'."
                ) 
