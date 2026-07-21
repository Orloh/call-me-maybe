from typing import Callable 
from .primitives import BaseFSM, NumberFSM, StringLiteralFSM, ExactMatchFSM
from src.schema import FunctionDefinition


type CompiledSchema = dict[str, Callable[[], BaseFSM]] 


class UnsupportedSchemaTypeError(Exception):
    """
    Raised when the SchemaCompiler encouters an unsuported a type error.
    """
    pass


class SchemaCompiler:
    def __init__(self) -> None:
        """
        Prevents instantiation of utility class.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} cannot be instantiated."
        )

    @classmethod
    def compile_router_table(cls, tools: list[FunctionDefinition]) -> CompiledSchema:
        """
        Creates a routing dictionary that ONLY accpets the "name" key
        and validates against the exact names of the available functions.
        """
        valid_function_names = [f'"{tool.name}"' for tool in tools]

        if not valid_function_names:
            return {}

        return {
            "name": lambda names=valid_function_names: ExactMatchFSM(names)
        }

    @classmethod
    def compile_extractor_table(cls, tool: FunctionDefinition) -> CompiledSchema:
        """
        Creates a routing dictionary strictly limited to the parameters of
        ONE specific target function.
        """
        routing_table: CompiledSchema = {}

        for param_key, param_field in tool.parameters.items():
            routing_table[param_key] = cls._map_type_to_fsm(param_field.type)

        return routing_table
    
    @classmethod
    def compile_tools(cls, tools: list[FunctionDefinition]) -> CompiledSchema:
        """
        Parses a list of FunctionDefinitions and creates a universal
        routing dictionary mappin JSON keys to FSM generators.
        """
        routing_table: CompiledSchema = {}
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
                    "Fatal internal error: "
                    f"Pydantic allowed an unsupported type '{data_type}'."
                ) 
