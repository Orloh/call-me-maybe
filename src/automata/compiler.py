import json
from dataclasses import dataclass
from typing import Callable, Mapping
from .primitives import (
    BaseFSM,
    NumberFSM,
    StringLiteralFSM,
    ExactMatchFSM
)
from src.schema import FunctionDefinition, ParameterField


@dataclass(frozen=True)
class ArraySpec:
    """Describes an array parameter: the spec of its element type."""
    item: "ValueSpec"


@dataclass(frozen=True)
class ObjectSpec:
    """Describes an object parameter: its (all-required) fields."""
    fields: Mapping[str, "ValueSpec"]


type ValueSpec = Callable[[], BaseFSM] | ArraySpec | ObjectSpec

type CompiledSchema = dict[str, ValueSpec]


class UnsupportedSchemaTypeError(Exception):
    """
    Raised when the SchemaCompiler encouters an unsuported a type error.
    """
    pass


def _make_exact_match_factory(names: list[str]) -> Callable[[], BaseFSM]:
    """
    Builds an FSM factory that produces a fresh ExactMatchFSM over the
    given candidate strings on every call.
    """
    def factory() -> BaseFSM:
        return ExactMatchFSM(names)

    return factory


class SchemaCompiler:
    """
    Utility class that compiles FunctionDefinition pydantic models into
    PDA-compatible routing tables.

    Each method produces a CompiledSchema — a dict mapping JSON keys to
    either FSM factories or NestedObjectSchema wrappers — used by the
    JSONPushdownAutomaton during constrained generation.
    """
    def __init__(self) -> None:
        """
        Prevents instantiation of utility class.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} cannot be instantiated."
        )

    @classmethod
    def compile_router_table(
        cls,
        tools: list[FunctionDefinition]
    ) -> CompiledSchema:
        """
        Creates a routing dictionary that ONLY accpets the "name" key
        and validates against the exact names of the available functions.
        """
        valid_function_names = [
            json.dumps(tool.name, ensure_ascii=False) for tool in tools
        ]

        if not valid_function_names:
            return {}

        return {
            "name": _make_exact_match_factory(valid_function_names)
        }

    @classmethod
    def compile_extractor_table(
        cls,
        tool: FunctionDefinition
    ) -> CompiledSchema:
        """
        Creates a routing dictionary strictly limited to the parameters of
        ONE specific target function.
        """
        routing_table: CompiledSchema = {}

        for param_key, param_field in tool.parameters.items():
            routing_table[param_key] = cls._compile_field(param_field)

        return routing_table

    @classmethod
    def _compile_field(cls, field: ParameterField) -> ValueSpec:
        """
        Translates a ParameterField into a ValueSpec: either a scalar FSM
        factory or an ArraySpec wrapping the element FSM factory.
        """
        if field.type == "array":
            if field.items is None:
                raise UnsupportedSchemaTypeError(
                    "Fatal internal error: array parameter without 'items'."
                )
            return ArraySpec(item=cls._compile_field(field.items))

        if field.type == "object":
            if field.properties is None:
                raise UnsupportedSchemaTypeError(
                    "Fatal internal error: object parameter without "
                    "'properties'."
                )
            return ObjectSpec(fields={
                key: cls._compile_field(sub_field)
                for key, sub_field in field.properties.items()
            })

        return cls._map_type_to_fsm(field.type)

    @staticmethod
    def _map_type_to_fsm(data_type: str) -> Callable[[], BaseFSM]:
        """
        Translates specific ParameterField string types into lambda factories.
        """
        match data_type.lower():
            case "string":
                return lambda: StringLiteralFSM()

            case "integer":
                return lambda: NumberFSM(mode="integer")

            case "number" | "float":
                return lambda: NumberFSM(mode="number")

            case "boolean":
                return lambda: ExactMatchFSM(["true", "false"])

            case "null":
                return lambda: ExactMatchFSM(["null"])

            case _:
                raise UnsupportedSchemaTypeError(
                    "Fatal internal error: "
                    f"Pydantic allowed an unsupported type '{data_type}'."
                )
