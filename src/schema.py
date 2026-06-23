from pydantic import BaseModel

class ParameterField(BaseModel):
    type: str

class FunctionDefinition(BaseModel):
    """Represents a single available function the LLM can call."""
    name: str
    description: str
    parameters: dict[str, ParameterField] 
    returns: ParameterField
