from pydantic import BaseModel, ConfigDict

class ParameterField(BaseModel):
    """Represents the type definition of a specific parameter."""
    model_config = ConfigDict(extra='forbid')

    type: str

class FunctionDefinition(BaseModel):
    """Represents a single available function the LLM can call."""
    model_config = ConfigDict(extra='forbid')
    
    name: str
    description: str
    parameters: dict[str, ParameterField] 
    returns: ParameterField
