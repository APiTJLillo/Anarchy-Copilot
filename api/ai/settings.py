from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Union, Literal

router = APIRouter()

class BaseModelConfig(BaseModel):
    model: str
    apiKey: Optional[str]
    maxTokens: int = Field(default=1024, gt=0)
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    topP: float = Field(default=1.0, ge=0.0, le=1.0)
    frequencyPenalty: float = Field(default=0.0, ge=0.0, le=2.0)
    presencePenalty: float = Field(default=0.0, ge=0.0, le=2.0)
    
    # Reasoning settings
    reasoningCapability: bool = Field(default=True)
    reasoningEffort: int = Field(default=80, ge=0, le=100)
    chainOfThought: bool = Field(default=True)
    selfReflection: bool = Field(default=True)
    
    # Response settings
    responseFormat: Literal["text", "json", "structured"] = "text"
    streamResponse: bool = Field(default=False)
    maxResponseSegments: int = Field(default=5, gt=0)
    
    # Specialized capabilities
    contextWindow: int = Field(default=4096, gt=0)
    embeddingDimension: Optional[int] = Field(default=None, gt=0)
    multilingualCapability: bool = Field(default=True)
    codeGeneration: bool = Field(default=True)
    
    # Performance settings
    cacheResults: bool = Field(default=True)
    timeoutMs: int = Field(default=30000, gt=0)
    retryAttempts: int = Field(default=3, ge=0)
    
    # Cost management
    costPerToken: float = Field(default=0.0002, ge=0.0)
    budgetLimit: Optional[float] = Field(default=None, gt=0.0)
    
    # Task-specific tuning
    defaultPersona: Optional[str]
    domainExpertise: Optional[List[str]]
    customPromptPrefix: Optional[str]

class ReasoningModelConfig(BaseModelConfig):
    reasoningStrategy: Literal["step-by-step", "tree-of-thought", "parallel"] = "step-by-step"
    verificationSteps: bool = Field(default=True)
    uncertaintyThreshold: float = Field(default=0.8, ge=0.0, le=1.0)
    maxReasoningSteps: int = Field(default=5, gt=0)
    feedbackLoop: bool = Field(default=True)

class CompletionModelConfig(BaseModelConfig):
    completionStyle: Literal["creative", "precise", "balanced"] = "balanced"
    stopSequences: List[str] = Field(default_factory=list)
    biasTokens: Dict[str, float] = Field(default_factory=dict)
    logitBias: Optional[Dict[str, float]] = Field(default_factory=dict)

class AISettings(BaseModel):
    defaultModel: str = "gpt-4"
    models: Dict[str, Union[BaseModelConfig, ReasoningModelConfig, CompletionModelConfig]]
    translationModel: Literal["neural", "basic"] = "neural"
    autoDetectLanguage: bool = True
    enableCulturalContext: bool = True
    defaultRegion: str = "US"
    enableCache: bool = True

# Default configurations
default_reasoning_config = ReasoningModelConfig(
    model="gpt-4",
    apiKey=None,
    defaultPersona=None,
    domainExpertise=None,
    customPromptPrefix=None,
    maxTokens=2048,
    temperature=0.7,
    reasoningCapability=True,
    reasoningEffort=80,
    chainOfThought=True,
    selfReflection=True,
    responseFormat="structured",
    contextWindow=8192,
    reasoningStrategy="step-by-step",
    verificationSteps=True,
    uncertaintyThreshold=0.8,
    maxReasoningSteps=5,
    feedbackLoop=True
)

default_completion_config = CompletionModelConfig(
    model="gpt-3.5-turbo",
    apiKey=None,
    defaultPersona=None,
    domainExpertise=None,
    customPromptPrefix=None,
    maxTokens=1024,
    temperature=0.9,
    topP=0.9,
    frequencyPenalty=0.2,
    presencePenalty=0.2,
    reasoningCapability=False,
    reasoningEffort=40,
    chainOfThought=False,
    selfReflection=False,
    responseFormat="text",
    streamResponse=True,
    contextWindow=4096,
    completionStyle="creative",
    stopSequences=["\n\n", "###"],
    biasTokens={},
    logitBias={}
)

# In-memory settings store (replace with database in production)
current_settings = AISettings(
    models={
        "gpt-4": default_reasoning_config,
        "gpt-3.5-turbo": default_completion_config
    }
)

@router.get("/api/settings/ai")
async def get_ai_settings():
    """Get current AI settings."""
    return current_settings

@router.put("/api/settings/ai")
async def update_ai_settings(settings: AISettings):
    """Update AI settings."""
    try:
        global current_settings
        current_settings = settings
        return {"status": "success", "settings": settings}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/settings/ai/reset")
async def reset_ai_settings():
    """Reset AI settings to defaults."""
    try:
        global current_settings
        current_settings = AISettings(
            models={
                "gpt-4": default_reasoning_config,
                "gpt-3.5-turbo": default_completion_config
            }
        )
        return {"status": "success", "settings": current_settings}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/settings/ai/models/{model_id}")
async def get_model_config(model_id: str):
    """Get configuration for a specific model."""
    if model_id not in current_settings.models:
        raise HTTPException(status_code=404, detail="Model not found")
    return current_settings.models[model_id]

@router.put("/api/settings/ai/models/{model_id}")
async def update_model_config(
    model_id: str,
    config: Union[BaseModelConfig, ReasoningModelConfig, CompletionModelConfig]
):
    """Update configuration for a specific model."""
    try:
        if model_id not in current_settings.models:
            raise HTTPException(status_code=404, detail="Model not found")
        current_settings.models[model_id] = config
        return {"status": "success", "config": config}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/settings/ai/models")
async def add_model_config(
    model_id: str,
    config: Union[BaseModelConfig, ReasoningModelConfig, CompletionModelConfig]
):
    """Add a new model configuration."""
    try:
        if model_id in current_settings.models:
            raise HTTPException(status_code=409, detail="Model already exists")
        current_settings.models[model_id] = config
        return {"status": "success", "config": config}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/api/settings/ai/models/{model_id}")
async def delete_model_config(model_id: str):
    """Delete a model configuration."""
    try:
        if model_id not in current_settings.models:
            raise HTTPException(status_code=404, detail="Model not found")
        if model_id == current_settings.defaultModel:
            raise HTTPException(status_code=400, detail="Cannot delete default model")
        del current_settings.models[model_id]
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
