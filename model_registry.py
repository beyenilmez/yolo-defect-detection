"""
Model Registry Module
Manages available YOLO models and their configurations
"""
import os
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class ModelInfo:
    """Information about a YOLO model"""
    id: str  # Unique identifier
    name: str  # Display name
    description: str  # Model description
    model_path: str  # Path to model file (.pt)
    classes: List[str]  # List of classes this model detects
    task: str = "detect"  # Task type: detect, segment, classify, etc.
    
    def __post_init__(self):
        """Validate model path exists (warning only, not blocking)"""
        if not os.path.exists(self.model_path):
            import warnings
            warnings.warn(f"Model file not found: {self.model_path}. Model will not be available until the file is added.")


# ============================================================================
# MODEL REGISTRY
# ============================================================================

# Steel defect detection model (Default)
STEEL_MODEL = ModelInfo(
    id="steel",
    name="Steel Defect Detection (Default)",
    description="Custom model for detecting steel surface defects: crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches",
    model_path="models/steel.pt",
    classes=[
        "crazing",
        "inclusion",
        "patches",
        "pitted_surface",
        "rolled-in_scale",
        "scratches"
    ],
    task="detect"
)

# Registry dictionary - maps model ID to ModelInfo
MODEL_REGISTRY: Dict[str, ModelInfo] = {
    STEEL_MODEL.id: STEEL_MODEL,
}


def get_model(model_id: str) -> Optional[ModelInfo]:
    """
    Get model information by ID
    
    Args:
        model_id: Model identifier
        
    Returns:
        ModelInfo object or None if not found
    """
    return MODEL_REGISTRY.get(model_id)


def get_all_models(only_available: bool = True) -> List[ModelInfo]:
    """
    Get all registered models
    
    Args:
        only_available: If True, only return models with existing files
    
    Returns:
        List of ModelInfo objects
    """
    if only_available:
        return [model for model in MODEL_REGISTRY.values() if os.path.exists(model.model_path)]
    return list(MODEL_REGISTRY.values())


def get_model_names() -> List[str]:
    """
    Get list of all model display names
    
    Returns:
        List of model names
    """
    return [model.name for model in MODEL_REGISTRY.values()]


def get_model_ids() -> List[str]:
    """
    Get list of all model IDs
    
    Returns:
        List of model IDs
    """
    return list(MODEL_REGISTRY.keys())


def register_model(model_info: ModelInfo) -> bool:
    """
    Register a new model in the registry
    
    Args:
        model_info: ModelInfo object to register
        
    Returns:
        True if successful, False if model ID already exists
    """
    if model_info.id in MODEL_REGISTRY:
        return False
    
    MODEL_REGISTRY[model_info.id] = model_info
    return True


def unregister_model(model_id: str) -> bool:
    """
    Remove a model from the registry
    
    Args:
        model_id: Model identifier to remove
        
    Returns:
        True if successful, False if model not found
    """
    if model_id not in MODEL_REGISTRY:
        return False
    
    # Don't allow removing default model
    if model_id == "steel":
        return False
    
    del MODEL_REGISTRY[model_id]
    return True


def get_default_model() -> ModelInfo:
    """
    Get the default model (Steel Defect Detection)
    
    Returns:
        Default ModelInfo object
    """
    return STEEL_MODEL

