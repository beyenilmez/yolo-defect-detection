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
    name="Steel Defect Detection",
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

# PCB defect detection model
PCB_MODEL = ModelInfo(
    id="pcb",
    name="PCB Defect Detection",
    description="Custom model for detecting PCB defects: copper, mousebite, open, pin-hole, short, spur",
    model_path="models/pcb.pt",
    classes=[
        "copper",
        "mousebite",
        "open",
        "pin-hole",
        "short",
        "spur"
    ],
    task="detect"
)

# Fabric defect detection model
FABRIC_MODEL = ModelInfo(
    id="fabric",
    name="Fabric Defect Detection",
    description="Custom model for detecting fabric defects: hole, objects, oil spot, thread error",
    model_path="models/fabric.pt",
    classes=[
        "hole",
        "objects",
        "oil spot",
        "thread error"
    ],
    task="detect"
)

# Leather defect detection model
LEATHER_MODEL = ModelInfo(
    id="leather",
    name="Leather Defect Detection",
    description="Custom model for detecting leather defects: Bacterial Injury, Crease, Growth Marks, Healed Injury, Hole, Rotten surface, Scratch, pinhole",
    model_path="models/leather.pt",
    classes=[
        "Bacterial Injury",
        "Crease",
        "Growth Marks",
        "Healed Injury",
        "Hole",
        "Rotten surface",
        "Scratch",
        "pinhole"
    ],
    task="detect"
)

# Wood defect detection model
WOOD_MODEL = ModelInfo(
    id="wood",
    name="Wood Defect Detection",
    description="Custom model for detecting wood defects: Blue_Stain, Crack, Dead_Knot, Knot_missing, Live_Knot, Marrow, Quartzity, knot_with_crack, overgrown, resin",
    model_path="models/wood.pt",
    classes=[
        "Blue_Stain",
        "Crack",
        "Dead_Knot",
        "Knot_missing",
        "Live_Knot",
        "Marrow",
        "Quartzity",
        "knot_with_crack",
        "overgrown",
        "resin"
    ],
    task="detect"
)

# Registry dictionary - maps model ID to ModelInfo
MODEL_REGISTRY: Dict[str, ModelInfo] = {
    STEEL_MODEL.id: STEEL_MODEL,
    PCB_MODEL.id: PCB_MODEL,
    FABRIC_MODEL.id: FABRIC_MODEL,
    LEATHER_MODEL.id: LEATHER_MODEL,
    WOOD_MODEL.id: WOOD_MODEL,
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

