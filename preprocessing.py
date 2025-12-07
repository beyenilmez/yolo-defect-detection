"""
Image Preprocessing Module
Modular preprocessing system with presets and JSON save/load support
"""
import cv2
import numpy as np
import json
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict


@dataclass
class PreprocessingConfig:
    """Configuration class for preprocessing settings"""
    # Basic adjustments
    brightness: int = 0  # -100 to 100
    contrast: int = 0    # -100 to 100
    gamma: float = 1.0  # 0.1 to 3.0
    
    # Resize settings
    resize_mode: str = "none"  # "none", "scale", "custom"
    resize_scale: float = 1.0  # 0.1 to 3.0
    resize_width: Optional[int] = None  # pixels
    resize_height: Optional[int] = None  # pixels
    
    # Blur settings
    blur_enabled: bool = False
    blur_type: str = "gaussian"  # "gaussian", "median", "bilateral"
    blur_kernel_size: int = 5  # must be odd: 3, 5, 7, 9, etc.
    
    # Sharpening
    sharpen_enabled: bool = False
    sharpen_strength: float = 1.0  # 0.5 to 2.0
    
    # Noise reduction
    denoise_enabled: bool = False
    denoise_strength: int = 10  # 1 to 20
    
    # Histogram equalization
    histogram_eq_enabled: bool = False
    histogram_eq_type: str = "clahe"  # "clahe", "global"
    
    # Geometric transformations
    rotation: float = 0.0  # degrees, -180 to 180
    flip_horizontal: bool = False
    flip_vertical: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for JSON serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PreprocessingConfig':
        """Create config from dictionary"""
        return cls(**data)
    
    def validate(self) -> bool:
        """Validate configuration values"""
        # Brightness and contrast range
        if not -100 <= self.brightness <= 100:
            return False
        if not -100 <= self.contrast <= 100:
            return False
        if not 0.1 <= self.gamma <= 3.0:
            return False
        
        # Resize validation
        if self.resize_mode not in ["none", "scale", "custom"]:
            return False
        if self.resize_mode == "scale" and not 0.1 <= self.resize_scale <= 3.0:
            return False
        if self.resize_mode == "custom":
            if self.resize_width is not None and (self.resize_width < 64 or self.resize_width > 3840):
                return False
            if self.resize_height is not None and (self.resize_height < 64 or self.resize_height > 2160):
                return False
        
        # Blur validation
        if self.blur_kernel_size < 3 or self.blur_kernel_size > 21 or self.blur_kernel_size % 2 == 0:
            return False
        
        # Rotation validation
        if not -180 <= self.rotation <= 180:
            return False
        
        return True


# ============================================================================
# PRESET DEFINITIONS
# ============================================================================

PRESETS = {
    "None": PreprocessingConfig(),
    
    "Low Light Enhancement": PreprocessingConfig(
        brightness=30,
        contrast=20,
        gamma=1.2,
        histogram_eq_enabled=True,
        histogram_eq_type="clahe"
    ),
    
    "High Contrast": PreprocessingConfig(
        contrast=50,
        sharpen_enabled=True,
        sharpen_strength=1.5
    ),
    
    "Noise Reduction": PreprocessingConfig(
        denoise_enabled=True,
        denoise_strength=10,
        blur_enabled=True,
        blur_type="bilateral",
        blur_kernel_size=5
    ),
    
    "Sharpening": PreprocessingConfig(
        sharpen_enabled=True,
        sharpen_strength=1.8,
        contrast=15
    ),
    
    "Blur Reduction": PreprocessingConfig(
        sharpen_enabled=True,
        sharpen_strength=2.0,
        contrast=25
    ),
    
    "Balanced Enhancement": PreprocessingConfig(
        brightness=10,
        contrast=15,
        gamma=1.1,
        sharpen_enabled=True,
        sharpen_strength=1.2
    )
}


def get_preset(name: str) -> Optional[PreprocessingConfig]:
    """Get a preset configuration by name"""
    return PRESETS.get(name)


def get_preset_names() -> list:
    """Get list of available preset names"""
    return list(PRESETS.keys())


# ============================================================================
# JSON SAVE/LOAD FUNCTIONS
# ============================================================================

def save_config_to_json(config: PreprocessingConfig, filepath: str) -> bool:
    """
    Save preprocessing configuration to JSON file
    
    Args:
        config: PreprocessingConfig object
        filepath: Path to save JSON file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        data = config.to_dict()
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving preprocessing config: {e}")
        return False


def load_config_from_json(filepath: str) -> Optional[PreprocessingConfig]:
    """
    Load preprocessing configuration from JSON file
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        PreprocessingConfig object or None if failed
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        config = PreprocessingConfig.from_dict(data)
        if config.validate():
            return config
        else:
            print("Warning: Loaded preprocessing config failed validation")
            return None
    except Exception as e:
        print(f"Error loading preprocessing config: {e}")
        return None


# ============================================================================
# INDIVIDUAL PREPROCESSING FUNCTIONS
# ============================================================================

def adjust_brightness_contrast(frame: np.ndarray, brightness: int, contrast: int) -> np.ndarray:
    """
    Adjust brightness and contrast using reliable OpenCV method
    
    Args:
        frame: Input frame (BGR format)
        brightness: Brightness adjustment (-100 to 100)
        contrast: Contrast adjustment (-100 to 100)
        
    Returns:
        Adjusted frame
    """
    if brightness == 0 and contrast == 0:
        return frame
    
    # Convert to float32 for precision
    frame_float = frame.astype(np.float32)
    
    # Calculate alpha (contrast) and beta (brightness)
    # Contrast: -100 to 100 -> alpha: 0.5 to 2.0
    alpha = 1.0 + (contrast / 100.0)
    alpha = max(0.5, min(2.0, alpha))  # Clamp to safe range
    
    # Brightness: -100 to 100 -> beta: -127 to 127
    beta = (brightness / 100.0) * 127
    
    # Apply transformation: output = (input - 128) * alpha + 128 + beta
    # This centers around 128 (middle gray) for natural contrast adjustment
    if contrast != 0:
        adjusted = (frame_float - 128.0) * alpha + 128.0 + beta
    else:
        adjusted = frame_float + beta
    
    # Clip to valid range and convert back
    adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
    return adjusted


def adjust_gamma(frame: np.ndarray, gamma: float) -> np.ndarray:
    """
    Adjust gamma correction
    
    Args:
        frame: Input frame (BGR format)
        gamma: Gamma value (0.1 to 3.0, 1.0 = no change)
        
    Returns:
        Gamma-corrected frame
    """
    if gamma == 1.0:
        return frame
    
    # Build lookup table for gamma correction
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
    
    # Apply lookup table
    return cv2.LUT(frame, table)


def apply_blur(frame: np.ndarray, blur_type: str, kernel_size: int) -> np.ndarray:
    """
    Apply blur filter
    
    Args:
        frame: Input frame (BGR format)
        blur_type: Type of blur ("gaussian", "median", "bilateral")
        kernel_size: Kernel size (must be odd)
        
    Returns:
        Blurred frame
    """
    if kernel_size < 3 or kernel_size % 2 == 0:
        kernel_size = 3
    
    if blur_type == "gaussian":
        return cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
    elif blur_type == "median":
        return cv2.medianBlur(frame, kernel_size)
    elif blur_type == "bilateral":
        # Bilateral filter preserves edges
        return cv2.bilateralFilter(frame, kernel_size, 80, 80)
    else:
        return frame


def apply_sharpening(frame: np.ndarray, strength: float) -> np.ndarray:
    """
    Apply unsharp masking for sharpening
    
    Args:
        frame: Input frame (BGR format)
        strength: Sharpening strength (0.5 to 2.0)
        
    Returns:
        Sharpened frame
    """
    if strength == 1.0:
        return frame
    
    # Create unsharp mask kernel
    # Standard unsharp mask: -1/9 * [[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]], dtype=np.float32)
    
    # Adjust strength
    kernel[1, 1] = 8 + strength
    kernel = kernel / (8 + strength - 1)
    
    # Apply filter
    sharpened = cv2.filter2D(frame, -1, kernel)
    
    # Blend with original based on strength
    if strength > 1.0:
        # More aggressive sharpening
        alpha = min(1.0, (strength - 1.0))
        return cv2.addWeighted(sharpened, alpha, frame, 1 - alpha, 0)
    else:
        # Subtle sharpening
        alpha = strength
        return cv2.addWeighted(sharpened, alpha, frame, 1 - alpha, 0)


def apply_denoise(frame: np.ndarray, strength: int) -> np.ndarray:
    """
    Apply non-local means denoising
    
    Args:
        frame: Input frame (BGR format)
        strength: Denoising strength (1 to 20)
        
    Returns:
        Denoised frame
    """
    # Clamp strength to valid range
    h = max(1, min(20, strength))
    
    # Apply non-local means denoising
    # h: filter strength, templateWindowSize: template patch size, searchWindowSize: search window size
    return cv2.fastNlMeansDenoisingColored(frame, None, h, h, 7, 21)


def apply_histogram_equalization(frame: np.ndarray, eq_type: str) -> np.ndarray:
    """
    Apply histogram equalization
    
    Args:
        frame: Input frame (BGR format)
        eq_type: Type of equalization ("clahe", "global")
        
    Returns:
        Equalized frame
    """
    if eq_type == "clahe":
        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel only
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_eq = clahe.apply(l)
        
        # Merge channels and convert back
        lab_eq = cv2.merge([l_eq, a, b])
        return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    
    elif eq_type == "global":
        # Convert to YUV color space
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        y, u, v = cv2.split(yuv)
        
        # Apply equalization to Y channel only
        y_eq = cv2.equalizeHist(y)
        
        # Merge channels and convert back
        yuv_eq = cv2.merge([y_eq, u, v])
        return cv2.cvtColor(yuv_eq, cv2.COLOR_YUV2BGR)
    
    else:
        return frame


def apply_resize(frame: np.ndarray, mode: str, scale: float = 1.0, 
                 width: Optional[int] = None, height: Optional[int] = None) -> np.ndarray:
    """
    Resize frame
    
    Args:
        frame: Input frame (BGR format)
        mode: Resize mode ("none", "scale", "custom")
        scale: Scale factor (for "scale" mode)
        width: Target width (for "custom" mode)
        height: Target height (for "custom" mode)
        
    Returns:
        Resized frame
    """
    if mode == "none":
        return frame
    
    h, w = frame.shape[:2]
    
    if mode == "scale":
        if scale == 1.0:
            return frame
        new_width = int(w * scale)
        new_height = int(h * scale)
        return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    elif mode == "custom":
        if width is not None and height is not None:
            return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
        elif width is not None:
            aspect_ratio = h / w
            new_height = int(width * aspect_ratio)
            return cv2.resize(frame, (width, new_height), interpolation=cv2.INTER_LINEAR)
        elif height is not None:
            aspect_ratio = w / h
            new_width = int(height * aspect_ratio)
            return cv2.resize(frame, (new_width, height), interpolation=cv2.INTER_LINEAR)
    
    return frame


def apply_rotation(frame: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate frame
    
    Args:
        frame: Input frame (BGR format)
        angle: Rotation angle in degrees (-180 to 180)
        
    Returns:
        Rotated frame
    """
    if angle == 0:
        return frame
    
    h, w = frame.shape[:2]
    center = (w // 2, h // 2)
    
    # Get rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Calculate new dimensions to avoid cropping
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    # Adjust rotation matrix for new dimensions
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    
    # Apply rotation
    return cv2.warpAffine(frame, M, (new_w, new_h), flags=cv2.INTER_LINEAR, 
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


def apply_flip(frame: np.ndarray, flip_horizontal: bool, flip_vertical: bool) -> np.ndarray:
    """
    Flip frame
    
    Args:
        frame: Input frame (BGR format)
        flip_horizontal: Flip horizontally
        flip_vertical: Flip vertically
        
    Returns:
        Flipped frame
    """
    if not flip_horizontal and not flip_vertical:
        return frame
    
    flip_code = -1 if (flip_horizontal and flip_vertical) else (1 if flip_horizontal else 0)
    return cv2.flip(frame, flip_code)


# ============================================================================
# MAIN PREPROCESSING FUNCTION
# ============================================================================

def apply_preprocessing(frame: np.ndarray, config: PreprocessingConfig) -> np.ndarray:
    """
    Apply all preprocessing steps according to configuration
    
    Args:
        frame: Input frame (BGR format)
        config: PreprocessingConfig object
        
    Returns:
        Preprocessed frame
    """
    if not config.validate():
        print("Warning: Invalid preprocessing config, returning original frame")
        return frame
    
    processed = frame.copy()
    
    # 1. Brightness and Contrast (applied together for efficiency)
    if config.brightness != 0 or config.contrast != 0:
        processed = adjust_brightness_contrast(processed, config.brightness, config.contrast)
    
    # 2. Gamma correction
    if config.gamma != 1.0:
        processed = adjust_gamma(processed, config.gamma)
    
    # 3. Histogram equalization (before other filters)
    if config.histogram_eq_enabled:
        processed = apply_histogram_equalization(processed, config.histogram_eq_type)
    
    # 4. Denoising
    if config.denoise_enabled:
        processed = apply_denoise(processed, config.denoise_strength)
    
    # 5. Blur
    if config.blur_enabled:
        processed = apply_blur(processed, config.blur_type, config.blur_kernel_size)
    
    # 6. Sharpening
    if config.sharpen_enabled:
        processed = apply_sharpening(processed, config.sharpen_strength)
    
    # 7. Geometric transformations
    if config.rotation != 0:
        processed = apply_rotation(processed, config.rotation)
    
    if config.flip_horizontal or config.flip_vertical:
        processed = apply_flip(processed, config.flip_horizontal, config.flip_vertical)
    
    # 8. Resize (applied last to minimize processing on smaller images)
    processed = apply_resize(processed, config.resize_mode, config.resize_scale, 
                            config.resize_width, config.resize_height)
    
    return processed
