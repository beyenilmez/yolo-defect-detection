"""
Defect Detection System Module
Handles YOLO model loading and frame processing
"""
import time
import streamlit as st
from ultralytics import YOLO
import torch
import numpy as np


@st.cache_resource
def _load_cached_model(device, num_threads, model_path):
    """
    Load YOLO model with caching based on device, thread settings, and model path
    
    Args:
        device: Device string ('cuda:0' or 'cpu')
        num_threads: Number of CPU threads
        model_path: Path to model file (.pt)
    
    Returns:
        Loaded YOLO model
    """
    # Set CPU threads if specified (useful even with GPU for data loading, preprocessing, etc.)
    if num_threads is not None:
        torch.set_num_threads(num_threads)
    
    # Load model with explicit device specification
    # YOLO automatically handles device placement, but we can specify it
    model = YOLO(model_path)
    
    # Ensure model is on the correct device
    if device.startswith('cuda'):
        model.to(device)
        # Verify GPU is actually being used
        try:
            next(model.model.parameters()).device
        except:
            pass
    
    return model


class DefectDetectionSystem:
    """Handles defect detection using YOLO model"""
    
    def __init__(self, model_path='models/steel.pt', use_gpu=True, num_threads=None):
        self.model_path = model_path
        self.use_gpu = use_gpu
        self.num_threads = num_threads
        self.device = self._determine_device()
        self.model = _load_cached_model(self.device, self.num_threads, self.model_path)
        self._verify_device()
    
    def _determine_device(self):
        """Determine the device to use for inference"""
        if self.use_gpu and torch.cuda.is_available():
            return 'cuda:0'
        return 'cpu'
    
    def _verify_device(self):
        """Verify that the model is on the correct device"""
        if self.device.startswith('cuda'):
            try:
                # Check if model parameters are on GPU
                param_device = next(self.model.model.parameters()).device
                if param_device.type != 'cuda':
                    import warnings
                    warnings.warn(f"Model was requested on {self.device} but is on {param_device}")
            except:
                pass

    def _calculate_overlay_scale(self, frame):
        """
        Calculate font size and line width based on frame resolution
        
        Uses the minimum dimension (width or height) for more consistent scaling
        across different aspect ratios.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            tuple: (font_size, line_width)
        """
        height, width = frame.shape[:2]
        min_dimension = min(width, height)
        
        # Base dimension for scaling (640px is a common base)
        base_dimension = 640.0
        
        # Scale factor based on minimum dimension
        # This ensures consistent scaling regardless of aspect ratio
        scale_factor = min_dimension / base_dimension
        
        # Base font size and line width for base_dimension resolution
        # These values work well for 640px images
        base_font_size = 1.0
        base_line_width = 2.0
        
        # Calculate scaled values
        # For very small images (< 320px), use minimum values
        # For very large images (> 1920px), use maximum values
        font_size = base_font_size * scale_factor
        line_width = base_line_width * scale_factor
        
        # Clamp to reasonable ranges to prevent too small or too large overlays
        font_size = max(0.3, min(2.5, font_size))
        line_width = max(0.5, min(5.0, line_width))
        
        # Convert line_width to integer (OpenCV requires integer for thickness)
        # Round to nearest integer, but ensure minimum of 1
        line_width = max(1, int(round(line_width)))
        
        return font_size, line_width
    
    def process_frame(self, frame, conf_threshold, use_tracking=False):
        """
        Process a single frame for defect detection
        
        Args:
            frame: Input frame (BGR format)
            conf_threshold: Confidence threshold for detection
            use_tracking: Whether to use object tracking (for video streams)
            
        Returns:
            tuple: (annotated_frame, detection_count, inference_time_ms, defect_details)
            defect_details: dict with class names as keys and lists of (confidence, track_id) tuples as values
        """
        start_time = time.time()
        
        # Use tracking for video streams, regular detection for single images
        if use_tracking:
            # Use track() method with persist=True to maintain tracking across frames
            results = self.model.track(
                frame, 
                conf=conf_threshold, 
                device=self.device, 
                verbose=False,
                persist=True
            )
        else:
            # Regular detection for single images
            results = self.model(frame, conf=conf_threshold, device=self.device, verbose=False)
        
        # Calculate overlay scale based on frame resolution
        font_size, line_width = self._calculate_overlay_scale(frame)
        
        # Plot with resolution-adaptive overlay sizes
        annotated_frame = results[0].plot(
            line_width=line_width,
            font_size=font_size
        )
        
        # Count detected objects
        det_count = len(results[0].boxes)
        
        # Extract detailed defect information with tracking IDs
        defect_details = {}
        if len(results[0].boxes) > 0:
            # Get class names from model
            class_names = results[0].names
            
            # Extract class IDs and confidence scores
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()
            
            # Extract tracking IDs if available
            track_ids = None
            if use_tracking:
                # Check if tracking IDs are available
                if hasattr(results[0].boxes, 'id') and results[0].boxes.id is not None:
                    track_ids = results[0].boxes.id.cpu().numpy()
                    # Handle both tensor and numpy array cases
                    if hasattr(track_ids, 'astype'):
                        track_ids = track_ids.astype(int)
                else:
                    # Tracking IDs not available - this can happen if tracking hasn't initialized yet
                    track_ids = None
            
            # Extract bounding box centers for fallback matching when tracking fails
            box_centers = None
            if use_tracking and track_ids is None:
                # Use box centers as fallback identifier
                boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()
                box_centers = [(int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)) for box in boxes_xyxy]
            
            # Group by class name, store (confidence, track_id, box_center) tuples
            for idx, (class_id, conf) in enumerate(zip(class_ids, confidences)):
                class_name = class_names[class_id]
                if class_name not in defect_details:
                    defect_details[class_name] = []
                
                # Store confidence, track_id, and box_center for better matching
                track_id = int(track_ids[idx]) if track_ids is not None else None
                box_center = box_centers[idx] if box_centers is not None else None
                defect_details[class_name].append((float(conf), track_id, box_center))
        
        end_time = time.time()
        inference_time = (end_time - start_time) * 1000  # Convert to milliseconds

        return annotated_frame, det_count, inference_time, defect_details

