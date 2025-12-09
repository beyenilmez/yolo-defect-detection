"""
UI Components Module
Handles Streamlit UI components and dashboard updates
"""
import streamlit as st
import os
import torch
from .preprocessing import (
    PreprocessingConfig, 
    get_preset, 
    get_preset_names
)
from .model_registry import (
    get_all_models,
    get_model,
    get_default_model
)

# Widget keys for preprocessing controls - used to clear widget state when preset/config changes
PREPROCESSING_WIDGET_KEYS = [
    'preprocessing_brightness', 'preprocessing_contrast', 'preprocessing_gamma',
    'preprocessing_resize_mode', 'preprocessing_resize_scale', 
    'preprocessing_resize_width', 'preprocessing_resize_height',
    'preprocessing_blur_enabled', 'preprocessing_blur_type', 'preprocessing_blur_kernel_size',
    'preprocessing_sharpen_enabled', 'preprocessing_sharpen_strength',
    'preprocessing_denoise_enabled', 'preprocessing_denoise_strength',
    'preprocessing_histogram_eq_enabled', 'preprocessing_histogram_eq_type',
    'preprocessing_rotation', 'preprocessing_flip_horizontal', 'preprocessing_flip_vertical'
]


def render_sidebar_controls():
    """
    Render sidebar controls and return user selections
    
    Returns:
        tuple: (source_option, camera_index, confidence_threshold, target_fps, 
                use_gpu, num_threads, uploaded_file, start_camera_clicked)
    """
    st.sidebar.title("⚙️ Control Panel")
    
    # ============================================
    # 1. INPUT SOURCE
    # ============================================
    st.sidebar.header("📹 Input Source")
    source_option = st.sidebar.radio(
        "Select Source:",
        ["None", "Upload Image", "Upload Video", "Live Camera", "IP Camera", "Folder"],
        help="Choose input source: upload image/video, use live camera, IP camera, or process folder"
    )
    
    # Get system capabilities
    max_threads = os.cpu_count() or 4
    gpu_available = torch.cuda.is_available()
    
    # Show GPU info if available
    gpu_info = ""
    if gpu_available:
        try:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_info = f" ({gpu_name})"
        except:
            pass
    
    # Camera Specific Settings and Upload/Start controls
    cam_index = 0
    uploaded_file = None
    uploaded_image = None
    start_camera_clicked = False
    ip_camera_url = None
    start_ip_camera_clicked = False
    folder_path = None
    folder_mode = None
    start_folder_clicked = False
    
    if source_option == "Live Camera":
        cam_index = st.sidebar.selectbox(
            "📷 Camera Index", 
            options=[0, 1, 2, 3], 
            index=0,
            help="Select camera device (0 = default webcam, 1+ = external cameras)"
        )
        start_camera_clicked = st.sidebar.button("▶️ Start Camera", type="primary", use_container_width=True)
    elif source_option == "IP Camera":
        ip_camera_url = st.sidebar.text_input(
            "🌐 IP Camera URL",
            value=st.session_state.get('ip_camera_url', ''),
            help="Enter IP camera URL (e.g., rtsp://ip:port/stream, http://ip:port/video)",
            key="ip_camera_url_input"
        )
        st.sidebar.caption("💡 Examples: rtsp://192.168.1.100:554/stream, http://192.168.1.100:8080/video")
        start_ip_camera_clicked = st.sidebar.button("▶️ Start IP Camera", type="primary", use_container_width=True)
    elif source_option == "Upload Video":
        uploaded_file = st.sidebar.file_uploader(
            "📁 Upload Video File", 
            type=['mp4', 'avi', 'mov'],
            help="Upload video file (MP4, AVI, or MOV)"
        )
    elif source_option == "Upload Image":
        uploaded_image = st.sidebar.file_uploader(
            "🖼️ Upload Image File", 
            type=['jpg', 'jpeg', 'png', 'bmp', 'webp'],
            help="Upload image file (JPG, PNG, BMP, or WEBP)"
        )
    elif source_option == "Folder":
        folder_path = st.sidebar.text_input(
            "📂 Folder Path",
            value=st.session_state.get('folder_path', ''),
            help="Enter the path to the folder containing images",
            key="folder_path_input"
        )
        folder_mode = st.sidebar.radio(
            "Processing Mode:",
            ["Process All", "Watch Folder"],
            help="Process All: process all images sequentially\nWatch Folder: monitor folder for new images",
            key="folder_mode_radio"
        )
        start_folder_clicked = st.sidebar.button("▶️ Start Processing", type="primary", use_container_width=True)
    
    # ============================================
    # 2. DEFECT IMAGE SAVING SETTINGS
    # ============================================
    st.sidebar.markdown("---")
    st.sidebar.header("💾 Save Defect Images")
    
    # Enable/Disable saving
    # Widget with key automatically manages session_state, so we just use the returned value
    enable_save_defects = st.sidebar.checkbox(
        "Save Images with Defects",
        value=st.session_state.get('enable_save_defects', False),
        help="Save images with detected defects (including overlays) to a folder",
        key="enable_save_defects"
    )
    
    # Save directory path
    # Widget with key automatically manages session_state, so we just use the returned value
    default_save_dir = st.session_state.get('defect_save_dir', './defect_images')
    defect_save_dir = st.sidebar.text_input(
        "Save Directory",
        value=default_save_dir,
        help="Directory path where defect images will be saved (default: ./defect_images)",
        key="defect_save_dir",
        disabled=not enable_save_defects
    )
    
    # Note: Widgets with keys automatically update session_state, so no manual update needed
    # The values are already stored in session_state via the widget keys
    
    # ============================================
    # 3. YOLO MODEL SETTINGS
    # ============================================
    st.sidebar.markdown("---")
    st.sidebar.header("🧠 YOLO Model")
    
    # Model Selection
    available_models = get_all_models()
    model_names = [model.name for model in available_models]
    
    # Get default model or last selected model from session state
    default_model_id = st.session_state.get('selected_model_id', get_default_model().id)
    default_model = get_model(default_model_id) or get_default_model()
    
    # Find index of default model
    try:
        default_index = next(i for i, model in enumerate(available_models) if model.id == default_model.id)
    except StopIteration:
        default_index = 0
    
    selected_model_name = st.sidebar.selectbox(
        "Model",
        model_names,
        index=default_index,
        help="Select the YOLO model to use for detection",
        key="model_selection"
    )
    
    # Get selected model info
    selected_model = next(model for model in available_models if model.name == selected_model_name)
    st.session_state['selected_model_id'] = selected_model.id
    st.session_state['selected_model_path'] = selected_model.model_path
    
    # Show model description
    st.sidebar.caption(f"📝 {selected_model.description}")
    
    # Show detected classes if available
    if selected_model.classes:
        classes_text = ", ".join(selected_model.classes[:5])
        if len(selected_model.classes) > 5:
            classes_text += f" (+{len(selected_model.classes) - 5} more)"
        st.sidebar.caption(f"🔍 Classes: {classes_text}")
    
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold", 
        0.1, 1.0, 0.45, 0.05,
        help="Minimum detection confidence (lower = more detections, higher = only high confidence)"
    )
    
    target_fps = st.sidebar.slider(
        "FPS Limit", 
        1, 60, 30, 1,
        help="Maximum processing speed (frames per second)"
    )
    
    # ============================================
    # 4. PERFORMANCE SETTINGS
    # ============================================
    st.sidebar.markdown("---")
    st.sidebar.header("⚡ Performance")
    
    # GPU setting
    gpu_label = f"🚀 Use GPU{gpu_info}" if gpu_available else "🚀 Use GPU (Not Available)"
    use_gpu = st.sidebar.checkbox(
        gpu_label, 
        value=gpu_available,
        disabled=not gpu_available,
        help=f"Enable GPU acceleration for faster inference{gpu_info}" if gpu_available else "GPU not available. Install PyTorch with CUDA support"
    )
    
    # If GPU is selected but not available, force to False
    if use_gpu and not gpu_available:
        use_gpu = False
    
    # Show GPU status info (compact)
    if gpu_available:
        try:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            st.sidebar.caption(f"✅ {gpu_name} • {gpu_memory:.1f} GB VRAM")
        except:
            st.sidebar.caption(f"✅ GPU Detected{gpu_info}")
    else:
        # Check why GPU is not available
        cuda_installed = torch.version.cuda is not None
        if not cuda_installed:
            st.sidebar.caption("⚠️ PyTorch without CUDA support")
        else:
            st.sidebar.caption("⚠️ No GPU detected")
    
    # CPU threads
    default_threads = max(1, max_threads // 2)
    num_threads = st.sidebar.slider(
        "CPU Threads", 
        1, max_threads, default_threads, 1,
        help=f"Number of CPU threads (max: {max_threads}) for data loading and preprocessing"
    )
    
    # ============================================
    # 5. IMAGE PREPROCESSING
    # ============================================
    st.sidebar.markdown("---")
    st.sidebar.header("🎨 Image Preprocessing")
    
    # Enable/Disable checkbox
    enable_preprocessing = st.sidebar.checkbox(
        "Enable Preprocessing",
        value=False,
        help="Enable or disable image preprocessing"
    )
    
    # Initialize or get config from session state
    if 'preprocessing_config' not in st.session_state:
        st.session_state['preprocessing_config'] = PreprocessingConfig()
    
    config = st.session_state['preprocessing_config']
    
    if enable_preprocessing:
        # Preset Selection
        preset_names = get_preset_names()
        # Add "Custom" option
        preset_options = preset_names + ["Custom"]
        
        # Determine current preset or "Custom" from session state
        current_preset = st.session_state.get('last_preset', "None")
        if current_preset not in preset_options:
            current_preset = "Custom"
        
        # Get current preset index
        try:
            current_index = preset_options.index(current_preset)
        except ValueError:
            current_index = len(preset_options) - 1  # Default to "Custom"
        
        selected_preset = st.sidebar.selectbox(
            "📋 Preset",
            preset_options,
            index=current_index,
            help="Select a preprocessing preset or 'Custom' to configure manually",
            key="preset_selectbox"
        )
        
        # Apply preset immediately when selected (and not "Custom")
        # Check if preset changed from last known state
        last_known_preset = st.session_state.get('last_preset', None)
        
        if selected_preset != "Custom" and selected_preset in preset_names:
            preset_config = get_preset(selected_preset)
            if preset_config:
                # Apply preset if it's different from last known preset
                if last_known_preset != selected_preset:
                    # Step 1: Create a completely fresh default config (resets ALL settings)
                    default_config = PreprocessingConfig()
                    default_dict = default_config.to_dict()
                    
                    # Step 2: Get preset's settings
                    preset_dict = preset_config.to_dict()
                    
                    # Step 3: Merge preset settings onto defaults
                    # This ensures ALL settings are first reset to defaults, then preset values applied
                    merged_dict = {**default_dict, **preset_dict}
                    
                    # Step 4: Create new config from merged dict
                    new_config = PreprocessingConfig.from_dict(merged_dict)
                    
                    # Step 5: Clear widget states to force them to use new config values
                    for key in PREPROCESSING_WIDGET_KEYS:
                        if key in st.session_state:
                            del st.session_state[key]
                    
                    # Step 6: Save to session state
                    st.session_state['preprocessing_config'] = new_config
                    st.session_state['last_preset'] = selected_preset
                    
                    # Step 7: CRITICAL - Reload config from session_state to ensure UI uses new values
                    config = st.session_state['preprocessing_config']
                    
                    # Step 8: Force rerun to update all UI widgets with new preset values
                    st.rerun()
        elif selected_preset == "Custom":
            # If switching to Custom, update last_preset but keep current config
            if last_known_preset != "Custom":
                st.session_state['last_preset'] = "Custom"
        
        # CRITICAL: Sync widget values from config to session_state
        # This ensures that when a widget changes, other widgets don't revert to old values
        # We need to update session_state for all widgets based on current config
        # but only if the widget key doesn't exist in session_state (to preserve user changes)
        # However, we need to be careful: if a widget was just changed, its session_state value
        # should take precedence. But if config was updated (e.g., from preset), we need to
        # sync all widgets to config values.
        
        # The solution: Always sync widget session_state from config BEFORE creating widgets
        # This way, widgets will use config values, but user changes will update both config and session_state
        resize_mode_map = {"none": "None", "scale": "Scale", "custom": "Custom Size"}
        
        # Sync all widget values from config to session_state
        # This ensures widgets always reflect the current config state
        if 'preprocessing_brightness' not in st.session_state:
            st.session_state['preprocessing_brightness'] = config.brightness
        if 'preprocessing_contrast' not in st.session_state:
            st.session_state['preprocessing_contrast'] = config.contrast
        if 'preprocessing_gamma' not in st.session_state:
            st.session_state['preprocessing_gamma'] = config.gamma
        if 'preprocessing_resize_mode' not in st.session_state:
            st.session_state['preprocessing_resize_mode'] = resize_mode_map.get(config.resize_mode, "None")
        if 'preprocessing_resize_scale' not in st.session_state:
            st.session_state['preprocessing_resize_scale'] = config.resize_scale
        if 'preprocessing_resize_width' not in st.session_state:
            st.session_state['preprocessing_resize_width'] = config.resize_width or 640
        if 'preprocessing_resize_height' not in st.session_state:
            st.session_state['preprocessing_resize_height'] = config.resize_height or 480
        if 'preprocessing_blur_enabled' not in st.session_state:
            st.session_state['preprocessing_blur_enabled'] = config.blur_enabled
        if 'preprocessing_blur_type' not in st.session_state:
            st.session_state['preprocessing_blur_type'] = config.blur_type
        if 'preprocessing_blur_kernel_size' not in st.session_state:
            st.session_state['preprocessing_blur_kernel_size'] = config.blur_kernel_size
        if 'preprocessing_sharpen_enabled' not in st.session_state:
            st.session_state['preprocessing_sharpen_enabled'] = config.sharpen_enabled
        if 'preprocessing_sharpen_strength' not in st.session_state:
            st.session_state['preprocessing_sharpen_strength'] = config.sharpen_strength
        if 'preprocessing_denoise_enabled' not in st.session_state:
            st.session_state['preprocessing_denoise_enabled'] = config.denoise_enabled
        if 'preprocessing_denoise_strength' not in st.session_state:
            st.session_state['preprocessing_denoise_strength'] = config.denoise_strength
        if 'preprocessing_histogram_eq_enabled' not in st.session_state:
            st.session_state['preprocessing_histogram_eq_enabled'] = config.histogram_eq_enabled
        if 'preprocessing_histogram_eq_type' not in st.session_state:
            st.session_state['preprocessing_histogram_eq_type'] = config.histogram_eq_type
        if 'preprocessing_rotation' not in st.session_state:
            st.session_state['preprocessing_rotation'] = int(config.rotation)
        if 'preprocessing_flip_horizontal' not in st.session_state:
            st.session_state['preprocessing_flip_horizontal'] = config.flip_horizontal
        if 'preprocessing_flip_vertical' not in st.session_state:
            st.session_state['preprocessing_flip_vertical'] = config.flip_vertical
        
        # Use a subtle separator for sub-sections (different from main sections)
        st.sidebar.markdown("<hr style='margin: 0.5rem 0; border: none; border-top: 1px solid #e0e0e0;'>", unsafe_allow_html=True)
        st.sidebar.subheader("Basic Adjustments")
        
        # Brightness - read from session_state (which is synced from config)
        config.brightness = st.sidebar.slider(
            "Brightness", 
            -100, 100, st.session_state['preprocessing_brightness'], 1,
            help="Adjust image brightness (negative = darker, 0 = normal, positive = brighter)",
            key="preprocessing_brightness"
        )
        
        # Contrast - read from session_state (which is synced from config)
        config.contrast = st.sidebar.slider(
            "Contrast", 
            -100, 100, st.session_state['preprocessing_contrast'], 1,
            help="Adjust image contrast (negative = less contrast, 0 = normal, positive = more contrast)",
            key="preprocessing_contrast"
        )
        
        # Gamma - read from session_state (which is synced from config)
        config.gamma = st.sidebar.slider(
            "Gamma",
            0.1, 3.0, st.session_state['preprocessing_gamma'], 0.1,
            help="Gamma correction (1.0 = normal, lower = brighter, higher = darker)",
            key="preprocessing_gamma"
        )
        
        # Subtle separator for sub-sections
        st.sidebar.markdown("<hr style='margin: 0.5rem 0; border: none; border-top: 1px solid #e0e0e0;'>", unsafe_allow_html=True)
        st.sidebar.subheader("Resize")
        
        # Resize mode - read from session_state (which is synced from config)
        resize_mode_map = {"none": "None", "scale": "Scale", "custom": "Custom Size"}
        resize_mode_reverse = {v: k for k, v in resize_mode_map.items()}
        current_resize_mode = st.session_state['preprocessing_resize_mode']
        
        selected_resize_mode = st.sidebar.radio(
            "Resize Mode",
            ["None", "Scale", "Custom Size"],
            index=["None", "Scale", "Custom Size"].index(current_resize_mode),
            help="Resize frames before processing",
            key="preprocessing_resize_mode"
        )
        config.resize_mode = resize_mode_reverse[selected_resize_mode]
        
        if config.resize_mode == "scale":
            config.resize_scale = st.sidebar.slider(
                "Scale Factor", 
                0.1, 3.0, st.session_state['preprocessing_resize_scale'], 0.1,
                help="Resize factor (0.5 = 50% size, 1.0 = original, 2.0 = 200% size)",
                key="preprocessing_resize_scale"
            )
        elif config.resize_mode == "custom":
            col_width, col_height = st.sidebar.columns(2)
            with col_width:
                config.resize_width = st.number_input(
                    "Width (px)", 
                    min_value=64, 
                    max_value=3840, 
                    value=st.session_state['preprocessing_resize_width'], 
                    step=64,
                    help="Target width in pixels",
                    key="preprocessing_resize_width"
                )
            with col_height:
                config.resize_height = st.number_input(
                    "Height (px)", 
                    min_value=64, 
                    max_value=2160, 
                    value=st.session_state['preprocessing_resize_height'], 
                    step=64,
                    help="Target height in pixels",
                    key="preprocessing_resize_height"
                )
        
        # Subtle separator for sub-sections
        st.sidebar.markdown("<hr style='margin: 0.5rem 0; border: none; border-top: 1px solid #e0e0e0;'>", unsafe_allow_html=True)
        st.sidebar.subheader("Filters")
        
        # Blur - read from session_state (which is synced from config)
        config.blur_enabled = st.sidebar.checkbox(
            "Enable Blur",
            value=st.session_state['preprocessing_blur_enabled'],
            help="Apply blur filter",
            key="preprocessing_blur_enabled"
        )
        if config.blur_enabled:
            config.blur_type = st.sidebar.selectbox(
                "Blur Type",
                ["gaussian", "median", "bilateral"],
                index=["gaussian", "median", "bilateral"].index(st.session_state['preprocessing_blur_type']),
                help="Type of blur filter",
                key="preprocessing_blur_type"
            )
            # Ensure kernel size is odd
            kernel_size = st.sidebar.slider(
                "Kernel Size",
                3, 21, st.session_state['preprocessing_blur_kernel_size'], 2,
                help="Blur kernel size (must be odd number)",
                key="preprocessing_blur_kernel_size"
            )
            config.blur_kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        
        # Sharpening - read from session_state (which is synced from config)
        config.sharpen_enabled = st.sidebar.checkbox(
            "Enable Sharpening",
            value=st.session_state['preprocessing_sharpen_enabled'],
            help="Apply sharpening filter",
            key="preprocessing_sharpen_enabled"
        )
        if config.sharpen_enabled:
            config.sharpen_strength = st.sidebar.slider(
                "Sharpening Strength",
                0.5, 2.0, st.session_state['preprocessing_sharpen_strength'], 0.1,
                help="Sharpening strength (1.0 = normal, higher = stronger)",
                key="preprocessing_sharpen_strength"
            )
        
        # Denoising - read from session_state (which is synced from config)
        config.denoise_enabled = st.sidebar.checkbox(
            "Enable Denoising",
            value=st.session_state['preprocessing_denoise_enabled'],
            help="Apply noise reduction",
            key="preprocessing_denoise_enabled"
        )
        if config.denoise_enabled:
            config.denoise_strength = st.sidebar.slider(
                "Denoising Strength",
                1, 20, st.session_state['preprocessing_denoise_strength'], 1,
                help="Noise reduction strength (1 = light, 20 = strong reduction)",
                key="preprocessing_denoise_strength"
            )
        
        # Histogram Equalization - read from session_state (which is synced from config)
        config.histogram_eq_enabled = st.sidebar.checkbox(
            "Enable Histogram Equalization",
            value=st.session_state['preprocessing_histogram_eq_enabled'],
            help="Improve image contrast using histogram equalization",
            key="preprocessing_histogram_eq_enabled"
        )
        if config.histogram_eq_enabled:
            config.histogram_eq_type = st.sidebar.selectbox(
                "Equalization Type",
                ["clahe", "global"],
                index=["clahe", "global"].index(st.session_state['preprocessing_histogram_eq_type']),
                help="CLAHE (better quality) or Global (faster)",
                key="preprocessing_histogram_eq_type"
            )
        
        # Subtle separator for sub-sections
        st.sidebar.markdown("<hr style='margin: 0.5rem 0; border: none; border-top: 1px solid #e0e0e0;'>", unsafe_allow_html=True)
        st.sidebar.subheader("Geometric Transformations")
        
        # Rotation - read from session_state (which is synced from config)
        config.rotation = float(st.sidebar.slider(
            "Rotation (degrees)",
            -180, 180, st.session_state['preprocessing_rotation'], 1,
            help="Rotate image (negative = counterclockwise, positive = clockwise)",
            key="preprocessing_rotation"
        ))
        
        # Flip - read from session_state (which is synced from config)
        col_flip_h, col_flip_v = st.sidebar.columns(2)
        with col_flip_h:
            config.flip_horizontal = st.checkbox(
                "Flip Horizontal",
                value=st.session_state['preprocessing_flip_horizontal'],
                help="Flip image horizontally",
                key="preprocessing_flip_horizontal"
            )
        with col_flip_v:
            config.flip_vertical = st.checkbox(
                "Flip Vertical",
                value=st.session_state['preprocessing_flip_vertical'],
                help="Flip image vertically",
                key="preprocessing_flip_vertical"
            )
        
        # Update session state after all changes
        st.session_state['preprocessing_config'] = config
    else:
        # Reset to default when disabled
        if st.session_state.get('preprocessing_config') != PreprocessingConfig():
            st.session_state['preprocessing_config'] = PreprocessingConfig()
        config = PreprocessingConfig()
    
    # Get selected model path from session state
    selected_model_path = st.session_state.get('selected_model_path', get_default_model().model_path)
    
    return (source_option, cam_index, confidence_threshold, target_fps, use_gpu, num_threads, 
            uploaded_file, uploaded_image, start_camera_clicked, enable_preprocessing, config, selected_model_path,
            folder_path, folder_mode, start_folder_clicked, ip_camera_url, start_ip_camera_clicked,
            enable_save_defects, defect_save_dir)


def render_main_dashboard(source_option):
    """
    Render main dashboard and return container references
    
    Returns:
        tuple: (st_frame, metric_fps, metric_actual_fps, defect_table_container)
    """
    st.title("🏭 Automated Defect Detection")
    
    # Header with reset button
    col_header1, col_header2 = st.columns([3, 1])
    with col_header1:
        st.markdown(f"**Mode:** {source_option} | **Status:** Ready")
    with col_header2:
        if st.button("🔄 Reset Statistics", use_container_width=True, help="Clear all cumulative defect statistics"):
            st.session_state['cumulative_defect_details'] = {}
            st.session_state['tracked_defect_ids'] = set()
            st.session_state['frames_processed'] = 0
            st.rerun()
    
    # Metrics at the top
    col_metrics = st.columns(2)
    
    with col_metrics[0]:
        metric_fps = st.empty()
    with col_metrics[1]:
        metric_actual_fps = st.empty()
    
    # Defect table container (above video)
    defect_table_container = st.empty()
    
    # Video frame below defect table
    st_frame = st.empty()
    
    return st_frame, metric_fps, metric_actual_fps, defect_table_container


def update_dashboard(st_frame, metric_fps, metric_actual_fps, defect_table_container,
                     frame_rgb, count, inf_time, actual_fps, defect_details):
    """
    Update dashboard with new frame and metrics
    
    Args:
        st_frame: Streamlit container for video frame
        metric_fps: Streamlit container for latency metric
        metric_actual_fps: Streamlit container for actual FPS metric
        defect_table_container: Streamlit container for defect table
        frame_rgb: Processed frame in RGB format
        count: Number of detections (kept for compatibility but not displayed)
        inf_time: Inference time in milliseconds
        actual_fps: Actual frames per second
        defect_details: dict with class names as keys and lists of confidence scores as values
    """
    # Update Metrics
    metric_fps.metric("Latency", f"{inf_time:.1f} ms")
    metric_actual_fps.metric("FPS", f"{actual_fps:.1f}")
    
    # Update Defect Table
    _render_defect_table(defect_table_container, defect_details)
    
    # Update Video
    st_frame.image(frame_rgb, channels="RGB", use_container_width=True)


def _render_defect_table(container, defect_details):
    """
    Render defect detection table with statistics (both current frame and cumulative)
    
    Args:
        container: Streamlit container to render table in
        defect_details: dict with class names as keys and lists of confidence scores as values (current frame)
    """
    import pandas as pd
    
    # Get cumulative defect details from session state
    cumulative_defect_details = st.session_state.get('cumulative_defect_details', {})
    
    with container.container():
        # Create tabs for Current Frame and Cumulative Statistics
        tab1, tab2 = st.tabs(["📊 Current Frame", "📈 Cumulative Statistics"])
        
        # ===== CURRENT FRAME TAB =====
        with tab1:
            if not defect_details:
                st.info("✅ No defects detected in current frame")
            else:
                # Prepare data for current frame table
                table_data = []
                total_defects = 0
                
                for class_name, defect_list in defect_details.items():
                    # defect_list contains (confidence, track_id, box_center) tuples
                    confidences = [conf for conf, _, _ in defect_list]
                    count = len(confidences)
                    total_defects += count
                    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
                    max_conf = max(confidences) if confidences else 0.0
                    min_conf = min(confidences) if confidences else 0.0
                    
                    table_data.append({
                        "Defect Type": class_name.replace("_", " ").title(),
                        "Count": count,
                        "Avg Confidence": f"{avg_conf:.2%}",
                        "Max Confidence": f"{max_conf:.2%}",
                        "Min Confidence": f"{min_conf:.2%}"
                    })
                
                # Sort by count (descending), then by name
                table_data.sort(key=lambda x: (-x["Count"], x["Defect Type"]))
                
                # Summary metrics for current frame
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Defects (Frame)", total_defects)
                with col2:
                    st.metric("Defect Types (Frame)", len(defect_details))
                with col3:
                    if total_defects > 0:
                        # Extract all confidences from all defect lists
                        all_confs = [conf for defect_list in defect_details.values() for conf, _, _ in defect_list]
                        avg_all_conf = sum(all_confs) / len(all_confs) if all_confs else 0.0
                        st.metric("Avg Confidence (Frame)", f"{avg_all_conf:.2%}")
                    else:
                        st.metric("Avg Confidence (Frame)", "N/A")
                
                # Display current frame table
                df = pd.DataFrame(table_data)
                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Defect Type": st.column_config.TextColumn("Defect Type", width="medium"),
                        "Count": st.column_config.NumberColumn("Count", width="small"),
                        "Avg Confidence": st.column_config.TextColumn("Avg Confidence", width="small"),
                        "Max Confidence": st.column_config.TextColumn("Max Confidence", width="small"),
                        "Min Confidence": st.column_config.TextColumn("Min Confidence", width="small"),
                    }
                )
        
        # ===== CUMULATIVE STATISTICS TAB =====
        with tab2:
            if not cumulative_defect_details:
                st.info("📊 No cumulative statistics yet. Start processing to see accumulated defect data")
            else:
                # Prepare data for cumulative table
                cumulative_table_data = []
                cumulative_total_defects = 0
                
                for class_name, confidences in cumulative_defect_details.items():
                    count = len(confidences)
                    cumulative_total_defects += count
                    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
                    max_conf = max(confidences) if confidences else 0.0
                    min_conf = min(confidences) if confidences else 0.0
                    
                    cumulative_table_data.append({
                        "Defect Type": class_name.replace("_", " ").title(),
                        "Total Count": count,
                        "Avg Confidence": f"{avg_conf:.2%}",
                        "Max Confidence": f"{max_conf:.2%}",
                        "Min Confidence": f"{min_conf:.2%}"
                    })
                
                # Sort by count (descending), then by name
                cumulative_table_data.sort(key=lambda x: (-x["Total Count"], x["Defect Type"]))
                
                # Summary metrics for cumulative
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Defects (All)", cumulative_total_defects)
                with col2:
                    st.metric("Defect Types (All)", len(cumulative_defect_details))
                with col3:
                    if cumulative_total_defects > 0:
                        avg_all_conf = sum(conf for confs in cumulative_defect_details.values() for conf in confs) / cumulative_total_defects
                        st.metric("Overall Avg Confidence", f"{avg_all_conf:.2%}")
                    else:
                        st.metric("Overall Avg Confidence", "N/A")
                with col4:
                    # Calculate frames processed (approximate)
                    frames_processed = st.session_state.get('frames_processed', 0)
                    st.metric("Frames Processed", frames_processed)
                
                # Display cumulative table
                df_cumulative = pd.DataFrame(cumulative_table_data)
                st.dataframe(
                    df_cumulative,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Defect Type": st.column_config.TextColumn("Defect Type", width="medium"),
                        "Total Count": st.column_config.NumberColumn("Total Count", width="small"),
                        "Avg Confidence": st.column_config.TextColumn("Avg Confidence", width="small"),
                        "Max Confidence": st.column_config.TextColumn("Max Confidence", width="small"),
                        "Min Confidence": st.column_config.TextColumn("Min Confidence", width="small"),
                    }
                )

