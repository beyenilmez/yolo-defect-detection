"""
Main Application Entry Point
Automated Defect Detection System for Quality Control
"""
import streamlit as st
from detection import DefectDetectionSystem
from video_processor import (
    process_video_stream,
    create_video_capture_from_file,
    create_camera_capture,
    create_ip_camera_capture,
    process_image,
    process_folder_sequential,
    process_folder_watch
)
from ui_components import (
    render_sidebar_controls,
    render_main_dashboard,
    update_dashboard
)

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Quality Control System", 
    page_icon="🏭", 
    layout="wide"
)

# --- INITIALIZE SYSTEM ---
# Will be initialized after UI controls are rendered

# --- RENDER UI ---
(source_option, cam_index, confidence_threshold, target_fps, use_gpu, num_threads, 
 uploaded_file, uploaded_image, start_camera_clicked, enable_preprocessing, preprocessing_config, model_path,
 folder_path, folder_mode, start_folder_clicked, ip_camera_url, start_ip_camera_clicked,
 enable_save_defects, defect_save_dir) = render_sidebar_controls()
st_frame, metric_fps, metric_actual_fps, defect_table_container = render_main_dashboard(source_option)

# Create message containers for status messages
status_message = st.empty()

# --- INITIALIZE SYSTEM ---
# Check if GPU/thread settings changed
settings_changed = False
restart_needed = False

try:
    if 'system' in st.session_state:
        old_use_gpu = st.session_state.get('use_gpu', None)
        old_num_threads = st.session_state.get('num_threads', None)
        old_source = st.session_state.get('source_option', None)
        old_cam_index = st.session_state.get('cam_index', None)
        old_confidence = st.session_state.get('confidence_threshold', None)
        old_target_fps = st.session_state.get('target_fps', None)
        old_enable_preprocessing = st.session_state.get('enable_preprocessing', None)
        old_preprocessing_config = st.session_state.get('preprocessing_config', None)
        old_model_path = st.session_state.get('model_path', None)
        old_folder_path = st.session_state.get('folder_path', None)
        old_folder_mode = st.session_state.get('folder_mode', None)
        old_ip_camera_url = st.session_state.get('ip_camera_url', None)
        
        # Check if any critical settings changed (that require stream restart)
        # Note: Preprocessing config changes don't require restart - they apply on the fly
        # Preprocessing is independent of YOLO and can be changed without stopping the stream
        if (old_use_gpu != use_gpu or old_num_threads != num_threads or 
            old_source != source_option or old_cam_index != cam_index or
            old_confidence != confidence_threshold or old_target_fps != target_fps or
            old_enable_preprocessing != enable_preprocessing or old_model_path != model_path or
            old_folder_path != folder_path or old_folder_mode != folder_mode or
            old_ip_camera_url != ip_camera_url):
            settings_changed = True
            restart_needed = True
            
        # Model path change requires system reload
        if old_model_path != model_path or old_use_gpu != use_gpu or old_num_threads != num_threads:
            # Clear cache to reload model with new settings
            st.cache_resource.clear()
            system = DefectDetectionSystem(model_path=model_path, use_gpu=use_gpu, num_threads=num_threads)
            st.session_state['system'] = system
            st.session_state['use_gpu'] = use_gpu
            st.session_state['num_threads'] = num_threads
            st.session_state['model_path'] = model_path
        else:
            system = st.session_state['system']
    else:
        system = DefectDetectionSystem(model_path=model_path, use_gpu=use_gpu, num_threads=num_threads)
        st.session_state['system'] = system
        st.session_state['use_gpu'] = use_gpu
        st.session_state['num_threads'] = num_threads
        st.session_state['model_path'] = model_path
    
    # Update session state with current settings
    st.session_state['source_option'] = source_option
    st.session_state['cam_index'] = cam_index
    st.session_state['confidence_threshold'] = confidence_threshold
    st.session_state['target_fps'] = target_fps
    st.session_state['enable_preprocessing'] = enable_preprocessing
    st.session_state['preprocessing_config'] = preprocessing_config
    st.session_state['folder_path'] = folder_path
    st.session_state['folder_mode'] = folder_mode
    st.session_state['ip_camera_url'] = ip_camera_url
    
except Exception as e:
    st.error(f"System Error: {e}")
    st.stop()

# --- PROCESSING LOGIC ---
def create_update_callback():
    """Create callback function for dashboard updates"""
    def callback(frame_rgb, count, inf_time, actual_fps, defect_details):
        update_dashboard(
            st_frame, metric_fps, metric_actual_fps, defect_table_container,
            frame_rgb, count, inf_time, actual_fps, defect_details
        )
    return callback

# Check if we need to restart due to settings change
active_stream = st.session_state.get('active_stream', False)
should_restart = restart_needed and active_stream

# Clear status message initially
status_message.empty()

# --- NONE MODE ---
if source_option == "None":
    status_message.info("Please select an input source to begin.")
    st.session_state['active_stream'] = False
    st.session_state['last_file_id'] = None

# --- IMAGE UPLOAD MODE ---
elif source_option == "Upload Image":
    if uploaded_image:
        # Check if this is a new file or settings changed
        file_id = id(uploaded_image)
        last_file_id = st.session_state.get('last_image_id', None)
        
        # Process image if it's new or settings changed
        if file_id != last_file_id or should_restart:
            if should_restart:
                status_message.warning("⚠️ Settings changed. Reprocessing image...")
            else:
                status_message.info("🔄 Processing image...")
            
            st.session_state['last_image_id'] = file_id
            st.session_state['active_stream'] = False  # Not a stream, single image
            
            try:
                update_callback = create_update_callback()
                process_image(
                    uploaded_image, system, confidence_threshold, update_callback,
                    enable_preprocessing=enable_preprocessing,
                    preprocessing_config=preprocessing_config
                )
                status_message.empty()
            except Exception as e:
                status_message.error(f"❌ Failed to process image: {str(e)}")
        else:
            # Image already processed, just clear status
            status_message.empty()
    else:
        # No image uploaded
        if not active_stream:
            status_message.info("Waiting for image upload...")
        else:
            status_message.empty()
        st.session_state['active_stream'] = False
        st.session_state['last_image_id'] = None

# --- VIDEO UPLOAD MODE ---
elif source_option == "Upload Video":
    if uploaded_file:
        # Check if this is a new file or settings changed
        file_id = id(uploaded_file)
        last_file_id = st.session_state.get('last_file_id', None)
        
        # Always start/restart stream if file changed or critical settings changed
        # For preprocessing-only changes, stream will continue and pick up new config from session_state
        if file_id != last_file_id or should_restart or not active_stream:
            if should_restart:
                status_message.warning("⚠️ Settings changed. Restarting video stream...")
            elif file_id != last_file_id:
                status_message.info("🔄 Loading video...")
            
            st.session_state['last_file_id'] = file_id
            st.session_state['active_stream'] = True
            
            # Reset cumulative statistics for new video
            st.session_state['cumulative_defect_details'] = {}
            st.session_state['tracked_defect_ids'] = set()
            st.session_state['frames_processed'] = 0
            
            cap = create_video_capture_from_file(uploaded_file)
            if cap and cap.isOpened():
                # Clear status message when stream starts
                status_message.empty()
                update_callback = create_update_callback()
                process_video_stream(
                    cap, system, confidence_threshold, target_fps, update_callback,
                    enable_preprocessing=enable_preprocessing,
                    preprocessing_config=preprocessing_config
                )
                st.session_state['active_stream'] = False
            else:
                status_message.error("❌ Failed to open video file.")
                st.session_state['active_stream'] = False
        else:
            # Stream is running, preprocessing config changes are handled in process_video_stream
            # via session_state, so we don't need to restart
            status_message.empty()
    else:
        # No file uploaded - show message only if not streaming
        if not active_stream:
            status_message.info("Waiting for video upload...")
        else:
            status_message.empty()
        st.session_state['active_stream'] = False
        st.session_state['last_file_id'] = None

# --- LIVE CAMERA MODE ---
elif source_option == "Live Camera":
    # Always start/restart stream if button clicked, critical settings changed, or stream not active
    # Stream will automatically pick up preprocessing config changes from session_state
    if start_camera_clicked or should_restart or active_stream:
        if should_restart:
            status_message.warning("⚠️ Settings changed. Restarting camera stream...")
        elif active_stream and not start_camera_clicked and not should_restart:
            # Stream is running, preprocessing changes will be picked up automatically
            # But we need to keep stream running, so restart it to pick up changes
            status_message.empty()
        
        st.session_state['active_stream'] = True
        
        # Reset cumulative statistics when starting new camera stream
        # Only reset if this is a new start (not a continuation)
        if start_camera_clicked or should_restart:
            st.session_state['cumulative_defect_details'] = {}
            st.session_state['tracked_defect_ids'] = set()
            st.session_state['frames_processed'] = 0
        
        cap = create_camera_capture(cam_index)
        
        if cap is None:
            status_message.error(
                f"❌ Cannot access camera (index {cam_index}). "
                "Please try a different camera index."
            )
            st.session_state['active_stream'] = False
        else:
            # Clear status message when stream starts
            status_message.empty()
            update_callback = create_update_callback()
            process_video_stream(
                cap, system, confidence_threshold, target_fps, update_callback,
                enable_preprocessing=enable_preprocessing,
                preprocessing_config=preprocessing_config
            )
            st.session_state['active_stream'] = False
    else:
        # Camera not started - show message only if not streaming
        if not active_stream:
            status_message.info(
                f"Camera index: {cam_index}. "
                "Click 'Start Camera' to begin streaming."
            )
        else:
            status_message.empty()

# --- IP CAMERA MODE ---
elif source_option == "IP Camera":
    # Always start/restart stream if button clicked, critical settings changed, or stream not active
    # Stream will automatically pick up preprocessing config changes from session_state
    if start_ip_camera_clicked or should_restart or active_stream:
        if should_restart:
            status_message.warning("⚠️ Settings changed. Restarting IP camera stream...")
        elif active_stream and not start_ip_camera_clicked and not should_restart:
            # Stream is running, preprocessing changes will be picked up automatically
            # But we need to keep stream running, so restart it to pick up changes
            status_message.empty()
        
        # Check if URL is provided
        if not ip_camera_url or not ip_camera_url.strip():
            status_message.error("❌ Please enter an IP camera URL")
            st.session_state['active_stream'] = False
        else:
            st.session_state['active_stream'] = True
            
            # Reset cumulative statistics when starting new IP camera stream
            # Only reset if this is a new start (not a continuation)
            if start_ip_camera_clicked or should_restart:
                st.session_state['cumulative_defect_details'] = {}
                st.session_state['tracked_defect_ids'] = set()
                st.session_state['frames_processed'] = 0
            
            cap = create_ip_camera_capture(ip_camera_url)
            
            if cap is None:
                status_message.error(
                    f"❌ Cannot connect to IP camera: {ip_camera_url}\n\n"
                    "Please check:\n"
                    "• URL format (e.g., rtsp://ip:port/stream, http://ip:port/video)\n"
                    "• Network connectivity\n"
                    "• Camera credentials (if required)\n"
                    "• Firewall settings"
                )
                st.session_state['active_stream'] = False
            else:
                # Clear status message when stream starts
                status_message.empty()
                update_callback = create_update_callback()
                process_video_stream(
                    cap, system, confidence_threshold, target_fps, update_callback,
                    enable_preprocessing=enable_preprocessing,
                    preprocessing_config=preprocessing_config
                )
                st.session_state['active_stream'] = False
    else:
        # IP Camera not started - show message only if not streaming
        if not active_stream:
            current_url = ip_camera_url if ip_camera_url else "Not set"
            status_message.info(
                f"IP camera URL: {current_url}\n"
                "Click 'Start IP Camera' to begin streaming."
            )
        else:
            status_message.empty()

# --- FOLDER MODE ---
elif source_option == "Folder":
    if folder_path and folder_path.strip():
        folder_path = folder_path.strip()
        
        # Check if folder exists
        import os
        if not os.path.exists(folder_path):
            status_message.error(f"❌ Folder does not exist: {folder_path}")
            st.session_state['active_stream'] = False
            st.session_state['watching_folder'] = False
        elif not os.path.isdir(folder_path):
            status_message.error(f"❌ Path is not a directory: {folder_path}")
            st.session_state['active_stream'] = False
            st.session_state['watching_folder'] = False
        else:
            # Check if we should start/restart processing
            last_folder_path = st.session_state.get('last_folder_path', None)
            last_folder_mode = st.session_state.get('last_folder_mode', None)
            
            should_start = (start_folder_clicked or should_restart or 
                          folder_path != last_folder_path or 
                          folder_mode != last_folder_mode or
                          not active_stream)
            
            if should_start:
                if should_restart:
                    status_message.warning("⚠️ Settings changed. Restarting folder processing...")
                elif folder_path != last_folder_path or folder_mode != last_folder_mode:
                    status_message.info(f"🔄 Starting {folder_mode.lower()}...")
                else:
                    status_message.info(f"🔄 Processing folder ({folder_mode.lower()})...")
                
                st.session_state['last_folder_path'] = folder_path
                st.session_state['last_folder_mode'] = folder_mode
                st.session_state['active_stream'] = True
                
                # Reset cumulative statistics for new folder processing
                st.session_state['cumulative_defect_details'] = {}
                st.session_state['tracked_defect_ids'] = set()
                st.session_state['frames_processed'] = 0
                
                # Clear processed files tracking if restarting
                if should_restart or folder_path != last_folder_path or folder_mode != last_folder_mode:
                    if folder_mode == "Process All":
                        st.session_state['processed_folder_files'] = set()
                        st.session_state['restart_folder_processing'] = True
                    else:  # Watch Folder
                        st.session_state['watched_folder_files'] = {}
                        st.session_state['watching_folder'] = True
                
                try:
                    update_callback = create_update_callback()
                    
                    if folder_mode == "Process All":
                        # Process all images sequentially
                        process_folder_sequential(
                            folder_path, system, confidence_threshold, target_fps, update_callback,
                            enable_preprocessing=enable_preprocessing,
                            preprocessing_config=preprocessing_config
                        )
                        status_message.success("✅ Finished processing all images in folder")
                        st.session_state['active_stream'] = False
                    else:  # Watch Folder
                        # Watch folder for new images
                        st.session_state['watching_folder'] = True
                        status_message.info("👀 Watching folder for new images... (Press Stop to cancel)")
                        process_folder_watch(
                            folder_path, system, confidence_threshold, target_fps, update_callback,
                            enable_preprocessing=enable_preprocessing,
                            preprocessing_config=preprocessing_config,
                            watch_interval=1.0  # Check folder every 1 second
                        )
                        st.session_state['watching_folder'] = False
                        st.session_state['active_stream'] = False
                        status_message.info("⏹️ Stopped watching folder")
                except Exception as e:
                    status_message.error(f"❌ Failed to process folder: {str(e)}")
                    st.session_state['active_stream'] = False
                    st.session_state['watching_folder'] = False
            else:
                # Processing is running, just clear status
                status_message.empty()
    else:
        # No folder path provided
        if not active_stream:
            status_message.info("📂 Enter a folder path and select processing mode to begin")
        else:
            status_message.empty()
        st.session_state['active_stream'] = False
        st.session_state['watching_folder'] = False
