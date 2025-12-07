"""
Video Processing Module
Handles video capture and frame processing logic
"""
import cv2
import tempfile
import time
import numpy as np
import streamlit as st
import os
from pathlib import Path
from datetime import datetime
from detection import DefectDetectionSystem
from preprocessing import apply_preprocessing, PreprocessingConfig


def save_defect_image(processed_img, defect_details, save_dir, use_tracking=False):
    """
    Save image with defects (overlays) to a directory with timestamp filename
    Only saves if there are new defects (not already saved) when tracking is enabled
    
    Args:
        processed_img: Processed image with overlays (BGR format)
        defect_details: Dictionary with defect information (used to check if defects exist)
        save_dir: Directory path where image will be saved
        use_tracking: Whether tracking is enabled (for video streams, to avoid saving same defect multiple times)
        
    Returns:
        str or None: Path to saved file if saved, None otherwise
    """
    # Only save if defects are detected
    if not defect_details or len(defect_details) == 0:
        return None
    
    # Check if saving is enabled
    if not st.session_state.get('enable_save_defects', False):
        return None
    
    # Initialize saved defect IDs registry if not exists
    if 'saved_defect_ids' not in st.session_state:
        st.session_state['saved_defect_ids'] = set()
    
    # If tracking is enabled, check if this frame has any new defects
    if use_tracking:
        has_new_defect = False
        
        for class_name, defect_list in defect_details.items():
            # defect_list contains (confidence, track_id, box_center) tuples
            for conf, track_id, box_center in defect_list:
                added = False
                
                # First try to use tracking ID if available
                if track_id is not None:
                    unique_id = (class_name, track_id)
                    # Only save if this tracking ID hasn't been saved before
                    if unique_id not in st.session_state['saved_defect_ids']:
                        st.session_state['saved_defect_ids'].add(unique_id)
                        has_new_defect = True
                        added = True
                
                # If tracking ID failed or not available, use box center as fallback
                if not added and box_center is not None:
                    # Use box center with a tolerance (same object if within ~50 pixels)
                    tolerance = 50
                    unique_id = (class_name, 'box', box_center[0] // tolerance, box_center[1] // tolerance)
                    if unique_id not in st.session_state['saved_defect_ids']:
                        st.session_state['saved_defect_ids'].add(unique_id)
                        has_new_defect = True
                        added = True
                
                # If we found a new defect, we can break (we'll save the image)
                if has_new_defect:
                    break
            
            if has_new_defect:
                break
        
        # Only save if there's at least one new defect
        if not has_new_defect:
            return None
    
    try:
        # Get save directory from session state
        save_directory = st.session_state.get('defect_save_dir', save_dir)
        
        # Create directory if it doesn't exist
        os.makedirs(save_directory, exist_ok=True)
        
        # Generate timestamp filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
        filename = f"defect_{timestamp}.jpg"
        filepath = os.path.join(save_directory, filename)
        
        # Save image (processed_img is in BGR format, which is what cv2.imwrite expects)
        cv2.imwrite(filepath, processed_img)
        
        return filepath
    except Exception as e:
        # Silently fail to avoid disrupting the main processing flow
        # Could optionally log the error
        return None


def process_video_stream(cap, system, confidence_threshold, target_fps, update_callback,
                        enable_preprocessing=False, preprocessing_config=None):
    """
    Process video stream frame by frame with FPS control
    
    Args:
        cap: OpenCV VideoCapture object
        system: DefectDetectionSystem instance
        confidence_threshold: Detection confidence threshold
        target_fps: Target frames per second
        update_callback: Function to call for each processed frame
        enable_preprocessing: Whether to apply preprocessing
        preprocessing_config: PreprocessingConfig object (initial config, will be updated from session_state)
    """
    frame_time = 1.0 / target_fps if target_fps > 0 else 0.033  # Default to ~30 FPS
    last_frame_time = time.time()
    frame_count = 0
    fps_start_time = time.time()
    
    # Initialize cumulative defect details if starting new stream
    if 'cumulative_defect_details' not in st.session_state:
        st.session_state['cumulative_defect_details'] = {}
    if 'frames_processed' not in st.session_state:
        st.session_state['frames_processed'] = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get current preprocessing config from session_state (allows live updates without restart)
        current_enable_preprocessing = st.session_state.get('enable_preprocessing', enable_preprocessing)
        current_preprocessing_config = st.session_state.get('preprocessing_config', preprocessing_config)
        
        # Apply preprocessing if enabled (using current config from session_state)
        if current_enable_preprocessing and current_preprocessing_config is not None:
            preprocessed_frame = apply_preprocessing(frame, current_preprocessing_config)
        else:
            preprocessed_frame = frame
        
        # Process frame with tracking enabled for video streams
        processed_img, count, inf_time, defect_details = system.process_frame(
            preprocessed_frame, confidence_threshold, use_tracking=True
        )
        
        # Store defect details in session state for UI display
        st.session_state['current_defect_details'] = defect_details
        
        # Initialize tracking ID registry if not exists
        if 'tracked_defect_ids' not in st.session_state:
            st.session_state['tracked_defect_ids'] = set()
        
        # Update cumulative defect details (only count each tracking ID once)
        if 'cumulative_defect_details' not in st.session_state:
            st.session_state['cumulative_defect_details'] = {}
        
        # Add current frame's defects to cumulative data, but only if not already tracked
        for class_name, defect_list in defect_details.items():
            if class_name not in st.session_state['cumulative_defect_details']:
                st.session_state['cumulative_defect_details'][class_name] = []
            
            # defect_list contains (confidence, track_id, box_center) tuples
            for conf, track_id, box_center in defect_list:
                added = False
                
                # First try to use tracking ID if available
                if track_id is not None:
                    unique_id = (class_name, track_id)
                    # Only add if this tracking ID hasn't been seen before
                    if unique_id not in st.session_state['tracked_defect_ids']:
                        st.session_state['tracked_defect_ids'].add(unique_id)
                        st.session_state['cumulative_defect_details'][class_name].append(conf)
                        added = True
                
                # If tracking ID failed or not available, use box center as fallback
                # This helps when tracking IDs are inconsistent or None
                if not added and box_center is not None:
                    # Use box center with a tolerance (same object if within ~50 pixels)
                    tolerance = 50
                    unique_id = (class_name, 'box', box_center[0] // tolerance, box_center[1] // tolerance)
                    if unique_id not in st.session_state['tracked_defect_ids']:
                        st.session_state['tracked_defect_ids'].add(unique_id)
                        st.session_state['cumulative_defect_details'][class_name].append(conf)
                        added = True
                
                # Last resort: if neither tracking ID nor box center available, skip
                # This prevents counting the same detection multiple times
                if not added:
                    # Skip this detection to avoid double counting
                    pass
        
        # Save defect image if enabled and defects are detected (with tracking to avoid duplicates)
        save_defect_image(processed_img, defect_details, './defect_images', use_tracking=True)
        
        # Convert BGR to RGB for display
        processed_img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
        
        # Calculate actual FPS
        frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - fps_start_time
        if elapsed_time > 0:
            actual_fps = frame_count / elapsed_time
        else:
            actual_fps = 0
        
        # Update frames processed counter
        st.session_state['frames_processed'] = st.session_state.get('frames_processed', 0) + 1
        
        # Update dashboard
        update_callback(processed_img_rgb, count, inf_time, actual_fps, defect_details)
        
        # FPS control: sleep if processing too fast
        elapsed_since_last = current_time - last_frame_time
        if elapsed_since_last < frame_time:
            time.sleep(frame_time - elapsed_since_last)
        
        last_frame_time = time.time()
    
    cap.release()


def create_video_capture_from_file(uploaded_file):
    """
    Create VideoCapture object from uploaded file
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        cv2.VideoCapture object or None if failed
    """
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    tfile.close()
    return cv2.VideoCapture(tfile.name)


def create_camera_capture(camera_index):
    """
    Create VideoCapture object from camera
    
    Args:
        camera_index: Camera device index
        
    Returns:
        cv2.VideoCapture object or None if failed
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return None
    return cap


def create_ip_camera_capture(ip_camera_url):
    """
    Create VideoCapture object from IP camera URL
    
    Args:
        ip_camera_url: IP camera URL (e.g., rtsp://ip:port/stream, http://ip:port/video)
        
    Returns:
        cv2.VideoCapture object or None if failed
    """
    if not ip_camera_url or not ip_camera_url.strip():
        return None
    
    # OpenCV can handle various IP camera protocols:
    # - RTSP: rtsp://username:password@ip:port/stream
    # - HTTP: http://ip:port/video
    # - MJPEG: http://ip:port/mjpeg_stream
    cap = cv2.VideoCapture(ip_camera_url.strip())
    
    # Set buffer size to reduce latency (optional)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        return None
    
    # Try to read a frame to verify connection
    ret, _ = cap.read()
    if not ret:
        cap.release()
        return None
    
    return cap


def process_image(uploaded_image, system, confidence_threshold, update_callback,
                 enable_preprocessing=False, preprocessing_config=None):
    """
    Process a single uploaded image
    
    Args:
        uploaded_image: Streamlit uploaded file object (image)
        system: DefectDetectionSystem instance
        confidence_threshold: Detection confidence threshold
        update_callback: Function to call with processed image
        enable_preprocessing: Whether to apply preprocessing
        preprocessing_config: PreprocessingConfig object
    """
    import time
    
    # Read image from uploaded file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    tfile.write(uploaded_image.read())
    tfile.close()
    
    # Load image using OpenCV
    frame = cv2.imread(tfile.name)
    if frame is None:
        raise ValueError("Failed to load image file")
    
    # Get current preprocessing config from session_state
    current_enable_preprocessing = st.session_state.get('enable_preprocessing', enable_preprocessing)
    current_preprocessing_config = st.session_state.get('preprocessing_config', preprocessing_config)
    
    # Apply preprocessing if enabled
    if current_enable_preprocessing and current_preprocessing_config is not None:
        preprocessed_frame = apply_preprocessing(frame, current_preprocessing_config)
    else:
        preprocessed_frame = frame
    
    # Process frame (no tracking for single images)
    start_time = time.time()
    processed_img, count, inf_time, defect_details = system.process_frame(
        preprocessed_frame, confidence_threshold, use_tracking=False
    )
    end_time = time.time()
    
    # Store defect details in session state for UI display
    st.session_state['current_defect_details'] = defect_details
    
    # For single image, also update cumulative (though it's just one frame)
    # Since there's no tracking, we just add all detections
    if 'cumulative_defect_details' not in st.session_state:
        st.session_state['cumulative_defect_details'] = {}
    
    # Add current image's defects to cumulative data
    # defect_details contains (confidence, track_id, box_center) tuples, but track_id will be None for images
    for class_name, defect_list in defect_details.items():
        if class_name not in st.session_state['cumulative_defect_details']:
            st.session_state['cumulative_defect_details'][class_name] = []
        # Extract just the confidence values (track_id is None for images)
        for conf, _, _ in defect_list:
            st.session_state['cumulative_defect_details'][class_name].append(conf)
    
    # Save defect image if enabled and defects are detected (no tracking for single images)
    save_defect_image(processed_img, defect_details, './defect_images', use_tracking=False)
    
    # Convert BGR to RGB for display
    processed_img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
    
    # For single image, FPS is not applicable, use 0 or N/A
    actual_fps = 0.0
    
    # Update dashboard
    update_callback(processed_img_rgb, count, inf_time, actual_fps, defect_details)


def get_image_files(folder_path):
    """
    Get all image files from a folder
    
    Args:
        folder_path: Path to folder
        
    Returns:
        List of image file paths
    """
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        return []
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'}
    
    image_files = []
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path):
            ext = os.path.splitext(file)[1].lower()
            if ext in image_extensions:
                image_files.append(file_path)
    
    # Sort by filename for consistent processing order
    image_files.sort()
    return image_files


def process_folder_sequential(folder_path, system, confidence_threshold, target_fps, update_callback,
                             enable_preprocessing=False, preprocessing_config=None):
    """
    Process all images in a folder sequentially
    
    Args:
        folder_path: Path to folder containing images
        system: DefectDetectionSystem instance
        confidence_threshold: Detection confidence threshold
        target_fps: Target frames per second (for delay between images)
        update_callback: Function to call for each processed image
        enable_preprocessing: Whether to apply preprocessing
        preprocessing_config: PreprocessingConfig object
    """
    image_files = get_image_files(folder_path)
    
    if not image_files:
        raise ValueError(f"No image files found in folder: {folder_path}")
    
    frame_time = 1.0 / target_fps if target_fps > 0 else 0.033
    last_frame_time = time.time()
    frame_count = 0
    fps_start_time = time.time()
    
    # Initialize cumulative defect details if starting new stream
    if 'cumulative_defect_details' not in st.session_state:
        st.session_state['cumulative_defect_details'] = {}
    if 'frames_processed' not in st.session_state:
        st.session_state['frames_processed'] = 0
    if 'tracked_defect_ids' not in st.session_state:
        st.session_state['tracked_defect_ids'] = set()
    
    # Track processed files to avoid reprocessing
    processed_files = st.session_state.get('processed_folder_files', set())
    
    for image_path in image_files:
        # Skip if already processed (unless restarting)
        if image_path in processed_files and not st.session_state.get('restart_folder_processing', False):
            continue
        
        # Load image
        frame = cv2.imread(image_path)
        if frame is None:
            continue  # Skip invalid images
        
        # Get current preprocessing config from session_state
        current_enable_preprocessing = st.session_state.get('enable_preprocessing', enable_preprocessing)
        current_preprocessing_config = st.session_state.get('preprocessing_config', preprocessing_config)
        
        # Apply preprocessing if enabled
        if current_enable_preprocessing and current_preprocessing_config is not None:
            preprocessed_frame = apply_preprocessing(frame, current_preprocessing_config)
        else:
            preprocessed_frame = frame
        
        # Process frame (no tracking for folder images, each is independent)
        processed_img, count, inf_time, defect_details = system.process_frame(
            preprocessed_frame, confidence_threshold, use_tracking=False
        )
        
        # Store defect details in session state
        st.session_state['current_defect_details'] = defect_details
        
        # Update cumulative defect details
        if 'cumulative_defect_details' not in st.session_state:
            st.session_state['cumulative_defect_details'] = {}
        
        # Add defects to cumulative data (each image is independent, so count all)
        for class_name, defect_list in defect_details.items():
            if class_name not in st.session_state['cumulative_defect_details']:
                st.session_state['cumulative_defect_details'][class_name] = []
            # Extract confidence values
            for conf, _, _ in defect_list:
                st.session_state['cumulative_defect_details'][class_name].append(conf)
        
        # Mark file as processed
        processed_files.add(image_path)
        st.session_state['processed_folder_files'] = processed_files
        
        # Save defect image if enabled and defects are detected (no tracking for folder images)
        save_defect_image(processed_img, defect_details, './defect_images', use_tracking=False)
        
        # Convert BGR to RGB for display
        processed_img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
        
        # Calculate actual FPS
        frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - fps_start_time
        if elapsed_time > 0:
            actual_fps = frame_count / elapsed_time
        else:
            actual_fps = 0
        
        # Update frames processed counter
        st.session_state['frames_processed'] = st.session_state.get('frames_processed', 0) + 1
        
        # Update dashboard
        update_callback(processed_img_rgb, count, inf_time, actual_fps, defect_details)
        
        # FPS control: sleep if processing too fast
        elapsed_since_last = current_time - last_frame_time
        if elapsed_since_last < frame_time:
            time.sleep(frame_time - elapsed_since_last)
        
        last_frame_time = time.time()
    
    # Clear restart flag after processing
    if st.session_state.get('restart_folder_processing', False):
        st.session_state['restart_folder_processing'] = False


def process_folder_watch(folder_path, system, confidence_threshold, target_fps, update_callback,
                         enable_preprocessing=False, preprocessing_config=None, watch_interval=1.0):
    """
    Watch folder for new images and process them as they arrive
    
    Args:
        folder_path: Path to folder to watch
        system: DefectDetectionSystem instance
        confidence_threshold: Detection confidence threshold
        target_fps: Target frames per second (for delay between images)
        update_callback: Function to call for each processed image
        enable_preprocessing: Whether to apply preprocessing
        preprocessing_config: PreprocessingConfig object
        watch_interval: Time in seconds between folder checks
    """
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        raise ValueError(f"Folder does not exist: {folder_path}")
    
    frame_time = 1.0 / target_fps if target_fps > 0 else 0.033
    last_frame_time = time.time()
    frame_count = 0
    fps_start_time = time.time()
    
    # Initialize cumulative defect details if starting new stream
    if 'cumulative_defect_details' not in st.session_state:
        st.session_state['cumulative_defect_details'] = {}
    if 'frames_processed' not in st.session_state:
        st.session_state['frames_processed'] = 0
    if 'tracked_defect_ids' not in st.session_state:
        st.session_state['tracked_defect_ids'] = set()
    
    # Track processed files by modification time to detect new files
    processed_files = st.session_state.get('watched_folder_files', {})  # {file_path: mtime}
    
    # Watch for a reasonable amount of time (or until stopped)
    max_watch_time = 3600  # 1 hour max
    watch_start_time = time.time()
    
    while (time.time() - watch_start_time) < max_watch_time:
        # Check if we should stop watching (user changed source or settings)
        if not st.session_state.get('watching_folder', False):
            break
        
        # Get current list of image files
        current_files = get_image_files(folder_path)
        
        # Find new or modified files
        new_files = []
        for image_path in current_files:
            try:
                mtime = os.path.getmtime(image_path)
                if image_path not in processed_files or processed_files[image_path] < mtime:
                    new_files.append((image_path, mtime))
            except OSError:
                continue  # Skip files that can't be accessed
        
        # Process new files
        for image_path, mtime in new_files:
            # Load image
            frame = cv2.imread(image_path)
            if frame is None:
                continue  # Skip invalid images
            
            # Get current preprocessing config from session_state
            current_enable_preprocessing = st.session_state.get('enable_preprocessing', enable_preprocessing)
            current_preprocessing_config = st.session_state.get('preprocessing_config', preprocessing_config)
            
            # Apply preprocessing if enabled
            if current_enable_preprocessing and current_preprocessing_config is not None:
                preprocessed_frame = apply_preprocessing(frame, current_preprocessing_config)
            else:
                preprocessed_frame = frame
            
            # Process frame (no tracking for folder images)
            processed_img, count, inf_time, defect_details = system.process_frame(
                preprocessed_frame, confidence_threshold, use_tracking=False
            )
            
            # Store defect details in session state
            st.session_state['current_defect_details'] = defect_details
            
            # Update cumulative defect details
            if 'cumulative_defect_details' not in st.session_state:
                st.session_state['cumulative_defect_details'] = {}
            
            # Add defects to cumulative data
            for class_name, defect_list in defect_details.items():
                if class_name not in st.session_state['cumulative_defect_details']:
                    st.session_state['cumulative_defect_details'][class_name] = []
                # Extract confidence values
                for conf, _, _ in defect_list:
                    st.session_state['cumulative_defect_details'][class_name].append(conf)
            
            # Mark file as processed with its modification time
            processed_files[image_path] = mtime
            st.session_state['watched_folder_files'] = processed_files
            
            # Save defect image if enabled and defects are detected (no tracking for folder images)
            save_defect_image(processed_img, defect_details, './defect_images', use_tracking=False)
            
            # Convert BGR to RGB for display
            processed_img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
            
            # Calculate actual FPS
            frame_count += 1
            current_time = time.time()
            elapsed_time = current_time - fps_start_time
            if elapsed_time > 0:
                actual_fps = frame_count / elapsed_time
            else:
                actual_fps = 0
            
            # Update frames processed counter
            st.session_state['frames_processed'] = st.session_state.get('frames_processed', 0) + 1
            
            # Update dashboard
            update_callback(processed_img_rgb, count, inf_time, actual_fps, defect_details)
            
            # FPS control: sleep if processing too fast
            elapsed_since_last = current_time - last_frame_time
            if elapsed_since_last < frame_time:
                time.sleep(frame_time - elapsed_since_last)
            
            last_frame_time = time.time()
        
        # Sleep before next folder check
        time.sleep(watch_interval)

