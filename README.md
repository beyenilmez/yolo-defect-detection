# Manufacturing Defect Detection System

An automated quality control system for manufacturing lines that uses deep learning (YOLO) to detect defects in real-time. This Streamlit-based application provides a comprehensive interface for visual inspection with multiple input sources, advanced preprocessing, and detailed defect tracking.

## 📖 Overview

This project implements an automated defect detection system for manufacturing quality control:

- **Goal:** Prevent defective items from reaching customers through automated visual inspection
- **Method:** Deep Learning using YOLO (You Only Look Once) models for real-time object detection
- **Pipeline:** Image Acquisition → Preprocessing → Model Inference → Decision & Tracking

## ✨ Key Features

### 📹 Multiple Input Sources
- **Upload Image**: Process single images (JPG, PNG, BMP, WEBP)
- **Upload Video**: Analyze pre-recorded video files (MP4, AVI, MOV)
- **Live Camera**: Real-time processing from webcam or USB cameras
- **IP Camera**: Connect to network cameras via RTSP/HTTP protocols
- **Folder Processing**: 
  - Process all images in a folder sequentially
  - Watch folder mode for continuous monitoring of new images

### 🧠 Model Management
- **Model Registry System**: Manage multiple YOLO models for different defect types
- **Multiple Pre-trained Models**: 
  - **Steel Defect Detection**: Crazing, Inclusion, Patches, Pitted surface, Rolled-in scale, Scratches
  - **PCB Defect Detection**: Copper, Mousebite, Open, Pin-hole, Short, Spur
  - **Fabric Defect Detection**: Hole, Objects, Oil spot, Thread error
  - **Leather Defect Detection**: Bacterial Injury, Crease, Growth Marks, Healed Injury, Hole, Rotten surface, Scratch, Pinhole
  - **Wood Defect Detection**: Blue_Stain, Crack, Dead_Knot, Knot_missing, Live_Knot, Marrow, Quartzity, Knot_with_crack, Overgrown, Resin
- **Easy Model Switching**: Select models from dropdown menu
- **Model Information Display**: Shows model description and detected classes

### 🎨 Advanced Image Preprocessing
Comprehensive preprocessing pipeline with presets and custom configurations:

**Basic Adjustments:**
- Brightness adjustment (-100 to 100)
- Contrast adjustment (-100 to 100)
- Gamma correction (0.1 to 3.0)

**Resize Options:**
- None (original size)
- Scale factor (0.1x to 3.0x)
- Custom dimensions

**Filters:**
- Blur (Gaussian, Median, Bilateral)
- Sharpening with adjustable strength
- Denoising (non-local means)
- Histogram equalization (CLAHE or Global)

**Geometric Transformations:**
- Rotation (-180° to 180°)
- Horizontal/vertical flip

**Presets Available:**
- None (default)
- Low Light Enhancement
- High Contrast
- Noise Reduction
- Sharpening
- Blur Reduction
- Balanced Enhancement

### 📊 Defect Tracking & Statistics
- **Real-time Detection**: Live frame-by-frame defect detection
- **Object Tracking**: Track defects across video frames to avoid duplicate counting
- **Current Frame Statistics**: 
  - Defect count per type
  - Average, max, and min confidence scores
- **Cumulative Statistics**: 
  - Total defects detected across all frames
  - Overall confidence metrics
  - Frames processed counter
- **Interactive Dashboard**: Tabbed interface for current vs. cumulative data

### 💾 Defect Image Saving
- **Automatic Saving**: Save images with detected defects (including overlays)
- **Smart Tracking**: Avoids saving duplicate defects in video streams
- **Timestamped Files**: Automatic filename generation with timestamps
- **Configurable Directory**: Custom save location

### ⚡ Performance Features
- **GPU Acceleration**: Automatic CUDA detection and utilization
- **CPU Thread Control**: Adjustable number of threads for data loading
- **FPS Control**: Configurable target frames per second (1-60 FPS)
- **Real-time Metrics**: Display latency and actual FPS
- **Model Caching**: Efficient model loading with Streamlit caching

### 🎛️ User Interface
- **Streamlit Dashboard**: Modern, responsive web interface
- **Sidebar Controls**: Organized control panel with sections
- **Live Video Display**: Real-time annotated video feed
- **Statistics Reset**: One-click reset of cumulative statistics
- **Status Messages**: Clear feedback for all operations

## ⚙️ Prerequisites

- **Python**: 3.8 or higher
- **Hardware**: 
  - Webcam/USB camera (for live camera mode)
  - GPU with CUDA support (optional, for faster inference)
- **Dependencies**: See `requirements.txt`

## 🚀 Installation & Setup

### 1. Clone/Setup Directory
```bash
# Navigate to project directory
cd "path/to/project"
```

### 2. Create Virtual Environment
Isolate dependencies to avoid conflicts:

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Model Setup
Ensure the model files exist in the `models/` directory:
- **Steel Defect Detection**: `models/steel.pt` (default)
- **PCB Defect Detection**: `models/pcb.pt`
- **Fabric Defect Detection**: `models/fabric.pt`
- **Leather Defect Detection**: `models/leather.pt`
- **Wood Defect Detection**: `models/wood.pt`

The model registry (`src/model_registry.py`) manages all available models. Only models with existing `.pt` files will be available in the dropdown menu. To add new models, register them in the model registry.

## ▶️ How to Run

Start the Streamlit application:

```bash
streamlit run app.py
```

A browser window will open automatically at `http://localhost:8501`.

## 📖 Usage Guide

### Basic Workflow

1. **Select Input Source**: Choose from sidebar (None, Upload Image, Upload Video, Live Camera, IP Camera, or Folder)

2. **Configure Model Settings**:
   - Select YOLO model from dropdown
   - Adjust confidence threshold (0.1 to 1.0)
   - Set target FPS (1-60)

3. **Performance Settings**:
   - Enable/disable GPU acceleration
   - Adjust CPU thread count

4. **Preprocessing (Optional)**:
   - Enable preprocessing checkbox
   - Select a preset or configure custom settings
   - Adjust parameters as needed

5. **Defect Saving (Optional)**:
   - Enable "Save Images with Defects"
   - Specify save directory (default: `./defect_images`)

6. **Start Processing**:
   - For Live Camera: Click "Start Camera"
   - For IP Camera: Enter URL and click "Start IP Camera"
   - For Folder: Enter path, select mode, click "Start Processing"
   - For Upload: File selection starts processing automatically

### Input Source Details

#### Upload Image
- Supports: JPG, JPEG, PNG, BMP, WEBP
- Single image processing
- No tracking (each detection is independent)

#### Upload Video
- Supports: MP4, AVI, MOV
- Frame-by-frame processing with tracking
- Automatic stream management

#### Live Camera
- Select camera index (0 = default, 1+ = external cameras)
- Real-time streaming with tracking
- Click "Start Camera" to begin

#### IP Camera
- Supports RTSP and HTTP protocols
- URL format examples:
  - `rtsp://192.168.1.100:554/stream`
  - `http://192.168.1.100:8080/video`
- Enter URL and click "Start IP Camera"

#### Folder Processing
- **Process All Mode**: Processes all images in folder sequentially
- **Watch Folder Mode**: Monitors folder for new images and processes them automatically
- Supports: JPG, JPEG, PNG, BMP, WEBP, TIFF

### Understanding the Dashboard

**Metrics:**
- **Latency**: Inference time per frame in milliseconds
- **FPS**: Actual frames processed per second

**Defect Tables:**
- **Current Frame Tab**: Shows defects detected in the current frame
- **Cumulative Statistics Tab**: Shows all defects detected since start/reset
- Columns: Defect Type, Count, Avg/Max/Min Confidence

**Video Display:**
- Annotated frames with bounding boxes and labels
- Color-coded by defect type
- Confidence scores displayed on boxes

## 🧠 Model Information

### Available Models

The system includes multiple pre-trained YOLO models for different defect detection scenarios:

#### 1. Steel Defect Detection
- **Model Path**: `models/steel.pt`
- **Classes Detected**: 
  - Crazing
  - Inclusion
  - Patches
  - Pitted surface
  - Rolled-in scale
  - Scratches
- **Use Case**: Quality control for steel surface manufacturing

#### 2. PCB Defect Detection
- **Model Path**: `models/pcb.pt`
- **Classes Detected**: 
  - Copper
  - Mousebite
  - Open
  - Pin-hole
  - Short
  - Spur
- **Use Case**: Printed circuit board quality inspection

#### 3. Fabric Defect Detection
- **Model Path**: `models/fabric.pt`
- **Classes Detected**: 
  - Hole
  - Objects
  - Oil spot
  - Thread error
- **Use Case**: Textile manufacturing quality control

#### 4. Leather Defect Detection
- **Model Path**: `models/leather.pt`
- **Classes Detected**: 
  - Bacterial Injury
  - Crease
  - Growth Marks
  - Healed Injury
  - Hole
  - Rotten surface
  - Scratch
  - Pinhole
- **Use Case**: Leather product quality inspection

#### 5. Wood Defect Detection
- **Model Path**: `models/wood.pt`
- **Classes Detected**: 
  - Blue_Stain
  - Crack
  - Dead_Knot
  - Knot_missing
  - Live_Knot
  - Marrow
  - Quartzity
  - Knot_with_crack
  - Overgrown
  - Resin
- **Use Case**: Wood quality assessment and grading

### Adding Custom Models

To add a new model:

1. Place your trained `.pt` file in the `models/` directory
2. Register the model in `src/model_registry.py`:

```python
CUSTOM_MODEL = ModelInfo(
    id="custom_model",
    name="Custom Model Name",
    description="Description of what this model detects",
    model_path="models/custom_model.pt",
    classes=["class1", "class2", "class3"],
    task="detect"
)

MODEL_REGISTRY["custom_model"] = CUSTOM_MODEL
```

3. Restart the application to see the new model in the dropdown

## 🛠️ Technical Details

### Architecture
- **Frontend**: Streamlit web framework
- **Backend**: Python with OpenCV and Ultralytics YOLO
- **Model Framework**: YOLOv8 (Ultralytics)
- **Image Processing**: OpenCV
- **Tracking**: Built-in YOLO tracking with fallback to box center matching

### File Structure
```
project/
├── app.py                 # Main application entry point (Streamlit)
├── src/                   # Source code package
│   ├── __init__.py       # Package initialization
│   ├── detection.py       # Defect detection system
│   ├── video_processor.py # Video/image processing logic
│   ├── ui_components.py   # Streamlit UI components
│   ├── preprocessing.py  # Image preprocessing module
│   └── model_registry.py # Model management system
├── models/               # YOLO model files
│   ├── steel.pt         # Steel defect detection model
│   ├── pcb.pt           # PCB defect detection model
│   ├── fabric.pt        # Fabric defect detection model
│   ├── leather.pt       # Leather defect detection model
│   └── wood.pt          # Wood defect detection model
├── test/                 # Test data
│   ├── images/           # Test images
│   └── videos/          # Test videos
├── requirements.txt      # Python dependencies
├── README.md            # Project documentation
└── defect_images/        # Saved defect images (created automatically)
```

### Key Modules

**`detection.py`**: 
- `DefectDetectionSystem` class
- Model loading and caching
- Frame processing with tracking support
- Resolution-adaptive overlay scaling

**`video_processor.py`**:
- Video capture management (file, camera, IP)
- Frame processing loops
- Defect image saving
- Folder watching and processing

**`preprocessing.py`**:
- `PreprocessingConfig` dataclass
- Individual preprocessing functions
- Preset configurations

**`model_registry.py`**:
- `ModelInfo` dataclass
- Model registry management
- Model discovery and selection

**`ui_components.py`**:
- Sidebar controls rendering
- Dashboard layout
- Defect table display
- Statistics management

## ⚠️ Troubleshooting

### GPU Not Detected
If GPU is not detected on Windows, PyTorch may be installed without CUDA support:

```bash
# Uninstall existing PyTorch installations
pip uninstall torch torchvision ultralytics

# Install PyTorch with CUDA support
# For CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For CUDA 13.0:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# Reinstall Ultralytics
pip install ultralytics
```

### Camera Not Working
- Check camera permissions
- Try different camera indices (0, 1, 2, 3)
- Ensure camera is not being used by another application
- On Linux, check video device permissions: `ls -l /dev/video*`

### IP Camera Connection Issues
- Verify URL format (RTSP or HTTP)
- Check network connectivity
- Ensure camera credentials are included in URL if required: `rtsp://user:pass@ip:port/stream`
- Test URL with VLC or other media player first
- Check firewall settings

### Model File Not Found
- Ensure model file exists at specified path in `model_registry.py`
- Check file permissions
- Verify model file is a valid YOLO `.pt` file

### Low Performance
- Enable GPU acceleration if available
- Reduce target FPS
- Disable preprocessing if not needed
- Reduce image resolution using preprocessing resize
- Use smaller/faster YOLO model variant

### Folder Processing Issues
- Verify folder path is correct and accessible
- Check file permissions
- Ensure folder contains supported image formats
- For watch mode, ensure folder is not being actively written to by another process

## 📝 Notes

- **Tracking**: Object tracking is enabled for video streams to avoid duplicate counting. Single images use independent detection.
- **Preprocessing**: Changes to preprocessing settings apply immediately without restarting the stream (except for resize mode changes).
- **Statistics**: Cumulative statistics persist until manually reset or application restart.
- **Defect Saving**: Images are saved with full annotations (bounding boxes and labels) for later review.

## 🔮 Future Enhancements

Potential improvements for future versions:
- Real-time alerts/notifications
- Model training interface
- Database integration for defect logging
- Multi-camera support
- Custom annotation tools

## 📄 License

This project is part of the "Defect Detection in Manufacturing Lines" project (Group 10) for CME 4434 - Data Warehouses and Business Intelligence.

## 👥 Credits

Developed as part of academic coursework focusing on automated quality control in manufacturing environments.
