# Vehicle Counting and Traffic Flow Monitoring Using YOLO11, ByteTrack and OpenCV

An intelligent computer vision-based traffic monitoring system that detects, tracks, and counts vehicles from traffic surveillance videos using YOLO11x, ByteTrack, and OpenCV.

## Project Overview

This project aims to automate vehicle counting and basic traffic-flow monitoring using computer vision and deep learning.

The system processes traffic video frames, detects multiple vehicle categories, assigns persistent tracking IDs, and counts vehicles crossing predefined virtual lines. It also monitors vehicle occupancy within selected Regions of Interest (ROIs) and estimates traffic conditions using vehicle density and apparent movement.

## Problem Statement

Manual vehicle counting and traffic monitoring from surveillance footage are time-consuming and may produce inconsistent results, particularly in crowded traffic conditions.

The proposed system addresses this problem by automating vehicle detection, multi-object tracking, directional counting, ROI-based occupancy estimation, and basic traffic-condition monitoring.

## Objectives

- Detect vehicles from traffic surveillance videos.
- Identify cars, motorcycles, buses, and trucks.
- Track vehicles across consecutive video frames.
- Assign persistent tracking IDs using ByteTrack.
- Count vehicles crossing predefined virtual lines.
- Monitor vehicle occupancy in left and right ROIs.
- Estimate traffic conditions using vehicle density and movement information.
- Generate annotated traffic-monitoring videos.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| YOLO11x | Vehicle detection |
| ByteTrack | Multi-object tracking |
| OpenCV | Video processing and visualization |
| PyTorch | Deep learning framework |
| NumPy | Numerical operations |
| CUDA | GPU acceleration |

## Dataset Description

**Dataset:** UA-DETRAC

UA-DETRAC is the traffic-surveillance dataset identified for vehicle detection model development and evaluation.

Target vehicle categories:

- Car
- Motorcycle
- Bus
- Truck

Dataset access and preparation instructions are provided in [DATASET.md](DATASET.md).

The demonstration video used by the supplied video-processing script is separate from UA-DETRAC.

## System Architecture

The project follows this processing pipeline:

1. Traffic video input.
2. Frame extraction and resizing.
3. Vehicle detection using YOLO11x.
4. Multi-object tracking using ByteTrack.
5. Object-center smoothing.
6. ROI and virtual-line processing.
7. Directional vehicle counting.
8. ROI occupancy and traffic-condition estimation.
9. Annotated output video generation.

## Project Structure

```text
Vehicle-Counting-Traffic-Flow-Monitoring/
│
├── README.md
├── DATASET.md
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── capture_frame.py
│   ├── roi_selector.py
│   └── vehicle_counting.py
│
├── input_videos/
├── screenshots/
├── outputs/
│
├── models/
│   └── README.md
│
└── docs/
    └── IEEE_Project_Report.pdf
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Vehicle-Counting-Traffic-Flow-Monitoring.git

cd Vehicle-Counting-Traffic-Flow-Monitoring
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Install a CUDA-compatible PyTorch build appropriate for your system if GPU acceleration is required.

### 4. Add the input video

Place your traffic video inside:

```text
input_videos/traffic_video.mp4
```

The demonstration video is not included in this repository.

## Execution Steps

### Step 1: Extract a video frame

Run:

```bash
python src/capture_frame.py
```

This extracts the first frame from the input video and saves it as `traffic_frame.jpg` in the project root directory.

### Step 2: Select ROI coordinates

Run:

```bash
python src/roi_selector.py
```

Mouse and keyboard controls:

- Left-click: Select ROI boundary points.
- R: Reset selected points.
- C: Finish selection and print coordinates.
- Q: Quit.

Use the printed coordinates to configure the ROI polygons and virtual counting lines in `vehicle_counting.py`.

### Step 3: Run vehicle detection and tracking

Run:

```bash
python src/vehicle_counting.py
```

The main script:

- Loads YOLO11x.
- Uses ByteTrack for vehicle tracking.
- Detects vehicle classes.
- Maintains tracking IDs.
- Counts directional line crossings.
- Calculates current ROI occupancy.
- Estimates traffic conditions.
- Saves the annotated output video.

The current implementation requires an NVIDIA GPU with CUDA-enabled PyTorch.

## Model Information

**Final detector:** YOLO11x

**Tracking algorithm:** ByteTrack

The model weights `yolo11x.pt` are automatically downloaded by Ultralytics when required.

Refer to [models/README.md](models/README.md) for details.

## Sample Outputs

The following screenshots demonstrate the project implementation.

### 1. Vehicle Detection

![Vehicle Detection](screenshots/vehicle_detection.png)

### 2. ROI Selection

![ROI Selection](screenshots/roi_selection.png)

### 3. Vehicle Tracking and Counting

![Vehicle Counting](screenshots/vehicle_counting.png)

### 4. Traffic Monitoring Dashboard

![Traffic Dashboard](screenshots/traffic_dashboard.png)

*Replace these image placeholders with actual screenshots from the project.*

## Results and Limitations

The system generates:

- Cumulative left-side and right-side vehicle counts.
- Current vehicle occupancy in the left and right ROIs.
- Rule-based traffic-condition labels.
- Annotated output video.

The traffic-condition labels are heuristic estimates based on ROI density and apparent pixel movement. Pixel displacement should not be interpreted as physical vehicle speed.

The example detection metrics in the project report are illustrative and have not been verified against evaluation logs. Refer to the IEEE report for the evaluation qualifications and limitations.

## Challenges

- Vehicle occlusion and overlapping detections.
- Missed detections in crowded scenes.
- Tracking ID changes.
- GPU memory requirements.
- Camera-specific ROI and threshold calibration.

## Future Scope

- Improve tracking under heavy occlusion.
- Calibrate traffic-condition thresholds for different cameras.
- Estimate real-world vehicle speed using camera calibration.
- Improve detection and counting accuracy.
- Extend the system towards real-time traffic monitoring.

## Project Team

| Name | Role |
|---|---|
| Yash Joshi | Project Team Member |
| Vedant Wadibhasme | Project Team Member |
| Shantanu Bhaduri | Project Team Member |
| Shantanu Ramtekkar | Project Team Member |

**Project Guide:** Prof. Ketan Kurkute

**Department:** Electronics and Computer Science

**Institution:** Ramdeobaba University, Nagpur, India

## References

1. Ultralytics, YOLO11 Documentation.
2. Y. Zhang et al., "ByteTrack: Multi-object tracking by associating every detection box," ECCV, 2022.
3. L. Wen et al., "UA-DETRAC: A new benchmark and protocol for multi-object detection and tracking," 2015.
4. OpenCV Documentation.

Additional references are available in the IEEE project report.

## IEEE Project Report

The complete IEEE-format project report is available in the `docs/` directory.

## GitHub Repository

Repository: https://github.com/YOUR_USERNAME/Vehicle-Counting-Traffic-Flow-Monitoring

---

**Developed as an academic project at Ramdeobaba University, Nagpur.**