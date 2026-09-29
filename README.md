# 🔥 Fire Detection System

An end-to-end **AI-powered fire and smoke detection system** built with **YOLO object detection models**, PyTorch, and OpenCV.

The system is designed to detect **fire and smoke in images and video streams**, with a modular architecture covering data preparation, model training, evaluation, inference, and deployment.

It can serve as a foundation for **real-time surveillance, early-warning systems, industrial safety, smart monitoring, and computer vision applications**.

> **Note:** This project is intended for research, educational, and prototyping purposes. It should not be considered a certified fire-safety system without appropriate validation, testing, and hardware-level safety mechanisms.

---

# 📖 Overview

Fire and smoke detection is an important computer vision problem where the objective is to identify hazardous events as early as possible from visual data.

This project approaches the problem as an **object detection task**, allowing the model to identify not only whether fire or smoke exists, but also **where the detected object appears within the image**.

The system uses YOLO-based models to perform fast inference while maintaining a practical balance between detection accuracy and computational efficiency.

### Core capabilities

* 🔥 Fire detection
* 💨 Smoke detection
* 📦 Bounding-box localization
* 🎥 Video inference
* 🖼️ Image inference
* ⚡ Real-time detection
* 🧠 YOLO-based deep learning
* 📊 Model evaluation
* 🏋️ Model training
* 🐳 Deployment-ready architecture

---

# 🧠 How It Works

The complete pipeline can be summarized as:

```text
                    ┌──────────────────┐
                    │ Image / Video    │
                    │ Camera Stream     │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Frame Extraction │
                    │ & Preprocessing  │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │   YOLO Model     │
                    │    Inference     │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Object Detection │
                    └────────┬─────────┘
                             │
                    ┌────────┴─────────┐
                    │                  │
                    ▼                  ▼
              ┌───────────┐      ┌───────────┐
              │   Fire    │      │   Smoke   │
              └─────┬─────┘      └─────┬─────┘
                    │                  │
                    └────────┬─────────┘
                             ▼
                    ┌──────────────────┐
                    │ Bounding Boxes   │
                    │ + Confidence     │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Alert / Output   │
                    └──────────────────┘
```

---

# ✨ Features

## 🔥 Fire & Smoke Detection

The system is capable of detecting fire and smoke using trained YOLO object detection models.

For each detection, the model can provide:

* Detected class
* Bounding box coordinates
* Confidence score
* Detection location
* Annotated output frame

---

## 🎯 Object Localization

Unlike image classification, the system does not simply determine whether an image contains fire.

Instead, YOLO predicts **where the fire or smoke is located**.

```text
Input Image
     ↓
YOLO
     ↓
┌─────────────────────────────┐
│ Class: Fire                 │
│ Confidence: 0.94            │
│ Bounding Box: x1,y1,x2,y2   │
└─────────────────────────────┘
```

This makes the system suitable for visual monitoring and surveillance applications.

---

## ⚡ Real-Time Inference

YOLO is designed for fast object detection, making it suitable for applications where low inference latency is important.

The system can process:

* Static images
* Recorded videos
* Video streams
* Potentially live camera feeds

depending on the configured inference pipeline.

---

# 🤖 Models

The repository contains multiple YOLO model variants for experimentation and deployment.

| Model        | Variant | Primary Purpose                |
| ------------ | ------- | ------------------------------ |
| `yolo11s.pt` | Small   | Higher-capacity inference      |
| `yolo26n.pt` | Nano    | Lightweight / faster inference |

The models can be selected depending on the available hardware and the desired balance between computational cost and detection performance.

### Example

```bash
python src/detect.py \
    --source path/to/video.mp4 \
    --weights yolo11s.pt
```

---

# 🏗️ System Architecture

The project follows a modular structure that separates configuration, data, source code, training artifacts, and deployment.

```text
                         Fire Detection System
                                  │
             ┌────────────────────┼────────────────────┐
             │                    │                    │
             ▼                    ▼                    ▼
          Data Layer         Model Layer         Configuration
             │                    │                    │
             ▼                    ▼                    ▼
          Dataset          YOLO Weights          YAML / ENV
             │                    │                    │
             └────────────────────┼────────────────────┘
                                  │
                                  ▼
                           Training Pipeline
                                  │
                                  ▼
                           Evaluation / Runs
                                  │
                                  ▼
                           Inference Pipeline
                                  │
                                  ▼
                              Detection
                                  │
                                  ▼
                             Deployment
```

---

# 🧩 Project Components

## `src/`

Contains the main application and machine learning logic.

Typical responsibilities include:

* Model loading
* Training
* Inference
* Image/video processing
* Detection utilities
* Visualization
* Post-processing

---

## `config/`

Contains configuration files used by the system.

Possible configurations include:

* Dataset configuration
* Model parameters
* Training configuration
* Environment configuration
* Inference settings

Example:

```yaml
model: yolo11s.pt
confidence: 0.5
iou: 0.45
image_size: 640
```

---

## `data/`

Contains dataset-related resources.

Depending on the project version, this directory may contain:

```text
data/
├── images/
├── labels/
├── train/
├── val/
└── test/
```

For YOLO object detection datasets, annotations typically follow the YOLO label format:

```text
<class_id> <x_center> <y_center> <width> <height>
```

Coordinates are normalized relative to the image dimensions.

---

## `runs/`

Stores experiment and training artifacts generated during model development.

Typical contents include:

```text
runs/
├── train/
│   ├── weights/
│   ├── results.csv
│   ├── confusion_matrix.png
│   ├── results.png
│   └── ...
│
└── detect/
    └── ...
```

These artifacts can be used to analyze:

* Training progress
* Validation performance
* Loss curves
* Precision
* Recall
* mAP
* Confusion matrices

---

# 📂 Project Structure

```text
fire_detection_system/
│
├── .vscode/
│   └── VS Code workspace configuration
│
├── config/
│   ├── *.yaml
│   └── *.env
│
├── data/
│   ├── images/
│   ├── labels/
│   └── dataset configuration
│
├── deployment/
│   ├── Docker configuration
│   ├── deployment scripts
│   └── production configuration
│
├── runs/
│   ├── training experiments
│   └── detection outputs
│
├── src/
│   ├── detection
│   ├── training
│   ├── inference
│   └── utilities
│
├── check_env.py
│
├── yolo11s.pt
├── yolo26n.pt
│
├── requirements.txt
│
├── Dockerfile
│
└── README.md
```

---

# 🛠️ Tech Stack

| Category               | Technology             |
| ---------------------- | ---------------------- |
| Language               | Python 3.11+           |
| Object Detection       | Ultralytics YOLO       |
| Deep Learning          | PyTorch                |
| Computer Vision        | OpenCV                 |
| Configuration          | YAML                   |
| Environment Management | python-dotenv          |
| Containerization       | Docker                 |
| Deployment             | Docker / Shell Scripts |
| Experiment Tracking    | YOLO Training Runs     |

---

# 🚀 Installation & Setup

## 1. Clone the Repository

```bash
git clone https://github.com/salahmhmod7/fire_detection_system.git

cd fire_detection_system
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv .venv

.venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv .venv

source .venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available:

```bash
pip install ultralytics opencv-python python-dotenv
```

---

# 🔍 Environment Verification

Before running the application, verify that the required dependencies and environment are correctly configured:

```bash
python check_env.py
```

The environment verification script can be used to detect missing dependencies or configuration problems before starting inference or training.

---

# 🎥 Running Detection

## Image Detection

Example:

```bash
python src/detect.py \
    --source path/to/image.jpg \
    --weights yolo11s.pt
```

---

## Video Detection

```bash
python src/detect.py \
    --source path/to/video.mp4 \
    --weights yolo11s.pt
```

---

## Using the Lightweight Model

For environments where computational resources are limited:

```bash
python src/detect.py \
    --source path/to/video.mp4 \
    --weights yolo26n.pt
```

---

# 📊 Detection Output

A typical detection result contains:

```text
Class: Fire
Confidence: 0.94
Bounding Box:
    x1: 152
    y1: 87
    x2: 481
    y2: 392
```

The output can also be visualized directly on the original frame:

```text
┌───────────────────────────────────────┐
│                                       │
│       ┌─────────────────────┐         │
│       │       FIRE          │         │
│       │     94%             │         │
│       └─────────────────────┘         │
│                                       │
│                  ┌──────────┐         │
│                  │  SMOKE   │         │
│                  │   87%    │         │
│                  └──────────┘         │
│                                       │
└───────────────────────────────────────┘
```

---

# 🏋️ Model Training

The project structure supports YOLO model training and experimentation.

A typical training workflow consists of:

```text
Dataset
   ↓
Dataset Validation
   ↓
Data Configuration
   ↓
YOLO Model Initialization
   ↓
Training
   ↓
Validation
   ↓
Evaluation
   ↓
Best Model Selection
   ↓
Inference Testing
   ↓
Deployment
```

A typical Ultralytics training command looks like:

```bash
yolo detect train \
    model=yolo11s.pt \
    data=config/data.yaml \
    epochs=100 \
    imgsz=640
```

Training parameters can be adjusted according to the dataset, hardware, and experimental setup.

---

# 📈 Model Evaluation

Object detection performance can be evaluated using standard metrics such as:

### Precision

Measures how many predicted detections are actually correct.

```text
Precision = TP / (TP + FP)
```

### Recall

Measures how many relevant objects were successfully detected.

```text
Recall = TP / (TP + FN)
```

### mAP

Mean Average Precision is commonly used to evaluate object detection performance across classes and IoU thresholds.

The training artifacts inside `runs/` can be used to inspect these metrics and analyze model behavior.

---

# 🔬 Error Analysis

Model evaluation should not rely only on a single metric.

Important cases to inspect include:

* False positives
* False negatives
* Small fire regions
* Heavy smoke
* Low-light environments
* Dense smoke
* Background objects resembling fire
* Reflections and bright lights
* Partial occlusion
* Long-distance fire

Example analysis workflow:

```text
Prediction
    ↓
Correct Detection?
    │
 ┌──┴───┐
 │      │
Yes     No
 │      │
 ▼      ▼
TP    Error Analysis
        │
        ├── False Positive
        ├── False Negative
        ├── Localization Error
        └── Classification Error
```

---

# 🐳 Docker Deployment

The project includes Docker support for reproducible deployment.

## Build the Image

```bash
docker build -t fire-detection-system .
```

## Run the Container

```bash
docker run fire-detection-system
```

For deployment-specific configuration, refer to the files inside:

```text
deployment/
```

---

# ⚙️ Production Considerations

For a real-world deployment, several additional components may be required.

### Camera Integration

```text
IP Camera / CCTV
       ↓
Video Stream
       ↓
Frame Processing
       ↓
YOLO
       ↓
Fire / Smoke Detection
```

### Alert System

A production system could integrate:

* Email alerts
* SMS notifications
* Webhooks
* Mobile notifications
* Monitoring dashboards
* Alarm systems

### Monitoring

Production monitoring could track:

* Inference latency
* FPS
* Detection frequency
* Confidence distributions
* False-positive rate
* Hardware utilization
* Model performance over time

---

# 🧪 Testing

Recommended testing areas include:

### Input Testing

* Valid images
* Corrupted images
* Different resolutions
* Different video formats
* Empty video streams

### Model Testing

* Fire detection
* Smoke detection
* Multiple objects
* Small objects
* Occluded objects
* Low-light scenes

### Performance Testing

* CPU inference
* GPU inference
* Different image sizes
* Different YOLO model variants

---

# 🔐 Configuration & Environment Variables

Sensitive configuration values should be stored outside the source code.

Example:

```env
MODEL_PATH=yolo11s.pt
CONFIDENCE_THRESHOLD=0.5
IOU_THRESHOLD=0.45
IMAGE_SIZE=640
```

Do not commit sensitive `.env` files or credentials to GitHub.

Add them to `.gitignore`:

```gitignore
.env
.venv/
__pycache__/
*.pyc
```

---

# 📸 Demo

Add screenshots or GIFs demonstrating the detector:

```text
demo/
├── fire_detection.gif
├── image_detection.png
└── video_detection.png
```

Example:

```markdown
![Fire Detection Demo](demo/fire_detection.gif)
```

A visual demo is especially useful for object detection projects because it shows the model's bounding boxes and confidence scores directly.

---

# 🔮 Future Improvements

Potential extensions include:

* [ ] Real-time webcam detection.
* [ ] RTSP/IP camera support.
* [ ] Multi-camera monitoring.
* [ ] Real-time alert system.
* [ ] Email/SMS notifications.
* [ ] WebSocket-based live detection.
* [ ] FastAPI inference service.
* [ ] Streamlit monitoring dashboard.
* [ ] Model quantization.
* [ ] ONNX export.
* [ ] TensorRT optimization.
* [ ] GPU acceleration.
* [ ] Model benchmarking.
* [ ] Automated testing.
* [ ] CI/CD pipeline.
* [ ] Experiment tracking.
* [ ] Model monitoring.
* [ ] False-positive reduction.
* [ ] Hard-negative mining.
* [ ] Explainable AI visualizations.

---

# ⚠️ Safety Notice

This project is a **computer vision prototype** and should not be treated as a certified fire detection or life-safety system.

Real-world deployment requires rigorous validation under representative environmental conditions, hardware redundancy, reliable alert mechanisms, and compliance with applicable fire-safety standards and regulations.

---

# 👨‍💻 Author

**Salah Mahmoud**

* GitHub: [@salahmhmod7](https://github.com/salahmhmod7)
* Email: [flaysefeco.1@gmail.com](mailto:flaysefeco.1@gmail.com)

---

# ⭐ Support

If you find this project useful or interesting, consider giving the repository a ⭐ on GitHub!
