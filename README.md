# Shot-Classification-System
This is a repository of shot classification using Python-CV
<div align="center">

# 🎾Shot Classification System
### Shot Classification System 

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple?style=for-the-badge)
![OpenCV](https://img.shields.io/badge/OpenCV-4.10-green?style=for-the-badge&logo=opencv)
![CUDA](https://img.shields.io/badge/CUDA-12.8-76B900?style=for-the-badge&logo=nvidia)


A real-time computer vision pipeline that analyzes tennis match footage from a fixed overhead CCTV camera — detecting players, tracking the ball, and classifying shot types frame by frame.

[Overview](#overview) • [Demo](#demo) • [Features](#features) • [Setup](#setup) • [Usage](#usage) • [How It Works](#how-it-works) • [Output](#output)

</div>

---

## Overview

This project is a prototype of a ** Shot Classification System**. Given an overhead CCTV video, the system:
- Detects and tracks all 4 players with stable IDs.
- Detects the ball using HSV color filtering and circularity heuristics.
- Classifies shots as **Forehand, Backhand, Volley, Smash, or Serve**.
- Generates annotated video, JSON/CSV logs, and analytics charts.

---

## 🧠 Approach & Methodology

The system utilizes a modular pipeline to transform raw pixels into structured sports data:

### 1. Spatial Filtering (Geofencing)
To eliminate background noise (spectators, vehicles), we define a trapezoidal `COURT_POLYGON`. Only detections falling within this playable area are processed, significantly reducing false positives.

### 2. Stable Player Tracking
Standard YOLO IDs often swap during player crossovers. 
- **The Solution:** We implement position-based remapping. Every frame, detections are sorted by Y and X coordinates relative to the net. 
- **The Result:** P1/P2 always represent far-side players; P3/P4 represent near-side players, ensuring tracking stability.

### 3. Ball Detection (HSV + Circularity)
The ball appears as a tiny ~5px cluster from an overhead view.
- **Method:** We use **HSV Color Masking** to isolate fluorescent yellow.
- **Filtering:** Candidates are scored based on **Circularity** and **Area** to distinguish the ball from court reflections or player clothing.

### 4. Shot Classification Logic
Classification is handled via **Vector Trajectory Analysis**:
- **Hit Detection:** Triggered when the cosine similarity between the ball's incoming and outgoing velocity vectors drops significantly (indicating a sharp change in direction).
- **Attribution:** The system assigns the shot to the player closest to the impact point.
- **Type Logic:** Shots are classified based on the player’s court zone and ball trajectory (e.g., high-speed downward trajectories near the net are labeled **Smashes**).

---

## 🚧 Challenges Faced

* **Tiny Target Tracking:** The ball's small size (~0.5% of frame height) makes it easy to lose against white court lines or net textures.
* **Perspective Distortion:** Near-camera players appear much larger than far-side players, requiring dynamic distance thresholds for player-ball attribution.
* **ID Instability:** YOLO track IDs jump when players are occluded; solved via spatial re-sorting to maintain consistent P1–P4 labels.

---

## 📈 Future Improvements

* **Homography Mapping:** Transform pixel coordinates into a "Bird's Eye View" to calculate real-world speeds in km/h.
* **YOLOv8-Pose:** Integrate skeleton keypoints to detect racket arm extension for 99% accurate Forehand vs. Backhand differentiation.
* **LSTM Models:** Replace rule-based logic with a Deep Learning sequence model to recognize complex patterns like "Lobs" or "Slices."

---

## Setup & Usage

### 1. Requirements
- Python 3.10+ | CUDA 12.8+ | NVIDIA GPU (RTX 5070 recommended)

### 2. Installation
```bash
git clone [https://github.com/prabhatbhusal/Shot-Classification-System.git]
cd Shot-Classification-System
python -m venv venv
# Activate venv and install dependencies
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu128](https://download.pytorch.org/whl/cu128)
pip install -r requirements.txt
```
</div>

