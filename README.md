# Shot-Classification-System
This is a repository of shot classification using Python-CV\
<div align="center">

# 🎾Shot Classification System
### Shot Classification System — Layman AI Internship Assignment

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple?style=for-the-badge)
![OpenCV](https://img.shields.io/badge/OpenCV-4.10-green?style=for-the-badge&logo=opencv)
![CUDA](https://img.shields.io/badge/CUDA-12.8-76B900?style=for-the-badge&logo=nvidia)


A real-time computer vision pipeline that analyzes padel match footage from a fixed overhead CCTV camera — detecting players, tracking the ball, and classifying shot types frame by frame.

[Overview](#overview) • [Demo](#demo) • [Features](#features) • [Setup](#setup) • [Usage](#usage) • [How It Works](#how-it-works) • [Output](#output)

</div>

---

## Overview

This project is a working prototype of a **Padel Game Analytics System** built using Computer Vision and Machine Learning techniques. Given an overhead padel match video, the system:

- Detects and tracks all 4 players with stable IDs across the entire video
- Detects the ball using HSV color filtering
- Classifies each shot as **Forehand, Backhand, Volley, Smash, or Serve**
- Outputs an annotated video, JSON/CSV shot log, and analytics charts

> Built as part of the Layman AI AI/ML Internship Assignment.

---

## Demo

| Input Frame | Annotated Output |
|---|---|
| Raw overhead CCTV footage | Player boxes, ball trail, shot labels, live shot log |

**What you see in the output video:**
- 🟢 **P1** — Neon green box (top-left player)
- 🔵 **P2** — Sky blue box (top-right player)
- 🟠 **P3** — Orange box (bottom-left player)
- 🟣 **P4** — Purple box (bottom-right player)
- 🟡 Ball trail fading cyan with speed arrow
- Shot badge with pulse ring at point of impact
- Live **RECENT SHOTS** log on the right side

---

## Features

### Core (Mandatory)
- [x] Player detection and tracking with stable IDs (1–4)
- [x] Ball detection via HSV color masking
- [x] Shot classification — Forehand, Backhand, Volley, Smash, Serve
- [x] Structured JSON + CSV output per shot

### Bonus
- [x] Shot count analytics per player and shot type
- [x] Annotated output video with overlays
- [x] Analytics bar chart (shot types, player distribution, timeline)
- [x] Court boundary polygon filter (removes spectator false detections)
- [x] Ball trajectory trail with speed arrow
- [x] Live shot log panel on video

---

## Shot Definitions

| Shot | Definition | How Detected |
|---|---|---|
| **Forehand** | Natural attacking swing on the dominant-hand side | Ball arrives from player's left; player in mid/baseline zone |
| **Backhand** | Non-dominant side stroke, arm crosses body | Ball arrives from player's right; player in mid/baseline zone |
| **Volley** | Intercepts ball before it bounces, near the net | Player center within 18% of net line |
| **Smash** | Overhead/side-overhead aggressive strike at net (Bandeja/Vibora) | Player within 8% of net + outgoing speed > 55px/frame + ball approaching |
| **Serve** | Underarm opening shot from behind service line | Only one player on serving side + ball nearly stationary before hit |

---

## Project Structure

```
Shot-Classification-System/
├── input/
│   └── input_sample_video.mp4     ← place your video here
├── output/
│   ├── output_annotated.mp4       ← annotated video result
│   ├── shots.json                 ← all shots in JSON format
│   ├── shots.csv                  ← all shots in CSV format
│   └── shot_chart.png             ← analytics charts
├── models/
│   └── yolov8x.pt                 ← auto-downloaded on first run
├── src/
│   ├── __init__.py
│   ├── config.py                  ← court polygon + shared constants
│   ├── detect_track.py            ← YOLO player detection + ball tracker
│   ├── shot_classifier.py         ← rule-based shot classification
│   ├── analytics.py               ← summary stats + chart generation
│   └── visualizer.py              ← video annotation drawing
├── main.py                        ← entry point — run this
├── requirements.txt
└── README.md
```

---

## Setup

### Requirements
- Python 3.10+
- CUDA 12.8+ (for GPU acceleration)
- Windows / Linux

### 1. Clone the Repository
```bash
git clone https://github.com/prabhatbhusal/Shot-Classification-System.git
cd Shot-Classification-System
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / Mac
source venv/bin/activate
```

### 3. Install PyTorch with CUDA 12.8
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### 4. Install Remaining Dependencies
```bash
pip install -r requirements.txt
```

### 5. Verify GPU is Detected
```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```
Expected output:
```
True
NVIDIA GeForce RTX 5070
```

---

## Usage

### 1. Place your video
Copy your padel match video into the `input/` folder and rename it `input_sample_video.mp4`, or update the path in `main.py`:
```python
INPUT_VIDEO = "input/your_video_name.mp4"
```

### 2. Run the pipeline
```bash
python main.py
```

### 3. Check results in `output/`
```
output/
├── output_annotated.mp4   ← watch this
├── shots.json             ← submit this
├── shots.csv              ← open in Excel
└── shot_chart.png         ← analytics overview
```

### Optional: Watch live while processing
In `main.py`, set:
```python
SHOW_LIVE = True
```
Press `Q` to stop early.

---

## How It Works

### Architecture

```
Video Input (MP4)
      │
      ▼
┌─────────────────┐
│  Frame Reader   │  OpenCV reads frame by frame at original FPS
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌────────┐  ┌──────────┐
│  YOLO  │  │  HSV     │
│  v8x   │  │  Mask    │
│ Player │  │  Ball    │
│ Track  │  │ Detect   │
└───┬────┘  └────┬─────┘
    │             │
    ▼             ▼
┌──────────────────────┐
│   Shot Classifier    │
│  (Rule-based logic)  │
│                      │
│ 1. Direction change? │
│ 2. Who is nearest?   │
│ 3. Court zone?       │
│ 4. Ball direction?   │
└──────────┬───────────┘
           │
    ┌──────┴──────┐
    │             │
    ▼             ▼
┌────────┐   ┌─────────┐
│ JSON/  │   │ Video   │
│  CSV   │   │ Overlay │
│ Output │   │  Draw   │
└────────┘   └─────────┘
```

### Player Tracking
YOLOv8x detects persons (COCO class 0) with `conf=0.20`. Raw YOLO track IDs are unstable (can jump from 1 → 196 → 24), so we remap them to stable IDs 1–4 based on court position — sorted by screen Y then X so:
- **P1** = top-left, **P2** = top-right
- **P3** = bottom-left, **P4** = bottom-right

A **court boundary polygon** filters out spectators and people outside the playable area.

### Ball Detection
Since the camera is overhead and the ball is very small (~5px radius), a trained model is overkill. Instead we use **HSV color masking** for yellow/white circular objects on the blue court, combined with circularity scoring to find the best candidate contour each frame. Ball positions outside the court polygon are rejected.

### Shot Classification
Because the camera angle is top-down, traditional pose estimation (arm angles) doesn't work. Instead, classification uses **ball trajectory analysis**:

1. **Hit detection** — ball trajectory is split in half; if the cosine similarity between the two velocity vectors is < 0.35 (angle > ~69°), a hit occurred
2. **Player attribution** — nearest player within 30% of frame width
3. **Shot type** — rule-based priority: Smash → Volley → Serve → Forehand/Backhand

---

## Output Format

### JSON (`shots.json`)
```json
[
  {
    "frame": 312,
    "timestamp": 12.48,
    "player_id": 3,
    "shot_type": "forehand",
    "ball_x": 724,
    "ball_y": 511,
    "ball_speed": 22.4,
    "confidence": 0.81,
    "description": "P3 plays a forehand drive (22.4px/f)"
  }
]
```

### CSV (`shots.csv`)
| frame | timestamp | player_id | shot_type | ball_x | ball_y | ball_speed | confidence | description |
|---|---|---|---|---|---|---|---|---|
| 312 | 12.48 | 3 | forehand | 724 | 511 | 22.4 | 0.81 | P3 plays a forehand drive |

---

## Tech Stack

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.11 | Core language |
| YOLOv8x | Ultralytics 8.3+ | Player detection + tracking |
| OpenCV | 4.10+ | Video I/O, color masking, annotation |
| NumPy | 2.0+ | Array math, HSV masking |
| Pandas | 2.2+ | CSV output + analytics |
| Matplotlib | 3.9+ | Analytics charts |
| PyTorch | 2.6+ | YOLO inference backend |
| CUDA | 12.8 | GPU acceleration (RTX series) |

---

## Challenges

- **Ball is tiny** (~5px radius at 1080p overhead) — HSV masking is sensitive to lighting changes and court reflections
- **Near-side players appear very large** and partially cut off at frame edge — required lowering YOLO confidence threshold and using feet position for court polygon test
- **YOLO track ID instability** — IDs jump every time a player is briefly occluded; solved with position-based stable remapping
- **Forehand/Backhand ambiguity** from top-down view — without skeleton data, horizontal ball arrival direction is the best available signal
- **Court polygon calibration** — the trapezoid court shape requires manual corner coordinate tuning per camera setup

---

## Improvements

Given more time or resources:

- Train a **custom YOLOv8 model** on padel-specific ball data for reliable ball detection
- Use **optical flow** (Farneback) as a secondary ball detection signal
- Implement **homography transform** to map pixel positions to real court coordinates (meters)
- Add a **lightweight LSTM** to learn temporal shot patterns from trajectory sequences
- Use **YOLOv8-pose** for skeleton keypoints when camera angle is side-on
- Add **rally detection** — grouping shots into points and games

---


<div align="center">
Built for the <strong>Layman AI</strong> AI/ML Internship Assignment
</div>
