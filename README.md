# Fire & Smoke Detection — Desktop GUI

A Tkinter desktop application for real-time fire and smoke detection
using OpenCV, with an optional YOLOv8 deep-learning layer.

## Quick start

```bash
pip install -r requirements.txt
python gui.py
```

## Features
- Live camera feed with annotated bounding boxes
- Start / stop detection with camera selector
- Real-time stat cards: fire pixels, smoke pixels, confidence
- Alert history log (SQLite) — persists across sessions
- Snapshot saved to /snapshots/ on every confirmed alert
- Optional YOLO model: place fire_smoke_yolov8n.pt in /models/

## Project layout
```
fire_detection_gui/
├── gui.py              ← entry point (run this)
├── requirements.txt
├── detections.db       ← auto-created on first run
├── snapshots/          ← auto-created, stores alert images
└── models/
    └── fire_smoke_yolov8n.pt   ← optional YOLO weights
```

## Tuning knobs (inside gui.py → DetectionEngine)
| Constant              | Default | Effect                                  |
|-----------------------|---------|-----------------------------------------|
| REQUIRED_CONSECUTIVE  | 4       | Frames before alert fires — raise to reduce false alarms |
| ALERT_COOLDOWN        | 30s     | Min seconds between DB writes           |
| fire_px threshold     | 2000    | Pixel count needed to flag fire         |
| smoke_px threshold    | 4000    | Pixel count needed to flag smoke        |
| YOLO conf threshold   | 0.5     | YOLO confidence cutoff                  |

## YOLO model sources
- https://universe.roboflow.com  (search "fire smoke detection")
- https://huggingface.co         (search "fire-yolov8")
Download the .pt file and place it at models/fire_smoke_yolov8n.pt
