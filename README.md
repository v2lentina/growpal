# 🌱 PiCrawler Weed Detector

Small computer-vision prototype built during a robotics hackathon.  
The script uses a YOLO model to detect weeds for a garden robot and applies a few heuristic filters to reduce false positives (e.g. purple flowers).

This was an experimental project and not intended as production-ready code.

## What it does
- Detects **weed vs crop** using a trained YOLO model
- Filters out likely **flower detections** using color (HSV) and size checks
- Supports single-image tests, real-time camera mode, and continuous scanning

## Requirements
- Python 3
- `ultralytics`
- `opencv-python`
- `numpy`
- `vilib` (for Raspberry Pi camera, OpenCV fallback included)

## Usage

```bash
python3 vision.py                 # interactive scan
python3 vision.py image.jpg       # analyze image file
python3 vision.py realtime        # live camera detection
python3 vision.py continuous      # headless scan mode
