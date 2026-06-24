# Vision Prime

Real-time YOLO object detection GUI for industrial quality inspection, built with PyQt5 and optimized for NVIDIA Jetson (aarch64).

---

## Features

- **Live detection feed** — camera index (0/1/2) or video file input at 1280×720
- **YOLO framework support** — YOLOv11 via Ultralytics `torch.hub`
- **OK / NG classification** — any detection whose label contains `ng`, `defect`, `bad`, `fail`, `error`, or `nok` is treated as NG; all others are OK
- **Auto NG capture** — NG frames are automatically saved to `NG_Captures/YYYY-MM-DD/`
- **Daily system check** — live camera test with a reference sample before each run
- **Detection log** — timestamped per-frame table with CSV export
- **Session statistics** — real-time OK count, NG count, OK rate, NG rate, and FPS
- **Data collection** — review unlabeled images with model predictions and save pseudo-labels (YOLO `.txt`) for retraining
- **Manual capture** — save the current live frame and its labels to `collected_labels/` at any time

---

## Requirements

**Hardware:** NVIDIA Jetson (tested on Jetson with L4T / tegra kernel)

**Python:** 3.10

**System packages:**
```bash
sudo apt-get update && sudo apt-get install python3-pip
```

**Python packages:**
```bash
pip install PyQt5 opencv-python>=4.8 numpy>=1.24 ultralytics>=8.0
```

**PyTorch (Jetson aarch64)** — install from the bundled wheels in this repo:
```bash
pip install torch-2.11.0-cp310-cp310-linux_aarch64.whl
pip install torchvision-0.26.0-cp310-cp310-linux_aarch64.whl
pip install torchaudio-2.10.0-cp310-cp310-linux_aarch64.whl
```

> YOLOv11 is loaded via `torch.hub` — no separate install needed.

---

## Quick Start

### 1. Verify GPU detection (optional)
```bash
python test_yolo_webcam.py
```
Opens camera 0 and runs `yolo11n.pt`. Press **Q** to quit.

### 2. Launch the main application
```bash
python vision_prime.py
```

### 3. Workflow inside the app

| Step | Action |
|------|--------|
| 1 | Select YOLO framework (YOLOv11) |
| 2 | Browse and load a `.pt` model file |
| 3 | Run the **Daily Check** — place an OK reference sample in view and press Run Test |
| 4 | Select video source (Camera 0/1/2 or Video File) |
| 5 | Adjust confidence threshold (default 0.50) |
| 6 | Press **START** |

---

## Included Models

| File | Description |
|------|-------------|
| `yolo11n.pt` | YOLOv11 nano — general-purpose baseline |
| `SRG14_L ver1.pt` | Custom model — left-side inspection (ver 1) |
| `SRG14_R ver1.pt` | Custom model — right-side inspection (ver 1) |

---

## Output Structure

```
NG_Captures/
└── YYYY-MM-DD/
    └── NG_HH-MM-SS-ffffff.jpg   # auto-saved NG frames

collected_labels/
├── images/
│   └── *.jpg                    # captured frames
└── labels/
    └── *.txt                    # YOLO-format pseudo-labels
```

---

## Data Collection for Retraining

1. Open **Collect Labels** from the right panel (model must be loaded)
2. Set the images folder, existing labels folder (already-labeled files are skipped), and output folder
3. Press **Start Review** — images with detections are shown one by one
4. Press **Y** (or Save button) to keep an image with its auto-generated label, **N** (or Skip) to discard
5. Collected data lands in `collected_labels/images/` and `collected_labels/labels/`

---

## Detection Log

The log table records one row per second during a run:

| Column | Description |
|--------|-------------|
| # | Row index |
| Timestamp | Date and time |
| Model Name | Loaded model stem |
| OK / NG | Count for that second |
| OK % / NG % | Per-frame percentage |
| Status | OK or NG (color-coded) |

Use **Export CSV** to save the full log, or **Clear Log** to reset.

---

## Project Structure

```
VISION-PRIME/
├── vision_prime.py                          # Main GUI application
├── test_yolo_webcam.py                      # GPU/camera smoke test
├── requirements.txt                         # Python dependencies
├── yolo11n.pt                               # YOLOv11 nano model
├── SRG14_L ver1.pt                          # Custom left model
├── SRG14_R ver1.pt                          # Custom right model
├── torch-2.11.0-cp310-cp310-linux_aarch64.whl
├── torchvision-0.26.0-cp310-cp310-linux_aarch64.whl
└── torchaudio-2.10.0-cp310-cp310-linux_aarch64.whl
```
