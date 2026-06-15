#!/usr/bin/env python3
"""Vision Prime v1.0 — YOLO Detection Software"""

import sys
import time
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
import torch

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QFileDialog,
    QTableWidget, QTableWidgetItem, QHeaderView, QDialog,
    QGroupBox, QFrame, QSizePolicy, QButtonGroup,
    QMessageBox, QStatusBar, QLineEdit,
    QAbstractItemView, QDoubleSpinBox,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QMutex
from PyQt5.QtGui import QImage, QPixmap, QColor, QPalette

# ─── Config ───────────────────────────────────────────────────
APP_NAME        = "Vision Prime"
COLLECT_OUTPUT  = "collected_labels"   # root folder for pseudo-label output
IMG_EXTS        = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
VERSION  = "1.0.0"
NG_BASE  = "NG_Captures"

YOLO_TYPES = {
    "YOLOv8+": "ultralytics",  # YOLOv8/v9/v10/v11
    "YOLOv5":  "yolov5",       # torch.hub
}

# Keywords in a class label that mark a detection as NG
NG_KEYWORDS = ("ng", "defect", "bad", "fail", "error", "nok")

DAILY_CHECKS = [
    "Camera / video source is connected and responsive",
    "Lighting conditions meet inspection requirements",
    "YOLO model file has been selected and verified",
    "Available storage space is ≥ 10 GB",
    "Work area is clear and objects are positioned correctly",
    "Detection confidence threshold has been configured",
    "NG capture folder path has been confirmed",
    "Previous session data has been saved or cleared",
]

# ─── Colour palette ───────────────────────────────────────────
C = {
    "bg":      "#f0f2f5",
    "card":    "#ffffff",
    "border":  "#e2e8f0",
    "txt1":    "#1a202c",
    "txt2":    "#64748b",
    "txt3":    "#94a3b8",
    "blue":    "#3b82f6",
    "blue_d":  "#2563eb",
    "blue_bg": "#eff6ff",
    "green":   "#10b981",
    "red":     "#ef4444",
    "amber":   "#f59e0b",
    "purple":  "#7c3aed",
    "row_alt": "#f8fafc",
}

# ─── Global stylesheet ────────────────────────────────────────
QSS = f"""
QMainWindow, QWidget#central {{ background: {C['bg']}; }}
QGroupBox {{
    background: {C['card']}; border: 1px solid {C['border']};
    border-radius: 8px; margin-top: 22px; padding: 2px 8px 8px 8px;
    font-size: 11px; font-weight: bold; color: {C['txt2']};
}}
QGroupBox::title {{
    subcontrol-origin: margin; left: 12px; top: 0px; padding: 2px 6px;
    color: {C['blue']}; font-size: 11px; background: {C['card']}; border-radius: 3px;
}}
QLabel {{ color: {C['txt1']}; font-size: 12px; background: transparent; }}
QComboBox {{
    background: {C['card']}; color: {C['txt1']}; border: 1px solid {C['border']};
    border-radius: 5px; padding: 5px 10px; font-size: 12px; min-height: 28px;
}}
QComboBox:hover {{ border-color: {C['blue']}; }}
QComboBox::drop-down {{ border: none; width: 20px; }}
QComboBox QAbstractItemView {{
    background: {C['card']}; color: {C['txt1']}; border: 1px solid {C['border']};
    selection-background-color: {C['blue_bg']}; selection-color: {C['blue_d']}; outline: none;
}}
QPushButton {{
    background: {C['card']}; color: {C['txt1']}; border: 1px solid {C['border']};
    border-radius: 5px; padding: 6px 14px; font-size: 12px;
    font-weight: bold; min-height: 28px;
}}
QPushButton:hover   {{ background: {C['blue_bg']}; border-color: {C['blue']}; color: {C['blue']}; }}
QPushButton:pressed {{ background: #dbeafe; }}
QPushButton:disabled {{ background: {C['row_alt']}; color: {C['txt3']}; border-color: {C['border']}; }}
QPushButton#seg_l {{ border-radius: 5px 0 0 5px; border-right: none; }}
QPushButton#seg_r {{ border-radius: 0 5px 5px 0; }}
QPushButton#seg_l:checked, QPushButton#seg_r:checked {{
    background: {C['blue']}; color: #ffffff; border-color: {C['blue_d']};
}}
QPushButton#seg_l:checked {{ border-right: none; }}
QLineEdit, QDoubleSpinBox {{
    background: {C['card']}; color: {C['txt1']}; border: 1px solid {C['border']};
    border-radius: 5px; padding: 5px 8px; font-size: 12px; min-height: 28px;
}}
QLineEdit:focus, QDoubleSpinBox:focus {{ border-color: {C['blue']}; }}
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
    background: {C['row_alt']}; border: none; border-radius: 3px; width: 16px;
}}
QTableWidget {{
    background: {C['card']}; color: {C['txt1']}; gridline-color: {C['border']};
    border: 1px solid {C['border']}; border-radius: 6px; font-size: 11px;
    alternate-background-color: {C['row_alt']};
    selection-background-color: {C['blue_bg']}; selection-color: {C['blue_d']}; outline: none;
}}
QTableWidget::item {{ padding: 0 4px; border: none; }}
QHeaderView::section {{
    background: {C['row_alt']}; color: {C['txt2']}; border: none;
    border-right: 1px solid {C['border']}; border-bottom: 1px solid {C['border']};
    padding: 7px 6px; font-size: 11px; font-weight: bold;
}}
QScrollBar:vertical   {{ background: {C['row_alt']}; width: 8px; border-radius: 4px; }}
QScrollBar:horizontal {{ background: {C['row_alt']}; height: 8px; border-radius: 4px; }}
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{
    background: #cbd5e1; border-radius: 4px; min-height: 24px;
}}
QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {{
    background: {C['blue']};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ height: 0; width: 0; }}
QStatusBar {{
    background: {C['card']}; color: {C['txt2']}; font-size: 11px;
    border-top: 1px solid {C['border']};
}}
QProgressBar {{
    background: {C['border']}; border: none; border-radius: 4px;
}}
QProgressBar::chunk {{ background: {C['blue']}; border-radius: 4px; }}
"""


# ─── Shared helper: solid-colour action button ────────────────
def _mk_btn(text: str, color: str, hover: str,
            height: int = 42, fsize: int = 14) -> QPushButton:
    btn = QPushButton(text)
    btn.setFixedHeight(height)
    btn.setStyleSheet(f"""
        QPushButton {{ background: {color}; color: #fff; border: none;
                      border-radius: 5px; font-size: {fsize}px; font-weight: bold; }}
        QPushButton:hover    {{ background: {hover}; }}
        QPushButton:disabled {{ background: {C['border']}; color: {C['txt3']}; }}
    """)
    return btn


# ─── Camera Thread ────────────────────────────────────────────
class CameraThread(QThread):
    frame_ready    = pyqtSignal(np.ndarray)
    error_occurred = pyqtSignal(str)

    def __init__(self, source):
        super().__init__()
        self.source  = source
        self._active = False

    def run(self):
        self._active = True
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self.error_occurred.emit(f"Cannot open: {self.source}")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
        while self._active:
            ret, frame = cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                self.frame_ready.emit(frame.copy())
            else:
                break
            self.msleep(33)
        cap.release()

    def stop(self):
        self._active = False
        self.wait(3000)


# ─── Detection Worker ─────────────────────────────────────────
class DetectionWorker(QThread):
    result_ready   = pyqtSignal(np.ndarray, int, int, float, float)
    error_occurred = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.model            = None
        self.yolo_type        = "ultralytics"
        self.model_name       = ""
        self.device           = "cuda" if torch.cuda.is_available() else "cpu"
        self.confidence       = 0.50
        self.last_raw_frame   = None          # raw frame from most recent inference
        self.last_detections  = []            # list of (x1,y1,x2,y2,cls,conf)
        self._queue           = []
        self._mutex           = QMutex()
        self._active          = False
        print(f"Running on: {'CUDA - ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    def load_model(self, path: str, yolo_type: str) -> bool:
        try:
            if yolo_type == "yolov5":
                import torch
                self.model = torch.hub.load(
                    "ultralytics/yolov5", "custom",
                    path=path, force_reload=False, trust_repo=True,
                )
                self.model.conf = self.confidence
            else:
                from ultralytics import YOLO
                self.model = YOLO(path)
            self.yolo_type  = yolo_type
            self.model_name = Path(path).stem
            return True
        except Exception as exc:
            self.error_occurred.emit(f"Model load error: {exc}")
            return False

    def push_frame(self, frame: np.ndarray):
        self._mutex.lock()
        self._queue = [frame]           # keep only latest
        self._mutex.unlock()

    def run(self):
        self._active = True
        while self._active:
            frame = None
            self._mutex.lock()
            if self._queue:
                frame = self._queue.pop(0)
            self._mutex.unlock()
            if frame is not None and self.model is not None:
                try:
                    self.result_ready.emit(*self._infer(frame))
                except Exception as exc:
                    self.error_occurred.emit(str(exc))
            self.msleep(15)

    def _infer(self, frame: np.ndarray):
        # Per-frame result: one frame = 1 OK or 1 NG (any NG box → NG frame)
        self.last_raw_frame  = frame.copy()
        self.last_detections = []
        frame_ng = False
        out = frame.copy()

        def draw_box(x1, y1, x2, y2, label, conf, is_ng):
            color = (220, 50, 0) if is_ng else (16, 185, 129)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            tag = f"{label}  {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
            cv2.rectangle(out, (x1, max(y1 - th - 8, 0)), (x1 + tw + 6, y1), color, -1)
            cv2.putText(out, tag, (x1 + 3, max(y1 - 3, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)

        if self.yolo_type == "yolov5":
            for _, row in self.model(frame[..., ::-1]).pandas().xyxy[0].iterrows():
                lbl   = str(row["name"]).lower()
                is_ng = any(k in lbl for k in NG_KEYWORDS)
                x1, y1, x2, y2 = (int(row["xmin"]), int(row["ymin"]),
                                   int(row["xmax"]), int(row["ymax"]))
                conf = float(row["confidence"])
                draw_box(x1, y1, x2, y2, row["name"], conf, is_ng)
                self.last_detections.append((x1, y1, x2, y2, int(row["class"]), conf))
                if is_ng:
                    frame_ng = True
        else:
            for r in self.model(frame, conf=self.confidence, verbose=False):
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    cls   = int(box.cls[0])
                    name  = r.names.get(cls, str(cls))
                    is_ng = any(k in name.lower() for k in NG_KEYWORDS)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    draw_box(x1, y1, x2, y2, name, conf, is_ng)
                    self.last_detections.append((x1, y1, x2, y2, cls, conf))
                    if is_ng:
                        frame_ng = True

        ng_n = int(frame_ng)
        ok_n = 1 - ng_n
        return out, ok_n, ng_n, float(ok_n * 100), float(ng_n * 100)

    def stop(self):
        self._active = False
        self.wait(3000)


# ─── Daily Check Dialog ───────────────────────────────────────
class DailyCheckDialog(QDialog):
    """Live camera test: place a reference sample → Run Test → model must return OK."""

    def __init__(self, det_worker, cam_source, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Daily System Check")
        self.setFixedSize(680, 500)
        self.setWindowFlags(Qt.Dialog | Qt.WindowTitleHint | Qt.WindowCloseButtonHint)
        self._worker     = det_worker
        self._cam_source = cam_source
        self._cap        = None
        self._frame      = None
        self._timer      = QTimer()
        self._timer.timeout.connect(self._grab_frame)
        self._build()
        self._open_camera()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Header
        hdr = QFrame()
        hdr.setFixedHeight(70)
        hdr.setStyleSheet(f"background:{C['blue']}; border:none;")
        hl = QHBoxLayout(hdr)
        hl.setContentsMargins(20, 0, 20, 0)
        ico = QLabel("☑")
        ico.setStyleSheet("color:rgba(255,255,255,0.9); font-size:28px; background:transparent;")
        col = QVBoxLayout()
        for text, style in [
            ("Daily System Check",
             "color:#fff; font-size:16px; font-weight:bold; background:transparent;"),
            ("Place a reference (OK) sample in view, then press  Run Test",
             "color:rgba(255,255,255,0.75); font-size:11px; background:transparent;"),
        ]:
            lbl = QLabel(text); lbl.setStyleSheet(style); col.addWidget(lbl)
        hl.addWidget(ico); hl.addSpacing(12); hl.addLayout(col); hl.addStretch()
        root.addWidget(hdr)

        # Live preview
        body = QFrame()
        body.setStyleSheet(f"background:{C['bg']}; border:none;")
        bl = QVBoxLayout(body)
        bl.setContentsMargins(16, 12, 16, 8)
        bl.setSpacing(8)

        self._preview_lbl = QLabel("Opening camera…")
        self._preview_lbl.setAlignment(Qt.AlignCenter)
        self._preview_lbl.setMinimumHeight(320)
        self._preview_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._preview_lbl.setStyleSheet(f"""
            QLabel {{ background:{C['row_alt']}; border:1.5px dashed {C['border']};
                     border-radius:6px; color:{C['txt3']}; font-size:14px; }}
        """)
        bl.addWidget(self._preview_lbl, 1)

        self._result_lbl = QLabel("Place reference sample in frame, then press  Run Test")
        self._result_lbl.setAlignment(Qt.AlignCenter)
        self._result_lbl.setStyleSheet(f"color:{C['txt2']}; font-size:12px;")
        bl.addWidget(self._result_lbl)
        root.addWidget(body, 1)

        # Footer
        ftr = QFrame()
        ftr.setFixedHeight(70)
        ftr.setStyleSheet(f"background:{C['card']}; border-top:1px solid {C['border']};")
        fl = QHBoxLayout(ftr)
        fl.setContentsMargins(18, 0, 18, 0)
        fl.setSpacing(10)
        fl.addStretch()

        btn_cancel = QPushButton("Cancel")
        btn_cancel.setFixedSize(90, 38)
        btn_cancel.clicked.connect(self.reject)

        self._btn_test = _mk_btn("▶  Run Test", C['blue'], C['blue_d'], height=38, fsize=13)
        self._btn_test.setFixedWidth(130)
        self._btn_test.clicked.connect(self._run_test)

        self._btn_confirm = _mk_btn("✓  Confirm", C['green'], "#059669", height=38, fsize=13)
        self._btn_confirm.setFixedWidth(120)
        self._btn_confirm.setEnabled(False)
        self._btn_confirm.clicked.connect(self.accept)

        fl.addWidget(btn_cancel)
        fl.addWidget(self._btn_test)
        fl.addWidget(self._btn_confirm)
        root.addWidget(ftr)

    def _open_camera(self):
        if isinstance(self._cam_source, str) and not self._cam_source:
            self._preview_lbl.setText("No video file selected.\nChoose a source in the main window first.")
            return
        self._cap = cv2.VideoCapture(self._cam_source)
        if self._cap.isOpened():
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self._timer.start(50)           # ~20 FPS live preview
        else:
            self._preview_lbl.setText(f"Cannot open camera source: {self._cam_source}")

    def _grab_frame(self):
        if self._cap and self._cap.isOpened():
            ret, frame = self._cap.read()
            if ret:
                self._frame = frame.copy()
                self._show(frame)

    def _show(self, frame: np.ndarray):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, c = rgb.shape
        pix = QPixmap.fromImage(QImage(rgb.data, w, h, c * w, QImage.Format_RGB888))
        self._preview_lbl.setPixmap(
            pix.scaled(self._preview_lbl.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _run_test(self):
        if self._frame is None:
            return
        if self._worker.model is None:
            self._result_lbl.setText("✗  No model loaded — load a model first")
            self._result_lbl.setStyleSheet(f"color:{C['red']}; font-size:12px; font-weight:bold;")
            return

        self._btn_test.setEnabled(False)
        self._result_lbl.setText("Running inference on sample…")
        self._result_lbl.setStyleSheet(f"color:{C['amber']}; font-size:12px;")
        QApplication.processEvents()

        try:
            annotated, _, ng_n, _, _ = self._worker._infer(self._frame)
            self._show(annotated)           # freeze on annotated result
            self._timer.stop()             # hold the annotated frame visible

            if ng_n == 0:                  # OK frame → pass
                self._result_lbl.setText("✓  PASS — Sample detected as OK.  System is ready.")
                self._result_lbl.setStyleSheet(
                    f"color:{C['green']}; font-size:13px; font-weight:bold;")
                self._btn_confirm.setEnabled(True)
            else:                          # NG detected → fail, allow retry
                self._result_lbl.setText("✗  FAIL — NG detected on sample.  Check sample and retry.")
                self._result_lbl.setStyleSheet(
                    f"color:{C['red']}; font-size:13px; font-weight:bold;")
                self._timer.start(50)      # resume live feed for retry
        except Exception as exc:
            self._result_lbl.setText(f"Inference error: {exc}")
            self._result_lbl.setStyleSheet(f"color:{C['red']}; font-size:12px;")
            self._timer.start(50)

        self._btn_test.setEnabled(True)
        self._btn_test.setText("↺  Re-test")

    def done(self, result):
        self._timer.stop()
        if self._cap:
            self._cap.release()
            self._cap = None
        super().done(result)


# ─── Collect Labels Dialog ────────────────────────────────────
class CollectLabelsDialog(QDialog):
    """
    Review unlabeled images one-by-one with model predictions overlaid.
    Y (Save) writes image + YOLO .txt to OUTPUT_DIR.
    N (Skip) moves to the next image without saving.
    """

    def __init__(self, det_worker: "DetectionWorker", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Collect Labels — Image Review")
        self.setMinimumSize(860, 640)
        self.setWindowFlags(Qt.Dialog | Qt.WindowTitleHint | Qt.WindowCloseButtonHint)
        self._worker      = det_worker
        self._candidates  = []
        self._current_idx = -1
        self._current_frame = None
        self._current_stem  = ""
        self._saved = 0
        self._build()

    # ── layout ────────────────────────────────────────────────
    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        # ─ Folder selectors ─
        folders = QGroupBox("Folders")
        fl = QVBoxLayout(folders)
        fl.setSpacing(6)
        self._img_edit = QLineEdit()
        self._img_edit.setPlaceholderText("Images folder (unlabeled)…")
        self._lbl_edit = QLineEdit()
        self._lbl_edit.setPlaceholderText("Existing labels folder (already-labeled images are skipped)…")
        self._out_edit = QLineEdit()
        self._out_edit.setPlaceholderText("Output folder…")
        self._out_edit.setText(str(Path(COLLECT_OUTPUT).resolve()))

        for edit, label in [
            (self._img_edit, "Images:"),
            (self._lbl_edit, "Labels (skip):"),
            (self._out_edit, "Output:"),
        ]:
            row = QHBoxLayout()
            lbl = QLabel(label)
            lbl.setFixedWidth(108)
            btn = QPushButton("Browse")
            btn.setFixedWidth(72)
            btn.clicked.connect(lambda _, e=edit: self._browse(e))
            row.addWidget(lbl)
            row.addWidget(edit, 1)
            row.addWidget(btn)
            fl.addLayout(row)
        root.addWidget(folders)

        # ─ Preview ─
        self._preview = QLabel("Select folders above, then press  Start Review")
        self._preview.setAlignment(Qt.AlignCenter)
        self._preview.setMinimumHeight(380)
        self._preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._preview.setStyleSheet(f"""
            QLabel {{ background:{C['row_alt']}; border:1.5px dashed {C['border']};
                     border-radius:6px; color:{C['txt3']}; font-size:14px; }}
        """)
        root.addWidget(self._preview, 1)

        # ─ Status ─
        self._status = QLabel("Ready")
        self._status.setAlignment(Qt.AlignCenter)
        self._status.setStyleSheet(f"color:{C['txt2']}; font-size:12px;")
        root.addWidget(self._status)

        # ─ Button row ─
        btn_row = QHBoxLayout()
        self._btn_start = _mk_btn("▶  Start Review", C['blue'],  C['blue_d'], height=38, fsize=13)
        self._btn_start.clicked.connect(self._start_review)
        self._btn_save  = _mk_btn("✓  Save  (Y)",    C['green'], "#059669",   height=38, fsize=13)
        self._btn_save.setEnabled(False)
        self._btn_save.clicked.connect(self._save_current)
        self._btn_skip  = QPushButton("✗  Skip  (N)")
        self._btn_skip.setFixedHeight(38)
        self._btn_skip.setEnabled(False)
        self._btn_skip.clicked.connect(self._next)
        btn_close = QPushButton("Close")
        btn_close.setFixedHeight(38)
        btn_close.clicked.connect(self.accept)

        btn_row.addWidget(self._btn_start)
        btn_row.addStretch()
        btn_row.addWidget(self._btn_save)
        btn_row.addWidget(self._btn_skip)
        btn_row.addSpacing(12)
        btn_row.addWidget(btn_close)
        root.addLayout(btn_row)

    # ── helpers ───────────────────────────────────────────────
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Y and self._btn_save.isEnabled():
            self._save_current()
        elif event.key() == Qt.Key_N and self._btn_skip.isEnabled():
            self._next()
        else:
            super().keyPressEvent(event)

    def _browse(self, edit: QLineEdit):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            edit.setText(folder)

    def _show_frame(self, frame: np.ndarray):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, c = rgb.shape
        pix = QPixmap.fromImage(QImage(rgb.data, w, h, c * w, QImage.Format_RGB888))
        self._preview.setPixmap(
            pix.scaled(self._preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _set_status(self, text: str, color: str = C['txt1']):
        self._status.setText(text)
        self._status.setStyleSheet(f"color:{color}; font-size:12px;")

    # ── review flow ───────────────────────────────────────────
    def _start_review(self):
        if self._worker.model is None:
            QMessageBox.warning(self, "No Model", "Load a model in the main window first.")
            return
        images_dir = Path(self._img_edit.text().strip())
        labels_dir = Path(self._lbl_edit.text().strip())
        for name, p in [("Images folder", images_dir), ("Labels folder", labels_dir)]:
            if not p.exists():
                QMessageBox.warning(self, "Folder Not Found",
                                    f"{name} does not exist:\n{p}")
                return

        all_imgs = sorted(p for p in images_dir.iterdir()
                          if p.suffix.lower() in IMG_EXTS)
        self._candidates = [p for p in all_imgs
                            if not (labels_dir / (p.stem + ".txt")).exists()]

        if not self._candidates:
            QMessageBox.information(self, "Nothing to Label",
                "All images already have labels, or no images found.")
            return

        self._current_idx = -1
        self._saved = 0
        self._btn_start.setEnabled(False)
        self._btn_save.setEnabled(True)
        self._btn_skip.setEnabled(True)
        self._next()

    def _next(self):
        self._current_idx += 1
        # Skip images with no detections automatically
        while self._current_idx < len(self._candidates):
            img_path = self._candidates[self._current_idx]
            self._set_status(
                f"[{self._current_idx + 1}/{len(self._candidates)}]  "
                f"{img_path.name}  —  running inference…", C['amber'])
            QApplication.processEvents()

            frame = cv2.imread(str(img_path))
            if frame is None:
                self._current_idx += 1
                continue

            try:
                annotated, _, _, _, _ = self._worker._infer(frame)
            except Exception as exc:
                self._set_status(f"Inference error: {exc}", C['red'])
                return

            n = len(self._worker.last_detections)
            if n == 0:
                self._current_idx += 1
                continue

            self._current_frame = frame
            self._current_stem  = img_path.stem
            self._show_frame(annotated)
            self._set_status(
                f"[{self._current_idx + 1}/{len(self._candidates)}]  "
                f"{img_path.name}  |  detections: {n}  |  saved: {self._saved}"
                f"  —  Y = Save  ·  N = Skip")
            return

        self._finish()

    def _save_current(self):
        if self._current_frame is None:
            return
        out_root   = Path(self._out_edit.text().strip() or COLLECT_OUTPUT)
        out_images = out_root / "images"
        out_labels = out_root / "labels"
        out_images.mkdir(parents=True, exist_ok=True)
        out_labels.mkdir(parents=True, exist_ok=True)

        img_h, img_w = self._current_frame.shape[:2]
        cv2.imwrite(str(out_images / f"{self._current_stem}.jpg"), self._current_frame)

        lines = []
        for x1, y1, x2, y2, cls, _ in self._worker.last_detections:
            cx = ((x1 + x2) / 2) / img_w
            cy = ((y1 + y2) / 2) / img_h
            bw = (x2 - x1) / img_w
            bh = (y2 - y1) / img_h
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        (out_labels / f"{self._current_stem}.txt").write_text(
            "\n".join(lines), encoding="utf-8")

        self._saved += 1
        self._next()

    def _finish(self):
        self._btn_save.setEnabled(False)
        self._btn_skip.setEnabled(False)
        self._btn_start.setEnabled(True)
        self._current_frame = None
        self._preview.setText("Review complete")
        out_root = Path(self._out_edit.text().strip() or COLLECT_OUTPUT)
        self._set_status(
            f"Done  ·  Reviewed: {len(self._candidates)}  ·  Saved: {self._saved}  "
            f"·  Output: {out_root.resolve()}", C['green'])
        QMessageBox.information(self, "Collection Complete",
            f"Review finished.\n\nSaved:  {self._saved} items\n"
            f"Output: {out_root.resolve()}\n\n"
            "Combine with unstable_labels/ from trial_run.py for retraining.")


# ─── Main Window ──────────────────────────────────────────────
class VisionPrimeWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME}  v{VERSION}")
        self.setMinimumSize(1300, 820)

        # State
        self._model_path     = ""
        self._daily_check_ok = False
        self._running        = False
        self._session_ok     = 0
        self._session_ng     = 0
        self._row_count      = 0
        self._fps_count      = 0
        self._last_log_ts    = 0.0
        self._last_raw_frame     = None           # latest unmodified frame for capture
        self._auto_capture_count = 0
        self._last_auto_ts       = 0.0
        self._total_det_count    = 0
        self._cam_thread: CameraThread | None = None

        # Detection worker
        self._det_worker = DetectionWorker()
        self._det_worker.result_ready.connect(self._on_detection)
        self._det_worker.error_occurred.connect(self._on_error)
        self._det_worker.start()

        self._fps_timer = QTimer()
        self._fps_timer.timeout.connect(self._tick_fps)

        self._build_ui()
        self._update_controls()

    # ══════════════════════════════════════════════════════════
    # UI — layout skeleton
    # ══════════════════════════════════════════════════════════
    def _build_ui(self):
        central = QWidget()
        central.setObjectName("central")
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 6)
        root.setSpacing(8)

        root.addWidget(self._make_title_bar())

        body = QHBoxLayout()
        body.setSpacing(8)
        body.addWidget(self._make_video_panel(), 3)
        body.addWidget(self._make_control_panel(), 1)
        root.addLayout(body, 3)

        root.addWidget(self._make_bottom_panel())   # Run Controls + Stats

        self._status_bar = QStatusBar()
        self._status_bar.setFixedHeight(24)
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage(f"Ready  ·  {APP_NAME} v{VERSION}")

    # ── Title bar ──────────────────────────────────────────────
    def _make_title_bar(self) -> QFrame:
        bar = QFrame()
        bar.setFixedHeight(54)
        bar.setStyleSheet(f"QFrame {{ background:{C['card']}; border:1px solid {C['border']}; border-radius:8px; }}")
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(16, 0, 16, 0)

        QLabel("◈").setStyleSheet(f"color:{C['blue']}; font-size:22px;")  # icon (local)
        logo = QLabel("◈")
        logo.setStyleSheet(f"color:{C['blue']}; font-size:22px;")
        app_lbl = QLabel(APP_NAME)
        app_lbl.setStyleSheet(f"color:{C['txt1']}; font-size:19px; font-weight:bold; letter-spacing:2px;")
        ver_lbl = QLabel(f"v{VERSION}")
        ver_lbl.setStyleSheet(f"color:{C['txt3']}; font-size:11px; margin-top:5px;")

        div = QFrame()
        div.setFrameShape(QFrame.VLine)
        div.setFixedWidth(1)
        div.setStyleSheet(f"background:{C['border']}; margin:10px 8px;")

        self._badge_ok  = self._badge("OK",  C['green'])
        self._badge_ng  = self._badge("NG",  C['red'])
        self._badge_fps = self._badge("FPS", C['blue'])

        self._dot  = QLabel("●")
        self._dot.setStyleSheet(f"color:{C['txt3']}; font-size:14px;")
        self._stxt = QLabel("STOPPED")
        self._stxt.setStyleSheet(f"color:{C['txt3']}; font-size:12px; font-weight:bold;")

        for w in (logo, app_lbl, ver_lbl, div, QLabel("Session:"),
                  self._badge_ok, self._badge_ng, self._badge_fps):
            lay.addWidget(w)
            if w is ver_lbl:
                lay.addSpacing(4)
        lay.addStretch()
        lay.addWidget(self._dot)
        lay.addSpacing(4)
        lay.addWidget(self._stxt)
        return bar

    def _badge(self, label: str, color: str) -> QLabel:
        w = QLabel(f"{label}: —")
        w.setStyleSheet(f"color:{color}; font-size:12px; font-weight:bold; "
                        f"background:{color}18; border:1px solid {color}55; "
                        f"border-radius:4px; padding:2px 10px;")
        return w

    # ── Video panel ────────────────────────────────────────────
    def _make_video_panel(self) -> QGroupBox:
        grp = QGroupBox("Live Detection Feed")
        lay = QVBoxLayout(grp)
        lay.setSpacing(8)

        self._video_lbl = QLabel()
        self._video_lbl.setAlignment(Qt.AlignCenter)
        self._video_lbl.setMinimumSize(640, 400)
        self._video_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._video_lbl.setStyleSheet(f"""
            QLabel {{ background:{C['row_alt']}; border:1.5px dashed {C['border']};
                     border-radius:6px; color:{C['txt3']}; font-size:14px; }}
        """)
        self._video_lbl.setText("No Signal\n\nLoad model  →  Daily Check  →  START")
        lay.addWidget(self._video_lbl, 1)

        # Source + confidence row
        ctrl = QHBoxLayout()
        ctrl.setSpacing(8)
        ctrl.addWidget(QLabel("Source:"))

        self._src_combo = QComboBox()
        self._src_combo.addItems(["Camera 0", "Camera 1", "Camera 2", "Video File…"])
        self._src_combo.setFixedWidth(130)
        self._src_combo.currentIndexChanged.connect(self._on_src_change)
        ctrl.addWidget(self._src_combo)

        self._video_path_edit = QLineEdit()
        self._video_path_edit.setPlaceholderText("Video file path…")
        self._video_path_edit.setVisible(False)
        ctrl.addWidget(self._video_path_edit, 1)

        self._video_browse_btn = QPushButton("Browse")
        self._video_browse_btn.setFixedWidth(76)
        self._video_browse_btn.setVisible(False)
        self._video_browse_btn.clicked.connect(
            lambda: self._video_path_edit.setText(
                QFileDialog.getOpenFileName(self, "Select Video", "",
                    "Video (*.mp4 *.avi *.mov *.mkv *.wmv);;All (*)")[0]
            )
        )
        ctrl.addWidget(self._video_browse_btn)

        ctrl.addStretch()
        ctrl.addWidget(QLabel("Confidence:"))
        self._conf_spin = QDoubleSpinBox()
        self._conf_spin.setRange(0.01, 0.99)
        self._conf_spin.setSingleStep(0.05)
        self._conf_spin.setValue(0.50)
        self._conf_spin.setDecimals(2)
        self._conf_spin.setFixedWidth(76)
        self._conf_spin.valueChanged.connect(
            lambda v: setattr(self._det_worker, "confidence", v)
        )
        ctrl.addWidget(self._conf_spin)
        lay.addLayout(ctrl)
        return grp

    # ── Right control panel ────────────────────────────────────
    def _make_control_panel(self) -> QWidget:
        w   = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)

        # ─ Model Settings ─
        mdl = QGroupBox("Model Settings")
        ml  = QVBoxLayout(mdl)
        ml.setSpacing(8)
        ml.setContentsMargins(8, 8, 8, 8)

        # YOLO framework — segmented toggle
        seg_row = QHBoxLayout()
        seg_row.setSpacing(0)
        self._btn_seg  = [QPushButton(k) for k in YOLO_TYPES.keys()]
        self._yolo_grp = QButtonGroup(self)
        tips = ["YOLOv8 / v9 / v10 / v11  (ultralytics)",
                "YOLOv5  (torch.hub)"]
        for i, btn in enumerate(self._btn_seg):
            btn.setCheckable(True)
            btn.setObjectName("seg_l" if i == 0 else "seg_r")
            btn.setFixedHeight(30)
            btn.setToolTip(tips[i])
            self._yolo_grp.addButton(btn, i)
            seg_row.addWidget(btn)
        self._btn_seg[0].setChecked(True)
        ml.addLayout(seg_row)

        # Model file browse row
        mrow = QHBoxLayout()
        mrow.setSpacing(4)
        self._model_edit = QLineEdit()
        self._model_edit.setPlaceholderText("Select .pt model file…")
        self._model_edit.setReadOnly(True)
        mrow.addWidget(self._model_edit)
        btn_browse = QPushButton("…")
        btn_browse.setFixedWidth(34)
        btn_browse.clicked.connect(self._browse_model)
        mrow.addWidget(btn_browse)
        ml.addLayout(mrow)

        self._btn_load = _mk_btn("Load Model", C['blue'], C['blue_d'], height=32, fsize=12)
        self._btn_load.setEnabled(False)
        self._btn_load.clicked.connect(self._load_model)
        ml.addWidget(self._btn_load)

        self._model_status = QLabel("No model loaded")
        self._model_status.setAlignment(Qt.AlignCenter)
        self._model_status.setStyleSheet(f"color:{C['txt3']}; font-size:11px;")
        ml.addWidget(self._model_status)
        lay.addWidget(mdl)

        # ─ Pre-Run Check ─
        chk = QGroupBox("Pre-Run Check")
        cl  = QVBoxLayout(chk)
        cl.setSpacing(6)
        self._btn_check = _mk_btn("☑  Run Daily Check", C['amber'], "#d97706", height=34, fsize=12)
        self._btn_check.clicked.connect(self._run_daily_check)
        cl.addWidget(self._btn_check)
        self._check_lbl = QLabel("⚠  Not completed")
        self._check_lbl.setAlignment(Qt.AlignCenter)
        self._check_lbl.setStyleSheet(f"color:{C['amber']}; font-size:11px;")
        cl.addWidget(self._check_lbl)
        lay.addWidget(chk)

        # ─ Data Collection ─
        coll = QGroupBox("Data Collection")
        col_lay = QVBoxLayout(coll)
        col_lay.setSpacing(6)
        col_lay.setContentsMargins(8, 8, 8, 8)
        self._btn_collect = _mk_btn("⊕  Collect Labels", C['purple'], "#6d28d9",
                                    height=34, fsize=12)
        self._btn_collect.setToolTip(
            "Review unlabeled images with model predictions\n"
            "and save pseudo-labels for retraining")
        self._btn_collect.clicked.connect(self._open_collect_dialog)
        col_lay.addWidget(self._btn_collect)
        col_lbl = QLabel("Pseudo-label images for retraining")
        col_lbl.setAlignment(Qt.AlignCenter)
        col_lbl.setStyleSheet(f"color:{C['txt3']}; font-size:11px;")
        col_lay.addWidget(col_lbl)
        lay.addWidget(coll)

        # ─ Detection Log ─
        lay.addWidget(self._make_table_panel(), 1)
        return w

    # ── Detection log table ────────────────────────────────────
    def _make_table_panel(self) -> QGroupBox:
        grp = QGroupBox("Detection Log")
        lay = QVBoxLayout(grp)
        lay.setSpacing(6)

        # Toolbar
        tb = QHBoxLayout()
        self._rec_lbl = QLabel("0 records")
        self._rec_lbl.setStyleSheet(f"color:{C['txt2']}; font-size:11px;")
        btn_exp = QPushButton("Export CSV")
        btn_exp.setFixedSize(100, 26)
        btn_exp.clicked.connect(self._export_csv)
        btn_clr = QPushButton("Clear Log")
        btn_clr.setFixedSize(88, 26)
        btn_clr.clicked.connect(self._clear_log)
        tb.addWidget(self._rec_lbl)
        tb.addStretch()
        tb.addWidget(btn_exp)
        tb.addWidget(btn_clr)
        lay.addLayout(tb)

        # Table
        headers = ["#", "Timestamp", "Model Name", "OK", "NG", "OK %", "NG %", "Status"]
        self._table = QTableWidget(0, len(headers))
        self._table.setHorizontalHeaderLabels(headers)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setAlternatingRowColors(True)
        self._table.verticalHeader().setVisible(False)
        self._table.setMinimumHeight(120)
        self._table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._table.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self._table.verticalHeader().setDefaultSectionSize(28)

        hdr = self._table.horizontalHeader()
        hdr.setSectionResizeMode(2, QHeaderView.Stretch)   # Model Name
        for col, w in {0: 44, 1: 160, 3: 58, 4: 58, 5: 70, 6: 70, 7: 66}.items():
            self._table.setColumnWidth(col, w)
            hdr.setSectionResizeMode(col, QHeaderView.Fixed)

        lay.addWidget(self._table)
        return grp

    # ── Bottom bar: Run Controls + Session Statistics ──────────
    def _make_bottom_panel(self) -> QWidget:
        pane = QWidget()
        pane.setFixedHeight(106)
        lay = QHBoxLayout(pane)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        # Run Controls
        run = QGroupBox("Run Controls")
        rl  = QHBoxLayout(run)
        rl.setSpacing(8)
        rl.setContentsMargins(10, 8, 10, 8)

        self._btn_start   = _mk_btn("▶  START",   C['green'],  "#059669")
        self._btn_stop    = _mk_btn("■  STOP",    C['red'],    "#dc2626")
        self._btn_reset   = _mk_btn("⟳  RESET",  C['purple'], "#6d28d9")
        self._btn_capture = _mk_btn("📷  Capture", C['amber'],  "#d97706")
        self._btn_start.setEnabled(False)
        self._btn_stop.setEnabled(False)
        self._btn_capture.setEnabled(False)
        self._btn_capture.setToolTip(
            "Save current frame + YOLO labels to collected_labels/")
        self._btn_start.clicked.connect(self._start)
        self._btn_stop.clicked.connect(self._stop)
        self._btn_reset.clicked.connect(self._reset_all)
        self._btn_capture.clicked.connect(self._capture_current_frame)
        for b in (self._btn_start, self._btn_stop, self._btn_reset, self._btn_capture):
            rl.addWidget(b)
        lay.addWidget(run, 2)

        # Session Statistics
        stat = QGroupBox("Session Statistics")
        sl   = QHBoxLayout(stat)if
        sl.setSpacing(8)
        sl.setContentsMargins(10, 8, 10, 8)

        self._w_ok_cnt  = self._stat_card("0", C['green'], "Detect Count")
        self._w_ng_cnt  = self._stat_card("0", C['red'],   "Not detect")
        self._w_ok_rate = self._stat_card("—", C['blue'],  "OK Rate")
        self._w_ng_rate = self._stat_card("—", "#f97316",  "NG Rate")
        for card in (self._w_ok_cnt, self._w_ng_cnt, self._w_ok_rate, self._w_ng_rate):
            sl.addWidget(card)
        lay.addWidget(stat, 3)
        return pane

    def _stat_card(self, value: str, color: str, label: str) -> QFrame:
        card = QFrame()
        card.setStyleSheet(f"""
            QFrame {{ background:{C['card']}; border:1px solid {C['border']};
                     border-left:3px solid {color}; border-radius:6px; }}
        """)
        vl = QVBoxLayout(card)
        vl.setContentsMargins(6, 8, 6, 6)
        vl.setSpacing(2)
        val = QLabel(value)
        val.setAlignment(Qt.AlignCenter)
        val.setStyleSheet(f"color:{color}; font-size:22px; font-weight:bold; border:none;")
        lbl = QLabel(label)
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet(f"color:{C['txt2']}; font-size:10px; border:none;")
        vl.addWidget(val)
        vl.addWidget(lbl)
        card.val_lbl = val
        return card

    # ══════════════════════════════════════════════════════════
    # Event handlers
    # ══════════════════════════════════════════════════════════
    def _on_src_change(self, idx: int):
        is_file = (idx == 3)
        self._video_path_edit.setVisible(is_file)
        self._video_browse_btn.setVisible(is_file)

    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select YOLO Model", "", "PyTorch Model (*.pt);;All (*)")
        if path:
            self._model_path = path
            self._model_edit.setText(Path(path).name)
            self._btn_load.setEnabled(True)

    def _load_model(self):
        idx   = self._yolo_grp.checkedId()
        ytype = list(YOLO_TYPES.values())[idx]
        self._model_status.setText("Loading…")
        self._model_status.setStyleSheet(f"color:{C['amber']}; font-size:11px;")
        QApplication.processEvents()
        if self._det_worker.load_model(self._model_path, ytype):
            self._model_status.setText(f"✓  {self._det_worker.model_name}")
            self._model_status.setStyleSheet(f"color:{C['green']}; font-size:11px;")
            self._status_bar.showMessage(f"Model loaded: {self._det_worker.model_name}")
        else:
            self._model_status.setText("✗  Load failed — check console")
            self._model_status.setStyleSheet(f"color:{C['red']}; font-size:11px;")
        self._update_controls()

    def _run_daily_check(self):
        idx    = self._src_combo.currentIndex()
        source = idx if idx < 3 else self._video_path_edit.text().strip()
        dlg = DailyCheckDialog(self._det_worker, source, self)
        dlg.setStyleSheet(QSS)
        if dlg.exec_() == QDialog.Accepted:
            self._daily_check_ok = True
            self._check_lbl.setText("✓  Daily check passed")
            self._check_lbl.setStyleSheet(f"color:{C['green']}; font-size:11px;")
            self._btn_check.setText("☑  Re-run Daily Check")
        self._update_controls()

    def _start(self):
        if not self._daily_check_ok:
            QMessageBox.warning(self, "Check Required",
                "Complete the Daily Pre-Run Check before starting.")
            return
        if not self._det_worker.model:
            QMessageBox.warning(self, "No Model", "Load a YOLO model first.")
            return
        idx    = self._src_combo.currentIndex()
        source = idx if idx < 3 else self._video_path_edit.text().strip()
        if source == "":
            QMessageBox.warning(self, "No Source", "Select a video file.")
            return

        self._cam_thread = CameraThread(source)
        self._cam_thread.frame_ready.connect(self._on_frame)
        self._cam_thread.error_occurred.connect(self._on_error)
        self._cam_thread.start()

        self._running = True
        self._session_ok = self._session_ng = self._fps_count = 0
        self._total_det_count = 0
        self._last_log_ts = 0.0
        self._fps_timer.start(1000)
        self._set_run_state(True)
        self._status_bar.showMessage(
            f"Running  ·  Model: {self._det_worker.model_name}  ·  Source: {source}")

    def _stop(self):
        if self._cam_thread:
            self._cam_thread.stop()
            self._cam_thread = None
        self._fps_timer.stop()
        self._set_run_state(False)
        self._video_lbl.setText("Stopped\n\nPress START to resume")
        self._status_bar.showMessage("Stopped")

    def _on_frame(self, frame: np.ndarray):
        self._fps_count += 1
        self._last_raw_frame = frame
        if self._det_worker.model:
            self._det_worker.push_frame(frame)
        else:
            self._show_frame(frame)

    def _on_detection(self, frame: np.ndarray, ok_n: int, ng_n: int, ok_r: float, ng_r: float):
        self._show_frame(frame)
        det_n = len(self._det_worker.last_detections)
        self._session_ok += ok_n
        self._session_ng += ng_n
        total = self._session_ok + self._session_ng or 1
        self._w_ok_cnt.val_lbl.setText(str(det_n))
        self._w_ng_cnt.val_lbl.setText(str(self._session_ng))
        self._w_ok_rate.val_lbl.setText(f"{self._session_ok / total * 100:.1f}%")
        self._w_ng_rate.val_lbl.setText(f"{self._session_ng / total * 100:.1f}%")
        self._badge_ok.setText(f"OK: {self._session_ok}")
        self._badge_ng.setText(f"NG: {self._session_ng}")
        is_ng = ng_n > 0
        now = time.time()
        if now - self._last_log_ts >= 1.0:
            self._last_log_ts = now
            self._add_row(ok_n, ng_n, ok_r, ng_r, is_ng)
        if is_ng and self._det_worker.last_raw_frame is not None:
            self._save_ng(self._det_worker.last_raw_frame)

    def _show_frame(self, frame: np.ndarray):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, c = rgb.shape
        pix = QPixmap.fromImage(QImage(rgb.data, w, h, c * w, QImage.Format_RGB888))
        self._video_lbl.setPixmap(
            pix.scaled(self._video_lbl.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _save_ng(self, frame: np.ndarray):
        try:
            now    = datetime.now()
            folder = Path(NG_BASE) / now.strftime("%Y-%m-%d")
            folder.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(folder / f"NG_{now.strftime('%H-%M-%S-%f')[:18]}.jpg"), frame)
        except Exception as exc:
            self._status_bar.showMessage(f"NG save error: {exc}")

    def _add_row(self, ok_n: int, ng_n: int, ok_r: float, ng_r: float, is_ng: bool):
        self._row_count += 1
        row = self._table.rowCount()
        self._table.insertRow(row)
        cells = [str(self._row_count),
                 datetime.now().strftime("%Y-%m-%d  %H:%M:%S"),
                 self._det_worker.model_name,
                 str(ok_n), str(ng_n),
                 f"{ok_r:.1f}%", f"{ng_r:.1f}%",
                 "NG" if is_ng else "OK"]
        for col, txt in enumerate(cells):
            item = QTableWidgetItem(txt)
            item.setTextAlignment(Qt.AlignCenter)
            if col == 7:
                item.setForeground(QColor(C['red'] if is_ng else C['green']))
                f = item.font();  f.setBold(True);  item.setFont(f)
            self._table.setItem(row, col, item)
        self._table.scrollToBottom()
        self._rec_lbl.setText(f"{self._row_count} records")

    def _tick_fps(self):
        self._badge_fps.setText(f"FPS: {self._fps_count}")
        self._fps_count = 0

    def _on_error(self, msg: str):
        self._status_bar.showMessage(f"Error: {msg}")

    def _reset_all(self):
        if QMessageBox.question(
            self, "Reset All",
            "Stop detection and reset ALL data?\n\nThis cannot be undone.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        ) != QMessageBox.Yes:
            return

        self._stop()
        self._daily_check_ok = False
        self._model_path     = ""
        self._session_ok = self._session_ng = self._row_count = 0
        self._total_det_count = 0
        self._det_worker.model      = None
        self._det_worker.model_name = ""

        self._model_edit.clear()
        self._btn_load.setEnabled(False)
        self._model_status.setText("No model loaded")
        self._model_status.setStyleSheet(f"color:{C['txt3']}; font-size:11px;")
        self._check_lbl.setText("⚠  Not completed")
        self._check_lbl.setStyleSheet(f"color:{C['amber']}; font-size:11px;")
        self._btn_check.setText("☑  Run Daily Check")

        for card, val in ((self._w_ok_cnt, "0"), (self._w_ng_cnt, "0"),
                          (self._w_ok_rate, "—"), (self._w_ng_rate, "—")):
            card.val_lbl.setText(val)
        for badge, txt in ((self._badge_ok, "OK: —"),
                           (self._badge_ng, "NG: —"),
                           (self._badge_fps, "FPS: —")):
            badge.setText(txt)

        self._table.setRowCount(0)
        self._rec_lbl.setText("0 records")
        self._video_lbl.setText("No Signal\n\nLoad model  →  Daily Check  →  START")
        self._update_controls()
        self._status_bar.showMessage(f"Reset complete  ·  {APP_NAME} v{VERSION}")

    def _export_csv(self):
        if self._table.rowCount() == 0:
            QMessageBox.information(self, "No Data", "No records to export.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export CSV",
            f"vision_prime_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "CSV (*.csv)")
        if not path:
            return
        try:
            cols = [self._table.horizontalHeaderItem(i).text()
                    for i in range(self._table.columnCount())]
            with open(path, "w", newline="", encoding="utf-8") as f:
                f.write(",".join(cols) + "\n")
                for r in range(self._table.rowCount()):
                    f.write(",".join(
                        self._table.item(r, c).text() if self._table.item(r, c) else ""
                        for c in range(self._table.columnCount())) + "\n")
            QMessageBox.information(self, "Exported", f"Saved to:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))

    def _clear_log(self):
        if self._table.rowCount() and QMessageBox.question(
            self, "Clear Log", "Clear all log entries?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        ) == QMessageBox.Yes:
            self._table.setRowCount(0)
            self._row_count = 0
            self._rec_lbl.setText("0 records")

    # ── Helpers ────────────────────────────────────────────────
    def _update_controls(self):
        can_start = (self._det_worker.model is not None
                     and self._daily_check_ok and not self._running)
        self._btn_start.setEnabled(can_start)
        self._btn_stop.setEnabled(self._running)

    def _set_run_state(self, running: bool):
        self._running = running
        color = C['green'] if running else C['txt3']
        label = "RUNNING" if running else "STOPPED"
        self._dot.setStyleSheet(f"color:{color}; font-size:14px;")
        self._stxt.setText(label)
        self._stxt.setStyleSheet(f"color:{color}; font-size:12px; font-weight:bold;")
        self._btn_capture.setEnabled(running and self._det_worker.model is not None)
        self._update_controls()

    def _open_collect_dialog(self):
        dlg = CollectLabelsDialog(self._det_worker, self)
        dlg.setStyleSheet(QSS)
        dlg.exec_()

    def _capture_current_frame(self):
        """Save the current live frame and its YOLO labels to collected_labels/."""
        if self._last_raw_frame is None:
            self._status_bar.showMessage("No frame available yet.")
            return
        if not self._det_worker.last_detections:
            self._status_bar.showMessage("No detections on current frame — nothing saved.")
            return

        frame  = self._last_raw_frame.copy()
        img_h, img_w = frame.shape[:2]

        out_root   = Path(COLLECT_OUTPUT)
        out_images = out_root / "images"
        out_labels = out_root / "labels"
        out_images.mkdir(parents=True, exist_ok=True)
        out_labels.mkdir(parents=True, exist_ok=True)

        stem = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        cv2.imwrite(str(out_images / f"{stem}.jpg"), frame)

        lines = []
        for x1, y1, x2, y2, cls, _ in self._det_worker.last_detections:
            cx = ((x1 + x2) / 2) / img_w
            cy = ((y1 + y2) / 2) / img_h
            bw = (x2 - x1) / img_w
            bh = (y2 - y1) / img_h
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        (out_labels / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")

        n = len(lines)
        self._status_bar.showMessage(
            f"Captured: {stem}.jpg  ({n} label{'s' if n != 1 else ''})  →  {out_root.resolve()}")

    def closeEvent(self, event):
        self._stop()
        self._det_worker.stop()
        event.accept()


# ─── Entry Point ──────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setStyle("Fusion")

    pal = QPalette()
    pal.setColor(QPalette.Window,          QColor(C['bg']))
    pal.setColor(QPalette.WindowText,      QColor(C['txt1']))
    pal.setColor(QPalette.Base,            QColor(C['card']))
    pal.setColor(QPalette.AlternateBase,   QColor(C['row_alt']))
    pal.setColor(QPalette.Text,            QColor(C['txt1']))
    pal.setColor(QPalette.Button,          QColor(C['card']))
    pal.setColor(QPalette.ButtonText,      QColor(C['txt1']))
    pal.setColor(QPalette.Highlight,       QColor(C['blue']))
    pal.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    pal.setColor(QPalette.ToolTipBase,     QColor(C['card']))
    pal.setColor(QPalette.ToolTipText,     QColor(C['txt1']))
    app.setPalette(pal)
    app.setStyleSheet(QSS)

    win = VisionPrimeWindow()
    win.show()
    sys.exit(app.exec_())
