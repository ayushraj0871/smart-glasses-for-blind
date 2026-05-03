#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
# Smart Glasses for the Blind
# Module: Object Detection Test  –  detection_test.py
# ─────────────────────────────────────────────────────────────────────────────
# Validates TFLite MobileNetV2 SSD for real-time obstacle and traffic-light
# detection on Raspberry Pi 4 (also runs on any laptop for baseline testing).
#
# Pipeline:
#   Webcam → CLAHE preprocessing → TFLite inference → DecisionModel → TTS
#
# Keyboard controls (OpenCV window must be focused):
#   q  – quit
#   s  – save current annotated frame as PNG
#   p  – pause / resume
#
# Usage:
#   python detection_test.py                          # live webcam
#   python detection_test.py --model models/detect.tflite
#   python detection_test.py --threshold 0.6 --no-display
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ── Optional heavy dependencies (fail gracefully) ────────────────────────────
try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False
    logging.warning("psutil not installed – CPU/memory metrics disabled. "
                    "Install with: pip install psutil")

try:
    # Prefer lightweight tflite-runtime on RPi; fall back to full TensorFlow
    try:
        import tflite_runtime.interpreter as tflite
        _TF_BACKEND = 'tflite_runtime'
    except ImportError:
        import tensorflow as tf
        tflite = tf.lite
        _TF_BACKEND = 'tensorflow'
    _TFLITE_AVAILABLE = True
except ImportError:
    _TFLITE_AVAILABLE = False
    logging.warning("Neither tflite-runtime nor tensorflow found – "
                    "inference will run in MOCK mode.")

# ── Internal modules ──────────────────────────────────────────────────────────
# Resolve package root so this script can be run directly or imported.
_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from modules.decision_model import calculate_move_safe, DecisionResult  # noqa: E402

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-7s  %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# COCO 2017 label map (91 entries – indices match MobileNetV2 SSD TFLite)
# ─────────────────────────────────────────────────────────────────────────────
COCO_LABELS: List[str] = [
    'background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    'street sign', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
    'hat', 'backpack', 'umbrella', 'shoe', 'eye glasses', 'handbag', 'tie',
    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'plate', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'mirror', 'dining table', 'window',
    'desk', 'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'blender', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush', 'hair brush',
]

# ── Detection category sets ───────────────────────────────────────────────────
OBSTACLE_CLASSES: set = {
    'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'dog', 'cat',
    'horse', 'chair', 'couch', 'dining table', 'desk', 'bed', 'potted plant',
    'suitcase', 'backpack', 'bench',
}

TRAFFIC_LIGHT_CLASSES: set = {'traffic light'}

GROUND_HAZARD_CLASSES: set = {
    # Future sensor integration placeholders
    'stairs', 'drain', 'drop',
}

# Colour palette for bounding-box annotations (BGR)
_BBOX_COLOURS: Dict[str, Tuple[int, int, int]] = {
    'obstacle':      (0, 165, 255),   # Orange
    'traffic_light': (0, 255, 0),     # Green
    'ground_hazard': (0, 0, 255),     # Red
    'other':         (200, 200, 200), # Grey
}


# ─────────────────────────────────────────────────────────────────────────────
# Detection record dataclass
# ─────────────────────────────────────────────────────────────────────────────
class Detection:
    """Structured result for a single detected object."""

    __slots__ = (
        'class_name', 'confidence', 'bbox',
        'obstacle_probability', 'latency_ms', 'timestamp',
        'category', 'decision',
    )

    def __init__(
        self,
        class_name: str,
        confidence: float,
        bbox: Tuple[int, int, int, int],
        obstacle_probability: float,
        latency_ms: float,
        timestamp: str,
        category: str,
        decision: Optional[DecisionResult] = None,
    ):
        self.class_name          = class_name
        self.confidence          = confidence
        self.bbox                = bbox            # (x1, y1, x2, y2)
        self.obstacle_probability = obstacle_probability
        self.latency_ms          = latency_ms
        self.timestamp           = timestamp
        self.category            = category
        self.decision            = decision

    def to_dict(self) -> dict:
        """Return dict matching the output format specified in the brief."""
        return {
            'class':               self.class_name,
            'confidence':          round(self.confidence, 4),
            'bbox':                self.bbox,
            'obstacle_probability': round(self.obstacle_probability, 4),
            'latency_ms':          round(self.latency_ms, 2),
            'timestamp':           self.timestamp,
        }

    def __repr__(self) -> str:
        return (f"Detection(class={self.class_name!r}, "
                f"conf={self.confidence:.2f}, "
                f"bbox={self.bbox}, "
                f"latency={self.latency_ms:.1f}ms)")


# ─────────────────────────────────────────────────────────────────────────────
# CLAHE helper
# ─────────────────────────────────────────────────────────────────────────────
def apply_clahe(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE enhancement to a BGR frame.
    Matches the preprocessing in modules/clahe_test.py but returns BGR
    so it can be fed directly into the TFLite model.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab   = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_enhanced = clahe.apply(l)
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


# ─────────────────────────────────────────────────────────────────────────────
# TFLite model wrapper
# ─────────────────────────────────────────────────────────────────────────────
class TFLiteDetector:
    """
    Wraps a TFLite MobileNetV2 SSD object-detection model.

    Expected model outputs (standard TFLite Object Detection API):
      0 – detection_boxes    [1, N, 4]  (y1, x1, y2, x2 normalised)
      1 – detection_classes  [1, N]
      2 – detection_scores   [1, N]
      3 – num_detections     [1]
    """

    DEFAULT_MODEL_PATHS = [
        'models/detect.tflite',
        'models/mobilenet_v2.tflite',
        'models/ssd_mobilenet_v2.tflite',
    ]
    INPUT_SIZE = (300, 300)

    def __init__(self, model_path: Optional[str] = None):
        self._interpreter = None
        self._input_details  = None
        self._output_details = None
        self._input_size     = self.INPUT_SIZE
        self._is_loaded      = False
        self._mock_mode      = False

        resolved = self._resolve_model_path(model_path)
        if resolved is None:
            logger.warning(
                "TFLite model not found – running in MOCK mode. "
                "Detections will be simulated for integration testing."
            )
            self._mock_mode = True
            return
        self._load(resolved)

    # ── Path resolution ───────────────────────────────────────────────────
    @staticmethod
    def _resolve_model_path(user_path: Optional[str]) -> Optional[str]:
        candidates = []
        if user_path:
            candidates.append(user_path)
        candidates.extend(TFLiteDetector.DEFAULT_MODEL_PATHS)
        # Also check relative to repo root
        root = Path(__file__).resolve().parent
        candidates += [str(root / p) for p in TFLiteDetector.DEFAULT_MODEL_PATHS]

        for p in candidates:
            if Path(p).is_file():
                return p
        return None

    # ── Model loading ─────────────────────────────────────────────────────
    def _load(self, model_path: str):
        if not _TFLITE_AVAILABLE:
            logger.error("TFLite not available – install tflite-runtime or tensorflow.")
            self._mock_mode = True
            return
        try:
            logger.info(f"Loading TFLite model from: {model_path}  (backend={_TF_BACKEND})")
            interp = tflite.Interpreter(model_path=model_path)
            interp.allocate_tensors()
            self._interpreter    = interp
            self._input_details  = interp.get_input_details()
            self._output_details = interp.get_output_details()
            # Determine actual input size from the model
            shape = self._input_details[0]['shape']   # [1, H, W, C]
            self._input_size = (shape[1], shape[2])
            self._is_loaded  = True
            logger.info(f"Model loaded ✓  input_size={self._input_size}")
        except Exception as exc:
            logger.error(f"Failed to load model: {exc}")
            self._mock_mode = True

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def mock_mode(self) -> bool:
        return self._mock_mode

    # ── Preprocessing ─────────────────────────────────────────────────────
    def _preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Resize and normalise frame to model input tensor."""
        h, w = self._input_size
        resized = cv2.resize(frame_bgr, (w, h))
        rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        dtype   = self._input_details[0]['dtype'] if self._input_details else np.uint8
        if dtype == np.float32:
            return (rgb.astype(np.float32) / 255.0)[np.newaxis, ...]
        return rgb.astype(np.uint8)[np.newaxis, ...]

    # ── Inference ─────────────────────────────────────────────────────────
    def infer(
        self,
        frame_bgr: np.ndarray,
        confidence_threshold: float = 0.5,
    ) -> Tuple[List[dict], float]:
        """
        Run inference on a BGR frame.

        Returns
        -------
        raw_detections : list of dicts with keys
            class_id, score, box (y1,x1,y2,x2 normalised)
        latency_ms     : inference time in milliseconds
        """
        t0 = time.perf_counter()

        if self._mock_mode:
            latency_ms = (time.perf_counter() - t0) * 1000
            return self._mock_detections(frame_bgr), latency_ms

        tensor = self._preprocess(frame_bgr)
        self._interpreter.set_tensor(self._input_details[0]['index'], tensor)
        self._interpreter.invoke()

        # Read outputs – handle both index-based and name-based ordering
        boxes   = self._interpreter.get_tensor(self._output_details[0]['index'])[0]
        classes = self._interpreter.get_tensor(self._output_details[1]['index'])[0]
        scores  = self._interpreter.get_tensor(self._output_details[2]['index'])[0]
        count   = int(self._interpreter.get_tensor(self._output_details[3]['index'])[0])

        latency_ms = (time.perf_counter() - t0) * 1000

        raw = []
        for i in range(min(count, len(scores))):
            score = float(scores[i])
            if score < confidence_threshold:
                continue
            raw.append({
                'class_id': int(classes[i]) + 1,   # Model outputs 0-based IDs
                                                    # (0=person, 1=bicycle…); +1
                                                    # aligns with COCO_LABELS where
                                                    # index 0 is 'background'.
                                                    # NOTE: if your model already
                                                    # outputs 1-based IDs, remove +1.
                'score':    score,
                'box':      boxes[i].tolist(),       # [y1, x1, y2, x2] normalised
            })
        return raw, latency_ms

    # ── Mock detections (for testing without a real model) ────────────────
    @staticmethod
    def _mock_detections(frame_bgr: np.ndarray) -> List[dict]:
        """Return a deterministic fake detection for integration testing."""
        return [
            {
                'class_id': 1,        # 'person'
                'score':    0.82,
                'box':      [0.1, 0.1, 0.9, 0.5],
            }
        ]


# ─────────────────────────────────────────────────────────────────────────────
# Post-processing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_label(class_id: int) -> str:
    if 0 <= class_id < len(COCO_LABELS):
        return COCO_LABELS[class_id]
    return f'unknown_{class_id}'


def _categorise(label: str) -> str:
    if label in OBSTACLE_CLASSES:
        return 'obstacle'
    if label in TRAFFIC_LIGHT_CLASSES:
        return 'traffic_light'
    if label in GROUND_HAZARD_CLASSES:
        return 'ground_hazard'
    return 'other'


def _box_to_pixels(
    box: List[float],
    frame_h: int,
    frame_w: int,
) -> Tuple[int, int, int, int]:
    """Convert normalised [y1,x1,y2,x2] to pixel (x1,y1,x2,y2)."""
    y1, x1, y2, x2 = box
    return (
        int(x1 * frame_w),
        int(y1 * frame_h),
        int(x2 * frame_w),
        int(y2 * frame_h),
    )


def post_process(
    raw_detections: List[dict],
    frame_bgr: np.ndarray,
    latency_ms: float,
) -> List[Detection]:
    """
    Convert raw TFLite outputs into structured Detection objects.
    Also runs the decision model for each detected obstacle.
    """
    h, w = frame_bgr.shape[:2]
    ts   = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    results: List[Detection] = []

    for raw in raw_detections:
        label    = _get_label(raw['class_id'])
        score    = raw['score']
        category = _categorise(label)

        # Map to obstacle probability for decision model
        if category == 'obstacle':
            obs_prob = score
        elif category == 'traffic_light':
            obs_prob = score * 0.5   # Traffic lights need interpretation
        else:
            obs_prob = 0.0

        bbox     = _box_to_pixels(raw['box'], h, w)
        decision = calculate_move_safe(obstacle_prob=obs_prob)

        results.append(Detection(
            class_name=label,
            confidence=score,
            bbox=bbox,
            obstacle_probability=obs_prob,
            latency_ms=latency_ms,
            timestamp=ts,
            category=category,
            decision=decision,
        ))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def annotate_frame(
    frame: np.ndarray,
    detections: List[Detection],
    fps: float,
    latency_ms: float,
    cpu_pct: float,
    mem_mb: float,
    paused: bool,
) -> np.ndarray:
    """Draw bounding boxes, labels, and HUD stats onto the frame."""
    out = frame.copy()

    for det in detections:
        colour = _BBOX_COLOURS.get(det.category, _BBOX_COLOURS['other'])
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)

        label_text = f"{det.class_name} {det.confidence:.0%}"
        (lw, lh), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        bg_y1 = max(y1 - lh - baseline - 4, 0)
        cv2.rectangle(out, (x1, bg_y1), (x1 + lw + 2, y1), colour, cv2.FILLED)
        cv2.putText(out, label_text, (x1 + 1, y1 - baseline - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

        if det.decision:
            risk_colour = {
                'SAFE':    (0, 200, 0),
                'CAUTION': (0, 165, 255),
                'DANGER':  (0, 0, 220),
            }.get(det.decision.risk_level, (200, 200, 200))
            cv2.putText(out, det.decision.risk_level,
                        (x1, y2 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, risk_colour, 1, cv2.LINE_AA)

    # HUD overlay (top-left)
    hud_lines = [
        f"FPS: {fps:.1f}",
        f"Latency: {latency_ms:.0f} ms",
    ]
    if _PSUTIL_AVAILABLE:
        hud_lines += [
            f"CPU: {cpu_pct:.0f}%",
            f"Mem: {mem_mb:.0f} MB",
        ]
    if paused:
        hud_lines.append("** PAUSED **")

    for i, line in enumerate(hud_lines):
        y = 22 + i * 22
        cv2.putText(out, line, (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 255), 1, cv2.LINE_AA)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# CSV logger
# ─────────────────────────────────────────────────────────────────────────────
_CSV_FIELDS = [
    'timestamp', 'class', 'confidence', 'obstacle_probability',
    'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2',
    'latency_ms', 'risk_level', 'is_safe', 'composite_score',
]


class DetectionLogger:
    """Appends detection events to a CSV file."""

    def __init__(self, csv_path: str):
        self._path = csv_path
        self._file = None
        self._writer = None
        self._open()

    def _open(self):
        path = Path(self._path)
        new_file = (not path.exists()) or path.stat().st_size == 0
        self._file   = open(self._path, 'a', newline='', encoding='utf-8')
        self._writer = csv.DictWriter(self._file, fieldnames=_CSV_FIELDS)
        if new_file:
            self._writer.writeheader()

    def log(self, det: Detection):
        x1, y1, x2, y2 = det.bbox
        row = {
            'timestamp':           det.timestamp,
            'class':               det.class_name,
            'confidence':          round(det.confidence, 4),
            'obstacle_probability': round(det.obstacle_probability, 4),
            'bbox_x1':             x1,
            'bbox_y1':             y1,
            'bbox_x2':             x2,
            'bbox_y2':             y2,
            'latency_ms':          round(det.latency_ms, 2),
            'risk_level':          det.decision.risk_level if det.decision else '',
            'is_safe':             det.decision.is_safe    if det.decision else '',
            'composite_score':     round(det.decision.composite_score, 4)
                                   if det.decision else '',
        }
        self._writer.writerow(row)
        self._file.flush()

    def close(self):
        if self._file:
            self._file.close()


# ─────────────────────────────────────────────────────────────────────────────
# Performance monitor
# ─────────────────────────────────────────────────────────────────────────────

class PerfMonitor:
    """Tracks FPS, latency, CPU, and memory."""

    def __init__(self, window: int = 30):
        self._window    = window
        self._timestamps: List[float] = []
        self._latencies:  List[float] = []
        self._process   = psutil.Process() if _PSUTIL_AVAILABLE else None

    def update(self, latency_ms: float):
        now = time.perf_counter()
        self._timestamps.append(now)
        self._latencies.append(latency_ms)
        # Keep rolling window
        if len(self._timestamps) > self._window:
            self._timestamps.pop(0)
            self._latencies.pop(0)

    @property
    def fps(self) -> float:
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        return (len(self._timestamps) - 1) / elapsed if elapsed > 0 else 0.0

    @property
    def avg_latency_ms(self) -> float:
        return sum(self._latencies) / len(self._latencies) if self._latencies else 0.0

    @property
    def cpu_percent(self) -> float:
        if self._process:
            return self._process.cpu_percent(interval=None)
        return 0.0

    @property
    def memory_mb(self) -> float:
        if self._process:
            return self._process.memory_info().rss / (1024 ** 2)
        return 0.0

    def summary(self) -> dict:
        return {
            'fps':             round(self.fps, 2),
            'avg_latency_ms':  round(self.avg_latency_ms, 2),
            'cpu_percent':     round(self.cpu_percent, 2),
            'memory_mb':       round(self.memory_mb, 2),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Main detection loop
# ─────────────────────────────────────────────────────────────────────────────

def run_detection_test(
    model_path:           Optional[str] = None,
    camera_index:         int   = 0,
    confidence_threshold: float = 0.5,
    display:              bool  = True,
    save_csv:             str   = 'detection_log.csv',
    max_frames:           Optional[int] = None,
) -> dict:
    """
    Run the full detection pipeline on a live webcam feed.

    Parameters
    ----------
    model_path           : Path to .tflite file (auto-discovered if None).
    camera_index         : OpenCV camera device index.
    confidence_threshold : Minimum score to report a detection.
    display              : Whether to open an OpenCV window.
    save_csv             : Path for the CSV detection log.
    max_frames           : Stop after N frames (None = run until 'q').

    Returns
    -------
    dict with summary statistics.
    """
    logger.info("=== Smart Glasses – Object Detection Test ===")
    logger.info(f"  confidence_threshold = {confidence_threshold}")
    logger.info(f"  camera_index         = {camera_index}")
    logger.info(f"  display              = {display}")
    logger.info(f"  csv_log              = {save_csv}")

    # ── Load model ────────────────────────────────────────────────────────
    detector = TFLiteDetector(model_path)
    if detector.mock_mode:
        logger.warning("Running in MOCK mode – no real inference performed.")
    else:
        logger.info(f"TFLite model loaded (backend={_TF_BACKEND})")

    # ── Open camera ───────────────────────────────────────────────────────
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        logger.error(f"Cannot open camera (index={camera_index}). "
                     "Ensure webcam is connected and not in use by another app.")
        return {'error': 'camera_not_available'}

    # Lower resolution for RPi-like constraints (640×480 target)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    logger.info(f"Camera opened  "
                f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}×"
                f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

    csv_logger  = DetectionLogger(save_csv)
    perf        = PerfMonitor()
    frame_count = 0
    save_count  = 0
    paused      = False

    logger.info("Running… press 'q' to quit, 's' to save frame, 'p' to pause")

    try:
        while True:
            if max_frames is not None and frame_count >= max_frames:
                logger.info(f"Reached max_frames={max_frames}, stopping.")
                break

            if not paused:
                ret, frame = cap.read()
                if not ret:
                    logger.error("Failed to read frame from camera.")
                    break

                # ── CLAHE preprocessing ───────────────────────────────────
                frame_enhanced = apply_clahe(frame)

                # ── TFLite inference ──────────────────────────────────────
                raw_dets, latency_ms = detector.infer(
                    frame_enhanced, confidence_threshold=confidence_threshold)

                # ── Post-process → Detection objects ──────────────────────
                detections = post_process(raw_dets, frame_enhanced, latency_ms)

                # ── Log to CSV ────────────────────────────────────────────
                for det in detections:
                    csv_logger.log(det)
                    logger.info(f"  {det}")

                # Handle no-detection case
                if not detections:
                    logger.debug("No detections this frame.")

                # ── Performance tracking ──────────────────────────────────
                perf.update(latency_ms)
                frame_count += 1

                # Print periodic summary
                if frame_count % 30 == 0:
                    s = perf.summary()
                    logger.info(
                        f"  [frame {frame_count}] "
                        f"FPS={s['fps']:.1f}  "
                        f"latency={s['avg_latency_ms']:.0f}ms  "
                        f"cpu={s['cpu_percent']:.0f}%  "
                        f"mem={s['memory_mb']:.0f}MB"
                    )
                    # Warn if performance targets not met
                    if s['avg_latency_ms'] > 200:
                        logger.warning("⚠  Latency > 200ms target")
                    if 0 < s['fps'] < 5:
                        logger.warning("⚠  FPS < 5 target")

            # ── Visualisation ─────────────────────────────────────────────
            if display and not paused:
                annotated = annotate_frame(
                    frame, detections,
                    fps=perf.fps,
                    latency_ms=perf.avg_latency_ms,
                    cpu_pct=perf.cpu_percent,
                    mem_mb=perf.memory_mb,
                    paused=paused,
                )
                cv2.imshow("Smart Glasses – Detection Test", annotated)

            # ── Keyboard input ────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF if display else 0xFF
            if key == ord('q'):
                logger.info("'q' pressed – exiting.")
                break
            elif key == ord('s') and not paused:
                save_path = f"detection_frame_{save_count:04d}.png"
                cv2.imwrite(save_path, annotated if display else frame)
                logger.info(f"Frame saved → {save_path}")
                save_count += 1
            elif key == ord('p'):
                paused = not paused
                logger.info("Paused." if paused else "Resumed.")

    finally:
        cap.release()
        if display:
            cv2.destroyAllWindows()
        csv_logger.close()

    summary = perf.summary()
    summary['total_frames'] = frame_count
    summary['csv_log']      = save_csv

    logger.info("─── Session Summary ───────────────────────────────")
    for k, v in summary.items():
        logger.info(f"  {k:<18} {v}")
    logger.info("───────────────────────────────────────────────────")

    # Final performance assertions (logged as warnings, not exceptions)
    if summary['avg_latency_ms'] > 200:
        logger.warning(f"Target NOT met: avg latency {summary['avg_latency_ms']}ms > 200ms")
    else:
        logger.info(f"✓ Latency target met: {summary['avg_latency_ms']}ms < 200ms")

    if 0 < summary['fps'] < 5:
        logger.warning(f"Target NOT met: FPS {summary['fps']} < 5")
    elif summary['fps'] >= 5:
        logger.info(f"✓ FPS target met: {summary['fps']} ≥ 5")

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Unit / integration tests (no camera or model required)
# ─────────────────────────────────────────────────────────────────────────────

def _run_unit_tests() -> bool:
    """
    Run lightweight tests that do NOT require a webcam or TFLite model.
    Tests cover:
      1. CLAHE preprocessing
      2. Post-processing / Detection construction
      3. Decision model integration
      4. CSV logger
      5. Performance monitor
      6. Mock inference
      7. Annotate frame (no crash)
    """
    print("\n─── Detection Module Unit Tests ─────────────────────")
    passed = 0
    total  = 0

    # 1. CLAHE preprocessing
    total += 1
    try:
        dummy = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        result = apply_clahe(dummy)
        assert result.shape == dummy.shape, "CLAHE output shape mismatch"
        assert result.dtype == np.uint8,    "CLAHE output dtype mismatch"
        print("  [1] PASS ✓  CLAHE preprocessing")
        passed += 1
    except Exception as exc:
        print(f"  [1] FAIL ✗  CLAHE preprocessing: {exc}")

    # 2. Mock detector – no model file needed
    total += 1
    try:
        det = TFLiteDetector(model_path='nonexistent_model.tflite')
        assert det.mock_mode, "Should be in mock mode"
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        raw, latency = det.infer(dummy, confidence_threshold=0.5)
        assert isinstance(raw, list),   "infer() should return a list"
        assert isinstance(latency, float), "latency should be float"
        print("  [2] PASS ✓  Mock detector inference")
        passed += 1
    except Exception as exc:
        print(f"  [2] FAIL ✗  Mock detector inference: {exc}")

    # 3. Post-processing → Detection objects
    total += 1
    try:
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        raw = [{'class_id': 1, 'score': 0.85, 'box': [0.1, 0.1, 0.9, 0.5]}]
        dets = post_process(raw, dummy_frame, latency_ms=42.0)
        assert len(dets) == 1
        d = dets[0]
        assert d.class_name == 'person'
        assert abs(d.confidence - 0.85) < 1e-6
        assert d.category == 'obstacle'
        assert abs(d.obstacle_probability - 0.85) < 1e-6
        assert d.decision is not None
        print("  [3] PASS ✓  Post-processing & Detection objects")
        passed += 1
    except Exception as exc:
        print(f"  [3] FAIL ✗  Post-processing: {exc}")

    # 4. to_dict() output format matches specification
    total += 1
    try:
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        raw = [{'class_id': 1, 'score': 0.95, 'box': [0.0, 0.0, 1.0, 1.0]}]
        dets = post_process(raw, dummy_frame, latency_ms=50.0)
        d_dict = dets[0].to_dict()
        required_keys = {'class', 'confidence', 'bbox',
                         'obstacle_probability', 'latency_ms', 'timestamp'}
        missing = required_keys - d_dict.keys()
        assert not missing, f"Missing keys: {missing}"
        assert d_dict['class'] == 'person'
        assert isinstance(d_dict['bbox'], tuple) and len(d_dict['bbox']) == 4
        print("  [4] PASS ✓  to_dict() output format")
        passed += 1
    except Exception as exc:
        print(f"  [4] FAIL ✗  to_dict() output format: {exc}")

    # 5. Decision model integration (obstacle → DANGER path)
    total += 1
    try:
        dummy_frame = np.zeros((200, 200, 3), dtype=np.uint8)
        raw = [{'class_id': 1, 'score': 0.99, 'box': [0.0, 0.0, 1.0, 1.0]}]
        dets = post_process(raw, dummy_frame, latency_ms=30.0)
        decision = dets[0].decision
        assert decision is not None
        assert decision.risk_level in ('CAUTION', 'DANGER'), \
            f"Expected CAUTION/DANGER for high-confidence person, got {decision.risk_level}"
        print("  [5] PASS ✓  Decision model integration (high obstacle → not safe)")
        passed += 1
    except Exception as exc:
        print(f"  [5] FAIL ✗  Decision model integration: {exc}")

    # 6. No-detection case (empty frame)
    total += 1
    try:
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = post_process([], dummy_frame, latency_ms=10.0)
        assert dets == [], "Empty raw → empty Detection list"
        print("  [6] PASS ✓  No-detection edge case")
        passed += 1
    except Exception as exc:
        print(f"  [6] FAIL ✗  No-detection edge case: {exc}")

    # 7. annotate_frame does not crash on empty detections
    total += 1
    try:
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        out = annotate_frame(dummy_frame, [], fps=10.0, latency_ms=50.0,
                              cpu_pct=20.0, mem_mb=128.0, paused=False)
        assert out.shape == dummy_frame.shape
        print("  [7] PASS ✓  annotate_frame with empty detections")
        passed += 1
    except Exception as exc:
        print(f"  [7] FAIL ✗  annotate_frame: {exc}")

    # 8. CSV logger write / read-back
    total += 1
    try:
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        raw = [{'class_id': 2, 'score': 0.75, 'box': [0.2, 0.2, 0.8, 0.8]}]
        dets = post_process(raw, dummy_frame, latency_ms=80.0)
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False,
                                         mode='w') as f:
            tmp_path = f.name
        try:
            csv_log = DetectionLogger(tmp_path)
            csv_log.log(dets[0])
            csv_log.close()
            rows = list(csv.DictReader(open(tmp_path)))
            assert len(rows) == 1, f"Expected 1 row, got {len(rows)}"
            assert rows[0]['class'] == 'bicycle'
            print("  [8] PASS ✓  CSV logger write/read-back")
            passed += 1
        finally:
            os.unlink(tmp_path)
    except Exception as exc:
        print(f"  [8] FAIL ✗  CSV logger: {exc}")

    # 9. Performance monitor FPS calculation
    total += 1
    try:
        pm = PerfMonitor(window=5)
        for _ in range(5):
            time.sleep(0.01)
            pm.update(latency_ms=50.0)
        fps = pm.fps
        assert fps > 0, "FPS should be positive after updates"
        assert pm.avg_latency_ms == 50.0
        print(f"  [9] PASS ✓  Performance monitor (fps={fps:.1f})")
        passed += 1
    except Exception as exc:
        print(f"  [9] FAIL ✗  Performance monitor: {exc}")

    # 10. Label map / categorisation helpers
    total += 1
    try:
        assert _get_label(1)  == 'person'
        assert _get_label(10) == 'traffic light'
        assert _get_label(0)  == 'background'
        assert _categorise('person')        == 'obstacle'
        assert _categorise('traffic light') == 'traffic_light'
        assert _categorise('banana')        == 'other'
        print("  [10] PASS ✓  Label map & categorisation helpers")
        passed += 1
    except Exception as exc:
        print(f"  [10] FAIL ✗  Label/category helpers: {exc}")

    print(f"\n  {passed}/{total} tests passed")
    print("─────────────────────────────────────────────────────\n")
    return passed == total


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Smart Glasses – TFLite Object Detection Test',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--model',      type=str,   default=None,
                        help='Path to .tflite detection model')
    parser.add_argument('--camera',     type=int,   default=0,
                        help='Camera device index')
    parser.add_argument('--threshold',  type=float, default=0.5,
                        help='Confidence threshold (0–1)')
    parser.add_argument('--no-display', action='store_true',
                        help='Disable OpenCV window (headless mode)')
    parser.add_argument('--csv',        type=str,   default='detection_log.csv',
                        help='Output CSV log path')
    parser.add_argument('--max-frames', type=int,   default=None,
                        help='Maximum frames to process then exit')
    parser.add_argument('--test',       action='store_true',
                        help='Run unit tests only (no webcam required)')
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()

    if args.test:
        ok = _run_unit_tests()
        sys.exit(0 if ok else 1)

    run_detection_test(
        model_path=args.model,
        camera_index=args.camera,
        confidence_threshold=args.threshold,
        display=not args.no_display,
        save_csv=args.csv,
        max_frames=args.max_frames,
    )
