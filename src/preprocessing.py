"""SEM image loading and preprocessing (OpenCV)."""
from __future__ import annotations

import cv2
import numpy as np
import torch

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_gray(path_or_bytes) -> np.ndarray:
    """Load an SEM image as uint8 grayscale from a path or raw bytes."""
    if isinstance(path_or_bytes, (bytes, bytearray)):
        arr = np.frombuffer(path_or_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(str(path_or_bytes), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not decode image")
    return img


def load_mask(path) -> np.ndarray | None:
    """Load a binary mask (any non-zero pixel = defect). Returns None if unavailable."""
    if path is None or (isinstance(path, float)) or str(path) in ("", "nan"):
        return None
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return None if m is None else (m > 0).astype(np.uint8)


def enhance(gray: np.ndarray) -> np.ndarray:
    """Light denoise + CLAHE contrast normalisation, typical for SEM images."""
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


def to_tensor(gray: np.ndarray, size: int = 224) -> torch.Tensor:
    """Grayscale uint8 -> normalised 1x3xHxW float tensor for MobileNetV3."""
    img = cv2.resize(enhance(gray), (size, size), interpolation=cv2.INTER_AREA)
    img = np.repeat(img[..., None], 3, axis=2).astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)


def mask_area_pct(mask: np.ndarray | None) -> float | None:
    if mask is None or mask.size == 0:
        return None
    return float(mask.mean() * 100.0)


def overlay(gray: np.ndarray, heat: np.ndarray | None = None, mask: np.ndarray | None = None) -> np.ndarray:
    """RGB visualisation: optional heatmap (0..1 float) and/or mask contour."""
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    if heat is not None:
        h = cv2.resize(heat.astype(np.float32), (gray.shape[1], gray.shape[0]))
        cm = cv2.applyColorMap((np.clip(h, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
        rgb = cv2.addWeighted(rgb, 0.6, cv2.cvtColor(cm, cv2.COLOR_BGR2RGB), 0.4, 0)
    if mask is not None:
        m = cv2.resize(mask.astype(np.uint8), (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
        fill = rgb.copy()
        fill[m > 0] = (0, 255, 0)
        rgb = cv2.addWeighted(rgb, 0.7, fill, 0.3, 0)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(rgb, contours, -1, (0, 255, 0), 2)
    return rgb
