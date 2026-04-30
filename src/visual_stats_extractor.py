"""VisualStatsExtractor — Brightness, saturation, edge density features."""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
from PIL import Image
from tqdm.auto import tqdm


class VisualStatsExtractor:
    """Extract visual statistics from thumbnails."""

    COLUMNS = ["brightness", "saturation", "colorfulness", "red_ratio",
               "edge_density", "face_count", "face_size_ratio"]
    FALLBACK = [0.5, 0.5, 0.3, 0.1, 0.1, 0, 0.0]

    def __init__(self, thumbnail_manager, resize=160):
        self.tm = thumbnail_manager
        self.thumb_dir = thumbnail_manager.thumb_dir
        self.resize = resize

    def compute(self, video_ids):
        """Compute visual stats using ProcessPoolExecutor."""
        results = [None] * len(video_ids)

        with ProcessPoolExecutor(max_workers=4, initializer=_init_face_cascade) as pool:
            futures = {pool.submit(_process_thumbnail, vid, self.thumb_dir, self.resize): i
                      for i, vid in enumerate(video_ids)}
            for fut in tqdm(as_completed(futures), total=len(video_ids), desc="Visual stats"):
                idx = futures[fut]
                results[idx] = fut.result()

        return np.array(results, dtype=np.float32)


_face_cascade = None

def _init_face_cascade():
    """Initializer for ProcessPoolExecutor workers."""
    global _face_cascade
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    _face_cascade = cv2.CascadeClassifier(cascade_path)

def _process_thumbnail(vid, thumb_dir, resize=160):
    """Process a single video thumbnail. Module-level for pickling."""
    global _face_cascade
    try:
        path = os.path.join(thumb_dir, f"{vid}.jpg")
        img = Image.open(path).convert("RGB")
        img = img.resize((resize, resize), Image.Resampling.LANCZOS)
        img_rgb = np.array(img)
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        brightness = float(img_hsv[:, :, 2].mean()) / 255.0
        saturation = float(img_hsv[:, :, 1].mean()) / 255.0

        rg = img_rgb[:, :, 0].astype(float) - img_rgb[:, :, 1].astype(float)
        yg = img_rgb[:, :, 1].astype(float) - img_rgb[:, :, 2].astype(float)
        colorfulness = float(np.sqrt(rg**2 + yg**2).mean()) / 255.0

        red_ratio = float(img_rgb[:, :, 0].sum()) / (img_rgb.sum() + 1e-5)

        edges = cv2.Canny(img_gray, 50, 150)
        edge_density = float(edges.sum()) / (edges.size * 255.0)

        if _face_cascade is None:
            _init_face_cascade()
        faces = _face_cascade.detectMultiScale(img_gray, 1.3, 5)
        face_count = len(faces)
        if face_count > 0:
            total_face_area = sum(w * h for (x, y, w, h) in faces)
            face_size_ratio = total_face_area / (img_gray.shape[0] * img_gray.shape[1])
        else:
            face_size_ratio = 0.0

        return [brightness, saturation, colorfulness, red_ratio,
                edge_density, face_count, face_size_ratio]
    except Exception:
        return [0.5, 0.5, 0.3, 0.1, 0.1, 0, 0.0]
