"""Centrale CSV-schema's en rij-builders voor alle pipeline-stappen.

Eén bron voor kolomvolgorde, zodat `run_*` scripts en `compare_models.py`
niet divergeren. Detectie (9 kolommen, met `timestamp`), segmentatie
(10 kolommen, met `mask_area`), tracking (18 kolommen uit `tracking.py`)
en handpose (8 kolommen) staan hier vast.
"""
DETECTION_COLUMNS = ["video", "frame", "timestamp", "klasse", "confidence",
                     "bbox_x", "bbox_y", "bbox_w", "bbox_h"]

SEGMENTATION_COLUMNS = [*DETECTION_COLUMNS, "mask_area"]

HANDPOSE_COLUMNS = ["video", "frame", "hand", "landmark", "x", "y", "z",
                    "zichtbaar"]


def detection_row(video_path: str, frame_idx: int, fps: float,
                  klasse: str, confidence: float,
                  x1: int, y1: int, x2: int, y2: int) -> dict:
    """Eén detectie-rij met `timestamp` en `bbox_x/y/w/h` conventie."""
    return {
        "video": video_path,
        "frame": frame_idx,
        "timestamp": round(frame_idx / fps, 3) if fps else 0.0,
        "klasse": klasse,
        "confidence": round(float(confidence), 3),
        "bbox_x": int(x1),
        "bbox_y": int(y1),
        "bbox_w": int(x2 - x1),
        "bbox_h": int(y2 - y1),
    }


def segmentation_row(video_path: str, frame_idx: int, fps: float,
                     klasse: str, confidence: float,
                     x1: int, y1: int, x2: int, y2: int,
                     mask_area: int) -> dict:
    """Eén segmentatie-rij (`mask_area` als pixel-count op frame-resolutie)."""
    row = detection_row(video_path, frame_idx, fps, klasse, confidence,
                        x1, y1, x2, y2)
    row["mask_area"] = int(mask_area)
    return row
