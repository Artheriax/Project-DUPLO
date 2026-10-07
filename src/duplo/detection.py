"""YOLO-objectdetectie als zelfstandige pipelinecomponent."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Detection:
    """Eén objectdetectie met bbox in pixelcoördinaten (x1, y1, x2, y2)."""

    bbox: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str


class YoloDetector:
    """Wrapper rond een ultralytics YOLO-model.

    Los te vervangen component: de rest van de pipeline werkt met Detection
    objecten en is niet afhankelijk van ultralytics.
    """

    def __init__(self, cfg: dict, device: str = "cpu") -> None:
        # Pas importeren na een eventuele CUDA-torch-installatie, anders draait
        # een reeds geladen CPU-build.
        from ultralytics import YOLO

        self.device = "cuda" if device == "gpu" else "cpu"
        self.conf_threshold = cfg["conf_threshold"]
        self.model = YOLO(cfg["yolo_model"])
        self.class_names = dict(self.model.names)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        result = self.model.predict(
            frame, conf=self.conf_threshold, device=self.device, verbose=False
        )[0]
        detections = []
        for box in result.boxes:
            x1, y1, x2, y2 = (float(value) for value in box.xyxy[0])
            class_id = int(box.cls)
            detections.append(Detection(
                bbox=(x1, y1, x2, y2),
                confidence=float(box.conf),
                class_id=class_id,
                class_name=self.class_names[class_id],
            ))
        return detections
