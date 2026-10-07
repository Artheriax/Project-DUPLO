"""DeepSORT multi-object tracking: stabiele track-ID's en objectgeschiedenis.

DeepSORT koppelt de YOLO-detecties tussen frames aan elkaar op basis van
beweging (Kalman-filter) en appearance (embedding van de crop), geeft elk
object een uniek track_id en overbrugt korte occlusies.
"""
import importlib.resources
import sys
import types
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from duplo.detection import Detection

if TYPE_CHECKING:
    from deep_sort_realtime.deep_sort.track import Track


@dataclass(frozen=True)
class TrackObservation:
    """Toestand van één track in één frame.

    bbox, center en confidence zijn alleen gevuld als het object in dit frame
    gedetecteerd is (status "confirmed" of "tentative"). Bij status "occluded"
    was het object tijdelijk niet zichtbaar en bevat predicted_bbox de door het
    Kalman-filter voorspelde positie.
    """

    track_id: int
    class_id: int | None
    confidence: float | None
    bbox: tuple[float, float, float, float] | None
    predicted_bbox: tuple[float, float, float, float]
    center: tuple[float, float] | None
    velocity: tuple[float, float] | None
    track_age: int
    frames_since_detection: int
    status: str


@contextmanager
def _pkg_resources_fallback():
    """pkg_resources-shim voor deep_sort_realtime.

    deep_sort_realtime importeert pkg_resources, dat sinds setuptools 84 niet
    meer wordt meegeleverd. De shim voorziet alleen resource_filename (voor het
    vinden van de embedder-weights) en wordt daarna weer verwijderd.
    """
    try:
        import pkg_resources  # noqa: F401
    except ModuleNotFoundError:
        shim = types.ModuleType("pkg_resources")
        shim.resource_filename = lambda package, resource: str(
            importlib.resources.files(package).joinpath(resource))
        sys.modules["pkg_resources"] = shim
        try:
            yield
        finally:
            del sys.modules["pkg_resources"]
    else:
        yield


class ObjectTracker:
    """DeepSORT-wrapper die per track de beweging en geschiedenis bijhoudt."""

    def __init__(self, cfg: dict, device: str = "cpu") -> None:
        with _pkg_resources_fallback():
            from deep_sort_realtime.deepsort_tracker import DeepSort

            use_gpu = device == "gpu"
            self._deepsort = DeepSort(
                max_age=cfg.get("track_max_age", 30),
                n_init=cfg.get("track_n_init", 3),
                max_cosine_distance=cfg.get("track_max_cosine_distance", 0.2),
                embedder="mobilenet",
                embedder_gpu=use_gpu,
                half=use_gpu,
                bgr=True,
            )
        self.trajectory_length = cfg.get("track_trajectory_length", 60)
        self._last_seen: dict[int, tuple[int, float, float]] = {}
        self.trajectories: dict[int, deque[tuple[float, float]]] = {}

    def update(self, frame_number: int, frame: np.ndarray,
               detections: list[Detection]) -> list[TrackObservation]:
        """Koppelt de detecties van dit frame aan de bestaande tracks."""
        raw_detections = [
            ([detection.bbox[0], detection.bbox[1],
              detection.bbox[2] - detection.bbox[0],
              detection.bbox[3] - detection.bbox[1]],
             detection.confidence, detection.class_id)
            for detection in detections
        ]
        tracks = self._deepsort.update_tracks(raw_detections, frame=frame)
        return [self._observe(frame_number, track) for track in tracks]

    def _observe(self, frame_number: int, track: "Track") -> TrackObservation:
        # track_id is een numerieke string zolang DeepSort geen 'today' krijgt.
        track_id = int(track.track_id)
        visible = track.time_since_update == 0
        bbox = center = confidence = velocity = None
        if visible:
            # De originele detectie in plaats van de Kalman-voorspelling.
            x1, y1, x2, y2 = (float(value) for value in
                              track.to_ltrb(orig=True, orig_strict=True))
            bbox = (x1, y1, x2, y2)
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            confidence = float(track.det_conf)
            previous = self._last_seen.get(track_id)
            if previous is not None:
                elapsed = frame_number - previous[0]
                if elapsed > 0:
                    velocity = ((center[0] - previous[1]) / elapsed,
                                (center[1] - previous[2]) / elapsed)
            self._last_seen[track_id] = (frame_number, center[0], center[1])
            self.trajectories.setdefault(
                track_id, deque(maxlen=self.trajectory_length)).append(center)
        if not visible:
            status = "occluded"
        elif track.is_tentative():
            status = "tentative"
        else:
            status = "confirmed"
        return TrackObservation(
            track_id=track_id,
            class_id=track.det_class,
            confidence=confidence,
            bbox=bbox,
            predicted_bbox=tuple(float(value) for value in track.to_ltrb()),
            center=center,
            velocity=velocity,
            track_age=track.age,
            frames_since_detection=track.time_since_update,
            status=status,
        )
