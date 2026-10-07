"""ByteTrack multi-object tracking: stabiele track-ID's en objectgeschiedenis.

ByteTrack koppelt de YOLO-detecties tussen frames aan elkaar met een
Kalman-filter en IoU-associatie, geeft elk object een uniek track_id en
overbrugt korte occlusies door detecties met een lage confidence in een tweede
associatiestap te gebruiken. Hiervoor wordt de BYTETracker uit ultralytics
gebruikt (versie staat vast in requirements.txt); de tracker draait op de CPU.
"""
import types
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from duplo.detection import Detection

if TYPE_CHECKING:
    from ultralytics.trackers.byte_tracker import STrack


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


class _DetectionView:
    """Results-achtige view over Detection-objecten voor de ultralytics BYTETracker.

    De BYTETracker verwacht een object met `conf`, `cls` en `xywh` (pixels) dat
    boolean-indexering ondersteunt; deze klasse biedt dat op basis van onze
    eigen Detection-objecten.
    """

    def __init__(self, conf: np.ndarray, cls: np.ndarray,
                 xywh: np.ndarray) -> None:
        self.conf = conf
        self.cls = cls
        self.xywh = xywh

    @classmethod
    def from_detections(cls, detections: list[Detection]) -> "_DetectionView":
        boxes = np.array([detection.bbox for detection in detections],
                         dtype=np.float32).reshape(-1, 4)
        xywh = np.empty((len(detections), 4), dtype=np.float32)
        xywh[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2
        xywh[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2
        xywh[:, 2] = boxes[:, 2] - boxes[:, 0]
        xywh[:, 3] = boxes[:, 3] - boxes[:, 1]
        return cls(
            conf=np.array([detection.confidence for detection in detections],
                          dtype=np.float32),
            cls=np.array([detection.class_id for detection in detections],
                         dtype=np.float32),
            xywh=xywh,
        )

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, index) -> "_DetectionView":
        return _DetectionView(self.conf[index], self.cls[index],
                              self.xywh[index])


class ObjectTracker:
    """ByteTrack-wrapper die per track de beweging en geschiedenis bijhoudt."""

    def __init__(self, cfg: dict) -> None:
        # Pas importeren na een eventuele CUDA-torch-installatie, anders draait
        # een reeds geladen CPU-build.
        from ultralytics.trackers.byte_tracker import BYTETracker

        self._tracker = BYTETracker(types.SimpleNamespace(
            track_high_thresh=cfg.get("track_high_thresh", 0.25),
            track_low_thresh=cfg.get("track_low_thresh", 0.1),
            new_track_thresh=cfg.get("new_track_thresh", 0.25),
            track_buffer=cfg.get("track_buffer", 30),
            match_thresh=cfg.get("match_thresh", 0.8),
            fuse_score=cfg.get("fuse_score", True),
        ))
        self.trajectory_length = cfg.get("track_trajectory_length", 60)
        self._detections: list[Detection] = []
        self._last_seen: dict[int, tuple[int, float, float]] = {}
        self.trajectories: dict[int, deque[tuple[float, float]]] = {}

    def update(self, frame_number: int, frame: np.ndarray,
               detections: list[Detection]) -> list[TrackObservation]:
        """Koppelt de detecties van dit frame aan de bestaande tracks.

        De frame-pixels zijn voor ByteTrack niet nodig (associatie gebeurt op
        beweging en IoU); de parameter blijft zodat een tracker die de beelden
        wel gebruikt, zoals BoT-SORT, later uitwisselbaar is.
        """
        self._detections = detections
        self._tracker.update(_DetectionView.from_detections(detections))
        # tracked_stracks bevat de actieve tracks (bevestigd en net gestart),
        # lost_stracks de tracks binnen track_buffer die tijdelijk niet
        # gedetecteerd zijn. Beide zijn gedocumenteerde BYTETracker-attributen.
        observations = [self._observe(frame_number, track, visible=True)
                        for track in self._tracker.tracked_stracks]
        observations += [self._observe(frame_number, track, visible=False)
                         for track in self._tracker.lost_stracks]
        self._detections = []
        return observations

    def _observe(self, frame_number: int, track: "STrack",
                 visible: bool) -> TrackObservation:
        track_id = track.track_id
        predicted_bbox = tuple(float(value) for value in track.xyxy)
        bbox = center = confidence = velocity = None
        if visible:
            detection = self._matched_detection(track)
            # De originele detectie in plaats van de Kalman-schatting, zodat de
            # bbox gelijk blijft aan wat YOLO in dit frame zag.
            bbox = detection.bbox if detection is not None else predicted_bbox
            x1, y1, x2, y2 = bbox
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            confidence = float(track.score)
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
        elif track.is_activated:
            status = "confirmed"
        else:
            status = "tentative"
        return TrackObservation(
            track_id=track_id,
            class_id=int(track.cls),
            confidence=confidence,
            bbox=bbox,
            predicted_bbox=predicted_bbox,
            center=center,
            velocity=velocity,
            track_age=self._tracker.frame_id - track.start_frame,
            frames_since_detection=self._tracker.frame_id - track.frame_id,
            status=status,
        )

    def _matched_detection(self, track: "STrack") -> Detection | None:
        """De detectie uit dit frame die aan de track gekoppeld is (track.idx)."""
        index = int(track.idx)
        if 0 <= index < len(self._detections):
            return self._detections[index]
        return None
