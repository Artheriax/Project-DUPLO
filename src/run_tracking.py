"""YOLO + DeepSORT op een video; logt de trackgeschiedenis en schrijft een
debug-video met bboxes, track-ID's en trajectories."""
import cv2
import numpy as np

from duplo.cli import parse_args
from duplo.config import load_config, resolve_device
from duplo.detection import YoloDetector
from duplo.gpu import ensure_cuda_torch
from duplo.output import prepare_output_dir, write_rows_csv
from duplo.tracking import ObjectTracker, TrackObservation
from duplo.video import VideoFrames, video_fps
from duplo.video_writer import DebugVideoWriter


def track_color(track_id: int) -> tuple[int, int, int]:
    """Deterministische, per track goed te onderscheiden BGR-kleur."""
    hue = np.uint8([[[track_id * 47 % 180, 255, 255]]])
    blue, green, red = cv2.cvtColor(hue, cv2.COLOR_HSV2BGR)[0][0]
    return int(blue), int(green), int(red)


def draw_tracks(frame, observations: list[TrackObservation], trajectories: dict,
                class_names: dict[int, str]) -> None:
    """Tekent bboxes, track-ID's, klassen en trajectories voor de debug-video."""
    for observation in observations:
        color = track_color(observation.track_id)
        if observation.bbox is not None:
            x1, y1, x2, y2 = (int(round(value)) for value in observation.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            name = class_names.get(observation.class_id, str(observation.class_id))
            label = f"#{observation.track_id} {name} {observation.confidence:.2f}"
            cv2.putText(frame, label, (x1, max(y1 - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            x1, y1, x2, y2 = (int(round(value))
                              for value in observation.predicted_bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), 1)
            cv2.putText(frame,
                        f"#{observation.track_id} occluded "
                        f"({observation.frames_since_detection}f)",
                        (x1, max(y1 - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (128, 128, 128), 1)
        points = trajectories.get(observation.track_id)
        if points and len(points) > 1:
            cv2.polylines(frame, [np.array(points, dtype=np.int32)], False,
                          color, 2)


def build_row(video_path: str, frame_number: int, fps: float,
              observation: TrackObservation,
              class_names: dict[int, str]) -> dict:
    """Eén CSV-rij per track per frame; occlusie-rijen hebben geen bbox."""
    bbox = observation.bbox
    center = observation.center
    velocity = observation.velocity
    return {
        "video": video_path,
        "frame_number": frame_number,
        "timestamp": round(frame_number / fps, 3),
        "track_id": observation.track_id,
        "class_id": observation.class_id,
        "class_name": class_names.get(observation.class_id),
        "confidence": (None if observation.confidence is None
                       else round(observation.confidence, 3)),
        "bbox_x1": None if bbox is None else int(round(bbox[0])),
        "bbox_y1": None if bbox is None else int(round(bbox[1])),
        "bbox_x2": None if bbox is None else int(round(bbox[2])),
        "bbox_y2": None if bbox is None else int(round(bbox[3])),
        "center_x": None if center is None else round(center[0], 1),
        "center_y": None if center is None else round(center[1], 1),
        "velocity_x": None if velocity is None else round(velocity[0], 2),
        "velocity_y": None if velocity is None else round(velocity[1], 2),
        "track_age": observation.track_age,
        "frames_since_detection": observation.frames_since_detection,
        "status": observation.status,
    }


def main(cfg_path: str, device: str | None = None) -> None:
    cfg = load_config(cfg_path)
    out_dir = prepare_output_dir(cfg, "tracking")

    device = resolve_device(device, cfg)
    if device == "gpu":
        ensure_cuda_torch()
    detector = YoloDetector(cfg, device)
    tracker = ObjectTracker(cfg, device)
    print(f"YOLO + DeepSORT draaien op {device.upper()}")

    fps = video_fps(cfg["video_path"])
    stride = cfg.get("frame_stride", 1)
    save_video = cfg.get("save_tracking_video", True)

    rows = []
    writer = None
    frames = VideoFrames(cfg["video_path"], stride=stride, desc="DeepSORT")
    for frame_number, frame in frames:
        observations = tracker.update(frame_number, frame, detector.detect(frame))
        rows.extend(build_row(cfg["video_path"], frame_number, fps,
                              observation, detector.class_names)
                    for observation in observations)
        if save_video:
            if writer is None:
                height, width = frame.shape[:2]
                writer = DebugVideoWriter(
                    out_dir / "tracking_debug.mp4", fps / stride,
                    (width, height))
            draw_tracks(frame, observations, tracker.trajectories,
                        detector.class_names)
            writer.write(frame)
    if writer is not None:
        writer.release()

    write_rows_csv(out_dir, "tracking_history.csv", rows)
    track_ids = {row["track_id"] for row in rows}
    print(f"{frames.frames_read} frames verwerkt, {len(track_ids)} tracks, "
          f"{len(rows)} rijen -> {out_dir}")


if __name__ == "__main__":
    args = parse_args()
    main(args.config, args.device)
