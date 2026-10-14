"""YOLO + ByteTrack op een video; logt de trackgeschiedenis en schrijft een
debug-video met bboxes, track-ID's en trajectories."""
from duplo.cli import parse_args
from duplo.config import load_config, resolve_device
from duplo.detection import YoloDetector
from duplo.gpu import ensure_cuda_torch, setup_cuda_optimizations
from duplo.output import prepare_output_dir, write_rows_csv
from duplo.tracking import (CSV_COLUMNS, ObjectTracker, build_row, draw_tracks)
from duplo.video import VideoFrames, video_fps
from duplo.video_writer import DebugVideoWriter


def main(cfg_path: str, device: str | None = None) -> None:
    cfg = load_config(cfg_path)
    out_dir = prepare_output_dir(cfg, "tracking")

    device = resolve_device(device, cfg)
    if device == "gpu":
        ensure_cuda_torch()
    setup_cuda_optimizations(device)
    detector = YoloDetector(cfg, device)
    tracker = ObjectTracker(cfg)
    print(f"YOLO + ByteTrack draaien op {device.upper()}")

    fps = video_fps(cfg["video_path"])
    stride = cfg.get("frame_stride", 1)
    save_video = cfg.get("save_tracking_video", True)
    low_conf = cfg.get("track_low_thresh", 0.1)

    rows = []
    video = (DebugVideoWriter(out_dir / "tracking_debug.mp4", fps / stride,
                              encoder=cfg.get("video_encoder", "auto"))
             if save_video else None)
    frames = VideoFrames(cfg["video_path"], stride=stride, desc="ByteTrack")
    try:
        for frame_number, frame in frames:
            # Detecties met een lage score gaan mee: ByteTrack gebruikt die in
            # de tweede associatiestap om occlusies te overbruggen.
            detections = detector.detect(frame, conf=low_conf)
            observations = tracker.update(frame_number, frame, detections)
            rows.extend(build_row(cfg["video_path"], frame_number, fps,
                                  observation, detector.class_names)
                        for observation in observations)
            if video is not None:
                draw_tracks(frame, observations, tracker.trajectories,
                            detector.class_names)
                video.write(frame)
    finally:
        # Ook bij een onderbreking de debug-video netjes afsluiten.
        if video is not None:
            video.release()

    write_rows_csv(out_dir, "tracking_history.csv", rows, CSV_COLUMNS)
    track_ids = {row["track_id"] for row in rows}
    print(f"{frames.frames_read} frames verwerkt, {len(track_ids)} tracks, "
          f"{len(rows)} rijen -> {out_dir}")


if __name__ == "__main__":
    args = parse_args()
    main(args.config, args.device)
