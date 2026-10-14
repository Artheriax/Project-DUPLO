"""Grounding DINO (open-vocabulary) op een video; logt tool-detecties naar CSV.

Grounding DINO krijgt de gereedschapsnamen uit ``tool_classes`` als
tekstprompt ("screwdriver. wrench. ..."), zodat geen finetuning nodig is.
"""
import time

import cv2

from duplo.benchmark import InferenceTimer, write_run_summary
from duplo.cli import parse_args
from duplo.config import load_config, resolve_device
from duplo.detection import (Detection, GroundingDinoDetector, normalize_label)
from duplo.gpu import ensure_cuda_torch, setup_cuda_optimizations
from duplo.output import prepare_output_dir, save_screenshot, write_rows_csv
from duplo.video import VideoFrames, video_fps
from duplo.video_writer import DebugVideoWriter

CSV_COLUMNS = ["video", "frame", "timestamp", "klasse", "confidence",
               "bbox_x", "bbox_y", "bbox_w", "bbox_h"]


def draw_detections(frame, detections: list[Detection]):
    """Kopie van het frame met bboxes en labels voor de debug-video."""
    annotated = frame.copy()
    for detection in detections:
        x1, y1, x2, y2 = (int(round(value)) for value in detection.bbox)
        text = f"{detection.class_name} {detection.confidence:.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, text, (x1, max(y1 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return annotated


def main(cfg_path: str, device: str | None = None) -> None:
    cfg = load_config(cfg_path)
    out_dir = prepare_output_dir(cfg, "grounding_dino")

    device = resolve_device(device, cfg)
    if device == "gpu":
        ensure_cuda_torch()
    setup_cuda_optimizations(device)
    detector = GroundingDinoDetector(cfg, device)
    print(f"Grounding DINO ({detector.model_id}) draait op "
          f"{detector.torch_device.upper()} met prompt: {detector.prompt}")

    fps = video_fps(cfg["video_path"])
    stride = cfg.get("frame_stride", 1)
    screenshot_every = cfg.get("screenshot_every", 150)
    timer = InferenceTimer()
    video = (DebugVideoWriter(out_dir / "grounding_dino_debug.mp4", fps / stride,
                              encoder=cfg.get("video_encoder", "auto"))
             if cfg.get("save_grounding_dino_video", True) else None)
    rows = []
    frames = VideoFrames(cfg["video_path"], stride=stride, desc="Grounding DINO")
    started = time.perf_counter()
    try:
        for frame_idx, frame in frames:
            with timer:
                detections = detector.detect(frame)
            for detection in detections:
                x1, y1, x2, y2 = (int(round(value)) for value in detection.bbox)
                rows.append({
                    "video": cfg["video_path"], "frame": frame_idx,
                    "timestamp": round(frame_idx / fps, 3),
                    "klasse": detection.class_name,
                    "confidence": round(detection.confidence, 3),
                    "bbox_x": x1, "bbox_y": y1,
                    "bbox_w": x2 - x1, "bbox_h": y2 - y1,
                })
            need_video = video is not None
            need_shot = (screenshot_every > 0
                         and frame_idx % screenshot_every == 0)
            if need_video or need_shot:
                annotated = draw_detections(frame, detections)
                if need_video:
                    video.write(annotated)
                if need_shot:
                    save_screenshot(out_dir, "grounding_dino_frame", frame_idx,
                                    annotated)
    finally:
        # Ook bij een onderbreking de debug-video netjes afsluiten.
        if video is not None:
            video.release()

    write_rows_csv(out_dir, "grounding_dino_detections.csv", rows, CSV_COLUMNS)
    summary = write_run_summary(
        out_dir, "grounding_dino", model=detector.model_id, device=device,
        video=cfg["video_path"], frames_read=frames.frames_read,
        frames_processed=len(timer.times_ms),
        elapsed_s=time.perf_counter() - started, rows=rows, timer=timer,
        csv_file="grounding_dino_detections.csv",
        extra={"conf_threshold": detector.box_threshold, "frame_stride": stride,
               "tekstprompt": detector.prompt})
    print(f"{frames.frames_read} frames verwerkt, {len(rows)} detections, "
          f"{summary['inference_fps']} fps -> {out_dir}")


if __name__ == "__main__":
    args = parse_args()
    main(args.config, args.device)
