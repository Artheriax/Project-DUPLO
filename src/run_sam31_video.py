"""SAM 3.1 (Segment Anything 3.1, Object Multiplex-checkpoint) op een video.

SAM 3.1 segmenteert elk object dat bij een tekstprompt hoort
(open-vocabulary), zonder finetuning. Via Ultralytics draait de
beeldinferentie met ``SAM3SemanticPredictor`` op ``sam3.1_multiplex.pt``;
Object Multiplex-videotracking wordt door Ultralytics nog niet ondersteund,
dus deze test doet per-frame beeldinferentie (net als de SAM3-test).

De gewichten van ``facebook/sam3.1`` staan achter een Hugging
Face-toegangsaanvraag: vraag die aan op
https://huggingface.co/facebook/sam3.1 en log daarna in met
``huggingface-cli login`` (of ``hf auth login``). Daarna downloadt Ultralytics
``sam3.1_multiplex.pt`` automatisch, of zet een lokaal bestand via
``sam31_model`` in de config.
"""
import time

import cv2

from duplo.benchmark import InferenceTimer, write_run_summary
from duplo.cli import parse_args
from duplo.config import load_config, resolve_device
from duplo.detection import Sam31Detector
from duplo.gpu import ensure_cuda_torch, setup_cuda_optimizations
from duplo.output import prepare_output_dir, save_screenshot, write_rows_csv
from duplo.video import VideoFrames, video_fps
from duplo.video_writer import DebugVideoWriter

CSV_COLUMNS = ["video", "frame", "timestamp", "klasse", "confidence",
               "bbox_x", "bbox_y", "bbox_w", "bbox_h", "mask_area"]


def draw_segmentations(frame, detections):
    """Kopie van het frame met maskers, bboxes en labels voor de debug-video."""
    annotated = frame.copy()
    overlay = annotated.copy()
    for detection in detections:
        if detection.mask is not None:
            overlay[detection.mask] = (0, 200, 0)
        x1, y1, x2, y2 = (int(round(value)) for value in detection.bbox)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, f"{detection.class_name} "
                               f"{detection.confidence:.2f}",
                    (x1, max(y1 - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 2)
    return cv2.addWeighted(overlay, 0.4, annotated, 0.6, 0)


def main(cfg_path: str, device: str | None = None) -> None:
    cfg = load_config(cfg_path)
    out_dir = prepare_output_dir(cfg, "sam31")

    device = resolve_device(device, cfg)
    if device == "gpu":
        ensure_cuda_torch()
    setup_cuda_optimizations(device)
    detector = Sam31Detector(cfg, device)
    print(f"SAM 3.1 ({detector.model_id}) draait op {detector.device.upper()} "
          f"met prompts: {', '.join(detector.prompts)}")

    fps = video_fps(cfg["video_path"])
    stride = cfg.get("frame_stride", 1)
    screenshot_every = cfg.get("screenshot_every", 150)
    timer = InferenceTimer()
    video = (DebugVideoWriter(out_dir / "sam31_debug.mp4", fps / stride,
                              encoder=cfg.get("video_encoder", "auto"))
             if cfg.get("save_sam31_video", True) else None)
    rows = []
    frames = VideoFrames(cfg["video_path"], stride=stride, desc="SAM 3.1")
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
                    "mask_area": (0 if detection.mask is None
                                  else int(detection.mask.sum())),
                })
            need_video = video is not None
            need_shot = (screenshot_every > 0
                         and frame_idx % screenshot_every == 0)
            if need_video or need_shot:
                annotated = draw_segmentations(frame, detections)
                if need_video:
                    video.write(annotated)
                if need_shot:
                    save_screenshot(out_dir, "sam31_frame", frame_idx, annotated)
    finally:
        # Ook bij een onderbreking de debug-video netjes afsluiten.
        if video is not None:
            video.release()

    write_rows_csv(out_dir, "sam31_detections.csv", rows, CSV_COLUMNS)
    summary = write_run_summary(
        out_dir, "sam31", model=detector.model_id,
        device=device, video=cfg["video_path"],
        frames_read=frames.frames_read, frames_processed=len(timer.times_ms),
        elapsed_s=time.perf_counter() - started, rows=rows, timer=timer,
        csv_file="sam31_detections.csv",
        extra={"conf_threshold": cfg["conf_threshold"], "frame_stride": stride,
               "tekstprompt": detector.prompts})
    print(f"{frames.frames_read} frames verwerkt, {len(rows)} segmentaties, "
          f"{summary['inference_fps']} fps -> {out_dir}")


if __name__ == "__main__":
    args = parse_args()
    main(args.config, args.device)
