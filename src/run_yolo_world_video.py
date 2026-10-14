"""YOLO-World (open-vocabulary) op een video; logt tool-detecties naar CSV.

YOLO-World krijgt de gereedschapsnamen uit ``tool_classes`` als tekstprompt,
zodat het zonder finetuning op de DUPLO-gereedschappen kan worden getest.
"""
import time

from duplo.benchmark import InferenceTimer, write_run_summary
from duplo.cli import parse_args
from duplo.config import load_config, resolve_device
from duplo.gpu import ensure_cuda_torch
from duplo.output import prepare_output_dir, save_screenshot, write_rows_csv
from duplo.video import VideoFrames, video_fps
from duplo.video_writer import DebugVideoWriter

CSV_COLUMNS = ["video", "frame", "timestamp", "klasse", "confidence",
               "bbox_x", "bbox_y", "bbox_w", "bbox_h"]


def main(cfg_path: str, device: str | None = None) -> None:
    cfg = load_config(cfg_path)
    out_dir = prepare_output_dir(cfg, "yolo_world")

    device = resolve_device(device, cfg)
    if device == "gpu":
        ensure_cuda_torch()
    # Pas importeren na een eventuele torch-installatie, anders draait een
    # reeds geladen CPU-build.
    from ultralytics import YOLO

    yolo_device = "cuda" if device == "gpu" else "cpu"
    model_name = cfg.get("yolo_world_model", "models/yolov8x-worldv2.pt")
    tool_classes = list(cfg["tool_classes"])
    model = YOLO(model_name)
    model.set_classes(tool_classes)
    print(f"YOLO-World ({model_name}) draait op {yolo_device.upper()} "
          f"met klassen: {', '.join(tool_classes)}")

    fps = video_fps(cfg["video_path"])
    stride = cfg.get("frame_stride", 1)
    screenshot_every = cfg.get("screenshot_every", 150)
    timer = InferenceTimer()
    video = (DebugVideoWriter(out_dir / "yolo_world_debug.mp4", fps / stride)
             if cfg.get("save_yolo_world_video", True) else None)
    rows = []
    frames = VideoFrames(cfg["video_path"], stride=stride, desc="YOLO-World")
    started = time.perf_counter()
    try:
        for frame_idx, frame in frames:
            with timer:
                result = model.predict(
                    frame, conf=cfg["conf_threshold"], device=yolo_device,
                    verbose=False)[0]
            for box in result.boxes:
                rows.append({
                    "video": cfg["video_path"], "frame": frame_idx,
                    "timestamp": round(frame_idx / fps, 3),
                    "klasse": result.names[int(box.cls)],
                    "confidence": round(float(box.conf), 3),
                    "bbox_x": int(box.xyxy[0][0]), "bbox_y": int(box.xyxy[0][1]),
                    "bbox_w": int(box.xyxy[0][2] - box.xyxy[0][0]),
                    "bbox_h": int(box.xyxy[0][3] - box.xyxy[0][1]),
                })
            annotated = result.plot()
            if video is not None:
                video.write(annotated)
            if screenshot_every > 0 and frame_idx % screenshot_every == 0:
                save_screenshot(out_dir, "yolo_world_frame", frame_idx, annotated)
    finally:
        # Ook bij een onderbreking de debug-video netjes afsluiten.
        if video is not None:
            video.release()

    write_rows_csv(out_dir, "yolo_world_detections.csv", rows, CSV_COLUMNS)
    summary = write_run_summary(
        out_dir, "yolo_world", model=model_name, device=device,
        video=cfg["video_path"], frames_read=frames.frames_read,
        frames_processed=len(timer.times_ms),
        elapsed_s=time.perf_counter() - started, rows=rows, timer=timer,
        csv_file="yolo_world_detections.csv",
        extra={"conf_threshold": cfg["conf_threshold"], "frame_stride": stride,
               "tekstprompt": tool_classes})
    print(f"{frames.frames_read} frames verwerkt, {len(rows)} detections, "
          f"{summary['inference_fps']} fps -> {out_dir}")


if __name__ == "__main__":
    args = parse_args()
    main(args.config, args.device)
