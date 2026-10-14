"""YOLO-objectdetectie op een video; logt detections naar CSV en slaat frames op."""
import time

from duplo.benchmark import InferenceTimer, write_run_summary
from duplo.cli import parse_args
from duplo.config import load_config, resolve_device
from duplo.detection import _load_yolo_model
from duplo.gpu import ensure_cuda_torch, setup_cuda_optimizations
from duplo.output import prepare_output_dir, save_screenshot, write_rows_csv
from duplo.video import VideoFrames, video_fps
from duplo.video_writer import DebugVideoWriter

CSV_COLUMNS = ["video", "frame", "klasse", "confidence",
               "bbox_x", "bbox_y", "bbox_w", "bbox_h"]


def main(cfg_path: str, device: str | None = None) -> None:
    cfg = load_config(cfg_path)
    out_dir = prepare_output_dir(cfg, "yolo")

    device = resolve_device(device, cfg)
    if device == "gpu":
        ensure_cuda_torch()
    setup_cuda_optimizations(device)
    # Pas importeren na een eventuele torch-installatie, anders draait een
    # reeds geladen CPU-build.
    from ultralytics import YOLO

    yolo_device = "cuda" if device == "gpu" else "cpu"
    use_half = bool(cfg.get("yolo_half", True)) and yolo_device == "cuda"
    print(f"YOLO draait op {yolo_device.upper()}"
          f"{' (FP16)' if use_half else ''}")

    model = _load_yolo_model(YOLO, cfg, "yolo_model", yolo_device)
    if cfg.get("yolo_fuse", True):
        try:
            model.fuse(verbose=False)
        except Exception as exc:
            print(f"Waarschuwing: YOLO-fuse overgeslagen ({exc}).")
    screenshot_every = cfg.get("screenshot_every", 150)
    stride = cfg.get("frame_stride", 1)
    fps = video_fps(cfg["video_path"])
    timer = InferenceTimer()
    video = (DebugVideoWriter(out_dir / "yolo_debug.mp4", fps / stride,
                              encoder=cfg.get("video_encoder", "auto"))
             if cfg.get("save_yolo_video", True) else None)
    rows = []
    frames = VideoFrames(cfg["video_path"], stride=stride, desc="YOLO")
    started = time.perf_counter()
    try:
        for frame_idx, frame in frames:
            with timer:
                result = model.predict(
                    frame, conf=cfg["conf_threshold"], device=yolo_device,
                    verbose=False, quantize=16 if use_half else None,
                    augment=False,
                )[0]
            for box in result.boxes:
                rows.append({
                    "video": cfg["video_path"], "frame": frame_idx,
                    "klasse": result.names[int(box.cls)],
                    "confidence": round(float(box.conf), 3),
                    "bbox_x": int(box.xyxy[0][0]), "bbox_y": int(box.xyxy[0][1]),
                    "bbox_w": int(box.xyxy[0][2] - box.xyxy[0][0]),
                    "bbox_h": int(box.xyxy[0][3] - box.xyxy[0][1]),
                })
            # result.plot() is CPU-werk: eenmaal berekenen en hergebruiken
            # voor debug-video en voorbeeldframe (zelfde beeld, zelfde CSV).
            need_video = video is not None
            need_shot = (screenshot_every > 0
                         and frame_idx % screenshot_every == 0)
            if need_video or need_shot:
                annotated = result.plot()
                if need_video:
                    video.write(annotated)
                if need_shot:
                    save_screenshot(out_dir, "yolo_frame", frame_idx, annotated)
    finally:
        # Ook bij een onderbreking de debug-video netjes afsluiten.
        if video is not None:
            video.release()

    write_rows_csv(out_dir, "yolo_detections.csv", rows, CSV_COLUMNS)
    summary = write_run_summary(
        out_dir, "yolo", model=cfg["yolo_model"], device=device,
        video=cfg["video_path"], frames_read=frames.frames_read,
        frames_processed=len(timer.times_ms),
        elapsed_s=time.perf_counter() - started, rows=rows, timer=timer,
        csv_file="yolo_detections.csv",
        extra={"conf_threshold": cfg["conf_threshold"], "frame_stride": stride})
    print(f"{frames.frames_read} frames verwerkt, {len(rows)} detections, "
          f"{summary['inference_fps']} fps -> {out_dir}")


if __name__ == "__main__":
    args = parse_args()
    main(args.config, args.device)