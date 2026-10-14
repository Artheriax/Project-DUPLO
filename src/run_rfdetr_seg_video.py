"""RF-DETR-segmentatie op een video; logt detecties met maskeroppervlak naar CSV.

Gebruikt de pretrained COCO-gewichten van het ``rfdetr``-pakket
(RFDETRSeg*-modellen); de opzet volgt de RF-DETR-detectietest.
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
               "bbox_x", "bbox_y", "bbox_w", "bbox_h", "mask_area"]


def build_model(cfg: dict, device: str):
    """RFDETRSeg-model uit de config; pas importeren na een CUDA-installatie."""
    import rfdetr

    model_name = cfg.get("rfdetr_seg_model", "RFDETRSeg2XLarge")
    return model_name, getattr(rfdetr, model_name)(
        device="cuda" if device == "gpu" else "cpu")


def annotate(frame, detections):
    """Frame met maskers, bboxes en labels voor de debug-video."""
    import supervision as sv

    labels = [f"{name} {float(confidence):.2f}" for name, confidence in
              zip(detections.data["class_name"], detections.confidence)]
    annotated = sv.MaskAnnotator().annotate(frame, detections)
    annotated = sv.BoxAnnotator(thickness=2).annotate(annotated, detections)
    return sv.LabelAnnotator(text_scale=0.5, text_thickness=1).annotate(
        annotated, detections, labels)


def main(cfg_path: str, device: str | None = None) -> None:
    cfg = load_config(cfg_path)
    out_dir = prepare_output_dir(cfg, "rfdetr_seg")

    device = resolve_device(device, cfg)
    if device == "gpu":
        ensure_cuda_torch()
    model_name, model = build_model(cfg, device)
    if device == "gpu" and cfg.get("rfdetr_optimize", True):
        # Gedocumenteerde real-time route van RF-DETR (FP16 + compile).
        import torch
        model.inference(dtype=torch.float16)
    print(f"RF-DETR-seg ({model_name}) draait op {device.upper()}")

    threshold = cfg["conf_threshold"]
    fps = video_fps(cfg["video_path"])
    stride = cfg.get("frame_stride", 1)
    screenshot_every = cfg.get("screenshot_every", 150)
    timer = InferenceTimer()
    video = (DebugVideoWriter(out_dir / "rfdetr_seg_debug.mp4", fps / stride)
             if cfg.get("save_rfdetr_seg_video", True) else None)
    rows = []
    frames = VideoFrames(cfg["video_path"], stride=stride, desc="RF-DETR-seg")
    started = time.perf_counter()
    try:
        for frame_idx, frame in frames:
            # RF-DETR verwacht RGB-beelden; OpenCV levert BGR.
            with timer:
                detections = model.predict(
                    frame[:, :, ::-1], threshold=threshold,
                    include_source_image=False)
            class_names = detections.data["class_name"]
            for index, (box, confidence) in enumerate(
                    zip(detections.xyxy, detections.confidence)):
                x1, y1, x2, y2 = (int(round(float(value))) for value in box)
                rows.append({
                    "video": cfg["video_path"], "frame": frame_idx,
                    "timestamp": round(frame_idx / fps, 3),
                    "klasse": str(class_names[index]),
                    "confidence": round(float(confidence), 3),
                    "bbox_x": x1, "bbox_y": y1,
                    "bbox_w": x2 - x1, "bbox_h": y2 - y1,
                    "mask_area": int(detections.mask[index].sum()),
                })
            annotated = annotate(frame, detections)
            if video is not None:
                video.write(annotated)
            if screenshot_every > 0 and frame_idx % screenshot_every == 0:
                save_screenshot(out_dir, "rfdetr_seg_frame", frame_idx, annotated)
    finally:
        # Ook bij een onderbreking de debug-video netjes afsluiten.
        if video is not None:
            video.release()

    write_rows_csv(out_dir, "rfdetr_seg_detections.csv", rows, CSV_COLUMNS)
    summary = write_run_summary(
        out_dir, "rfdetr_seg", model=model_name, device=device,
        video=cfg["video_path"], frames_read=frames.frames_read,
        frames_processed=len(timer.times_ms),
        elapsed_s=time.perf_counter() - started, rows=rows, timer=timer,
        csv_file="rfdetr_seg_detections.csv",
        extra={"conf_threshold": threshold, "frame_stride": stride})
    print(f"{frames.frames_read} frames verwerkt, {len(rows)} segmentaties, "
          f"{summary['inference_fps']} fps -> {out_dir}")


if __name__ == "__main__":
    args = parse_args()
    main(args.config, args.device)
