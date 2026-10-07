"""YOLO-objectdetectie op een video; logt detections naar CSV en slaat frames op."""
from duplo.cli import parse_args
from duplo.config import load_config, resolve_device
from duplo.gpu import ensure_cuda_torch
from duplo.output import prepare_output_dir, save_screenshot, write_rows_csv
from duplo.video import VideoFrames


def main(cfg_path: str, device: str | None = None) -> None:
    cfg = load_config(cfg_path)
    out_dir = prepare_output_dir(cfg, "yolo")

    device = resolve_device(device, cfg)
    if device == "gpu":
        ensure_cuda_torch()
    # Pas importeren na een eventuele torch-installatie, anders draait een
    # reeds geladen CPU-build.
    from ultralytics import YOLO

    yolo_device = "cuda" if device == "gpu" else "cpu"
    print(f"YOLO draait op {yolo_device.upper()}")

    model = YOLO(cfg["yolo_model"])
    rows = []
    frames = VideoFrames(cfg["video_path"], stride=cfg.get("frame_stride", 1),
                         desc="YOLO")
    for frame_idx, frame in frames:
        result = model.predict(
            frame, conf=cfg["conf_threshold"], device=yolo_device, verbose=False
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
        if frame_idx % cfg["screenshot_every"] == 0:
            save_screenshot(out_dir, "yolo_frame", frame_idx, result.plot())

    write_rows_csv(out_dir, "yolo_detections.csv", rows)
    print(f"{frames.frames_read} frames verwerkt, {len(rows)} detections -> {out_dir}")


if __name__ == "__main__":
    args = parse_args()
    main(args.config, args.device)