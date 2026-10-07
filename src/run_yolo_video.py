"""YOLO-objectdetectie op een video; logt detections naar CSV en slaat frames op."""
import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import pandas as pd
import yaml
from tqdm import tqdm

CUDA_TORCH_INDEX = "https://download.pytorch.org/whl/cu130"


def cuda_available() -> bool:
    probe = subprocess.run(
        [sys.executable, "-c", "import torch; print(torch.cuda.is_available())"],
        capture_output=True, text=True,
    )
    return probe.returncode == 0 and probe.stdout.strip() == "True"


def ensure_cuda_torch() -> None:
    """Installeert de CUDA-build van torch als die nog niet aanwezig is."""
    if cuda_available():
        return
    print("CUDA-torch niet gevonden; CUDA-build van torch wordt geinstalleerd "
          "(kan enkele minuten duren)...")
    result = subprocess.run([
        sys.executable, "-m", "pip", "install", "-U", "torch", "torchvision",
        "--index-url", CUDA_TORCH_INDEX,
    ])
    if result.returncode != 0 or not cuda_available():
        raise SystemExit(
            "GPU gevraagd, maar CUDA is niet beschikbaar "
            "(installatie mislukt of geen NVIDIA-GPU aanwezig)."
        )


def main(cfg_path: str, device: str | None = None) -> None:
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    device = device or cfg.get("device", "cpu")
    if device == "gpu":
        ensure_cuda_torch()
    # Pas importeren na een eventuele torch-installatie, anders draait de
    # reeds geladen CPU-build.
    from ultralytics import YOLO

    yolo_device = "cuda" if device == "gpu" else "cpu"
    print(f"YOLO draait op {yolo_device.upper()}")

    model = YOLO(cfg["yolo_model"])
    video = cv2.VideoCapture(cfg["video_path"])
    stride = cfg.get("frame_stride", 1)
    rows, frame_idx = [], 0

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    progress = tqdm(total=total_frames if total_frames > 0 else None,
                    desc="YOLO", unit="frame")
    while True:
        ok, frame = video.read()
        if not ok:
            break
        progress.update(1)
        if frame_idx % stride == 0:
            result = model.predict(
                frame, conf=cfg["conf_threshold"], device=yolo_device, verbose=False
            )[0]
            x1, y1, x2, y2 = result.boxes.xyxy[0] if len(result.boxes) else (0, 0, 0, 0)
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
                cv2.imwrite(str(out_dir / f"yolo_frame_{frame_idx:05d}.jpg"),
                            result.plot())
        frame_idx += 1

    progress.close()
    pd.DataFrame(rows).to_csv(out_dir / "yolo_detections.csv", index=False)
    video.release()
    print(f"{frame_idx} frames verwerkt, {len(rows)} detections -> {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--device", choices=["cpu", "gpu"],
                        help="cpu of gpu; overschrijft 'device' uit de config")
    args = parser.parse_args()
    main(args.config, args.device)