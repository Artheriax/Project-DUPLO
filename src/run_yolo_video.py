"""YOLO-objectdetectie op een video; logt detections naar CSV en slaat frames op."""
import argparse
from pathlib import Path

import cv2
import pandas as pd
import yaml
from ultralytics import YOLO


def main(cfg_path: str) -> None:
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(cfg["yolo_model"])
    video = cv2.VideoCapture(cfg["video_path"])
    stride = cfg.get("frame_stride", 1)
    rows, frame_idx = [], 0

    while True:
        ok, frame = video.read()
        if not ok:
            break
        if frame_idx % stride == 0:
            result = model.predict(frame, conf=cfg["conf_threshold"], verbose=False)[0]
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

    pd.DataFrame(rows).to_csv(out_dir / "yolo_detections.csv", index=False)
    video.release()
    print(f"{frame_idx} frames verwerkt, {len(rows)} detections -> {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    main(parser.parse_args().config)