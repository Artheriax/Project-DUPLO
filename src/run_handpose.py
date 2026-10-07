"""MediaPipe Hands op een video; logt 21 landmerken per hand naar CSV."""
import argparse
from urllib.request import urlretrieve
from pathlib import Path

import cv2
import mediapipe as mp
import pandas as pd
import yaml
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tqdm import tqdm


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


def get_model_path(cfg: dict) -> Path:
    model_path = Path(cfg.get("hand_model_path", "models/hand_landmarker.task"))
    if not model_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Handmodel downloaden naar {model_path}...")
        urlretrieve(MODEL_URL, model_path)
    return model_path


def draw_landmarks(frame, hand_landmarks) -> None:
    height, width = frame.shape[:2]
    points = [(int(lm.x * width), int(lm.y * height)) for lm in hand_landmarks]
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (0, 255, 0), 2)
    for point in points:
        cv2.circle(frame, point, 4, (0, 0, 255), -1)


def create_landmarker(cfg: dict, model_path: Path, delegate) -> vision.HandLandmarker:
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(model_path), delegate=delegate),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=cfg["max_hands"],
        min_hand_detection_confidence=cfg["conf_threshold"],
        min_hand_presence_confidence=cfg["conf_threshold"],
        min_tracking_confidence=cfg["conf_threshold"],
    )
    return vision.HandLandmarker.create_from_options(options)


def main(cfg_path: str, device: str | None = None) -> None:
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    device = device or cfg.get("device", "cpu")
    model_path = get_model_path(cfg)
    if device == "gpu":
        try:
            hands = create_landmarker(
                cfg, model_path, python.BaseOptions.Delegate.GPU)
        except RuntimeError as exc:
            print(f"GPU-delegate niet beschikbaar ({exc}); valt terug op CPU.")
            device = "cpu"
            hands = create_landmarker(
                cfg, model_path, python.BaseOptions.Delegate.CPU)
    else:
        hands = create_landmarker(
            cfg, model_path, python.BaseOptions.Delegate.CPU)
    print(f"MediaPipe draait op {device.upper()}")

    video = cv2.VideoCapture(cfg["video_path"])
    fps = video.get(cv2.CAP_PROP_FPS) or 30.0
    stride = cfg.get("frame_stride", 1)
    rows, frame_idx = [], 0

    with hands:
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        progress = tqdm(total=total_frames if total_frames > 0 else None,
                        desc="MediaPipe", unit="frame")
        while True:
            ok, frame = video.read()
            if not ok:
                break
            progress.update(1)
            if frame_idx % stride == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                timestamp_ms = int(frame_idx * 1000 / fps)
                res = hands.detect_for_video(image, timestamp_ms)
                if res.hand_landmarks:
                    for h_idx, landmarks in enumerate(res.hand_landmarks):
                        for p_idx, point in enumerate(landmarks):
                            rows.append({
                                "video": cfg["video_path"], "frame": frame_idx,
                                "hand": h_idx, "landmark": p_idx,
                                "x": round(point.x, 4), "y": round(point.y, 4),
                                "z": round(point.z, 4), "zichtbaar": 1.0,
                            })
                    if frame_idx % cfg["screenshot_every"] == 0:
                        annotated = frame.copy()
                        for landmarks in res.hand_landmarks:
                            draw_landmarks(annotated, landmarks)
                        cv2.imwrite(
                            str(out_dir / f"hand_frame_{frame_idx:05d}.jpg"),
                            annotated,
                        )
            frame_idx += 1
        progress.close()

    pd.DataFrame(rows).to_csv(out_dir / "hand_landmarks.csv", index=False)
    video.release()
    print(f"{frame_idx} frames verwerkt -> {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--device", choices=["cpu", "gpu"],
                        help="cpu of gpu; overschrijft 'device' uit de config")
    args = parser.parse_args()
    main(args.config, args.device)