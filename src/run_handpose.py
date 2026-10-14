"""MediaPipe Hands op een video; logt 21 landmerken per hand naar CSV."""
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from duplo.cli import parse_args
from duplo.config import load_config, resolve_device
from duplo.output import prepare_output_dir, save_screenshot, write_rows_csv
from duplo.video import VideoFrames, video_fps


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
CSV_COLUMNS = ["video", "frame", "hand", "landmark", "x", "y", "z", "zichtbaar"]


def get_model_path(cfg: dict) -> Path:
    model_path = Path(cfg.get("hand_model_path", "models/hand_landmarker.task"))
    if model_path.exists() and model_path.stat().st_size > 0:
        return model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Handmodel downloaden naar {model_path}...")
    # Eerst naar een tijdelijk bestand downloaden, zodat een afgebroken
    # download geen half model op de definitieve plek achterlaat.
    temp_path = model_path.with_name(f"{model_path.name}.part")
    try:
        urlretrieve(MODEL_URL, temp_path)
        temp_path.replace(model_path)
    finally:
        temp_path.unlink(missing_ok=True)
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
    cfg = load_config(cfg_path)
    out_dir = prepare_output_dir(cfg, "handpose")

    device = resolve_device(device, cfg)
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

    fps = video_fps(cfg["video_path"])
    screenshot_every = cfg.get("screenshot_every", 150)
    rows = []
    frames = VideoFrames(cfg["video_path"], stride=cfg.get("frame_stride", 1),
                         desc="MediaPipe")
    with hands:
        for frame_idx, frame in frames:
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
                if screenshot_every > 0 and frame_idx % screenshot_every == 0:
                    annotated = frame.copy()
                    for landmarks in res.hand_landmarks:
                        draw_landmarks(annotated, landmarks)
                    save_screenshot(out_dir, "hand_frame", frame_idx, annotated)

    write_rows_csv(out_dir, "hand_landmarks.csv", rows, CSV_COLUMNS)
    print(f"{frames.frames_read} frames verwerkt -> {out_dir}")


if __name__ == "__main__":
    args = parse_args()
    main(args.config, args.device)