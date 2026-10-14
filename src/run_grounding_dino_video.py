"""Grounding DINO (open-vocabulary) op een video; logt tool-detecties naar CSV.

Grounding DINO krijgt de gereedschapsnamen uit ``tool_classes`` als
tekstprompt ("screwdriver. wrench. ..."), zodat geen finetuning nodig is.
"""
import time

import cv2

from duplo.benchmark import InferenceTimer, write_run_summary
from duplo.cli import parse_args
from duplo.config import load_config, resolve_device
from duplo.gpu import ensure_cuda_torch
from duplo.output import prepare_output_dir, save_screenshot, write_rows_csv
from duplo.video import VideoFrames, video_fps
from duplo.video_writer import DebugVideoWriter

CSV_COLUMNS = ["video", "frame", "timestamp", "klasse", "confidence",
               "bbox_x", "bbox_y", "bbox_w", "bbox_h"]


def build_prompt(cfg: dict) -> str:
    """Tekstprompt voor Grounding DINO; klassen worden met punten gescheiden."""
    if cfg.get("grounding_dino_prompt"):
        return cfg["grounding_dino_prompt"]
    return ". ".join(cfg["tool_classes"]) + "."


def normalize_label(label: str, tool_classes: list[str]) -> str:
    """Koppel een DINO-label aan de promptklasse die erin voorkomt.

    Grounding DINO levert soms samengestelde labels ("screwdriver pliers")
    wanneer meerdere prompts op hetzelfde object passen; in de CSV staat dan
    de eerstgenoemde gereedschapsklasse. Soms bestaat een label uit een
    deelwoord ("screw", "##driver", "pl"); die woorden worden aan de
    gereedschapsklasse gekoppeld waar ze het begin of het eind van zijn.
    """
    text = str(label).strip().lower()
    if text in tool_classes:
        return text
    contained = [(text.find(tool), tool) for tool in tool_classes if tool in text]
    if contained:
        return min(contained)[1]
    token = text.removeprefix("##")
    if not token:
        return text
    partial = [tool for tool in tool_classes
               if tool.startswith(token) or tool.endswith(token)]
    return partial[0] if partial else text


def build_model(cfg: dict, device: str):
    """Grounding DINO uit de Hugging Face-hub; pas na een CUDA-installatie laden."""
    import torch
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    model_id = cfg.get("grounding_dino_model", "IDEA-Research/grounding-dino-base")
    torch_device = "cuda" if device == "gpu" else "cpu"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
    model.to(torch_device).eval()
    return model_id, processor, model, torch_device


def draw_detections(frame, result, tool_classes: list[str]):
    """Kopie van het frame met bboxes en labels voor de debug-video."""
    annotated = frame.copy()
    for box, confidence, label in zip(result["boxes"], result["scores"],
                                      result["labels"]):
        x1, y1, x2, y2 = (int(round(float(value))) for value in box)
        text = f"{normalize_label(label, tool_classes)} {float(confidence):.2f}"
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
    import torch

    model_id, processor, model, torch_device = build_model(cfg, device)

    tool_classes = list(cfg["tool_classes"])
    prompt = build_prompt(cfg)
    box_threshold = cfg["conf_threshold"]
    text_threshold = cfg.get("grounding_dino_text_threshold", 0.25)
    print(f"Grounding DINO ({model_id}) draait op {torch_device.upper()} "
          f"met prompt: {prompt}")

    fps = video_fps(cfg["video_path"])
    stride = cfg.get("frame_stride", 1)
    screenshot_every = cfg.get("screenshot_every", 150)
    timer = InferenceTimer()
    video = (DebugVideoWriter(out_dir / "grounding_dino_debug.mp4", fps / stride)
             if cfg.get("save_grounding_dino_video", True) else None)
    rows = []
    frames = VideoFrames(cfg["video_path"], stride=stride, desc="Grounding DINO")
    started = time.perf_counter()
    try:
        for frame_idx, frame in frames:
            # De processor verwacht RGB-beelden; OpenCV levert BGR.
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            with timer:
                inputs = processor(images=image, text=prompt,
                                   return_tensors="pt").to(torch_device)
                with torch.no_grad():
                    outputs = model(**inputs)
                result = processor.post_process_grounded_object_detection(
                    outputs, inputs.input_ids, threshold=box_threshold,
                    text_threshold=text_threshold,
                    target_sizes=[image.shape[:2]])[0]
            for box, confidence, label in zip(result["boxes"], result["scores"],
                                              result["labels"]):
                x1, y1, x2, y2 = (int(round(float(value))) for value in box)
                rows.append({
                    "video": cfg["video_path"], "frame": frame_idx,
                    "timestamp": round(frame_idx / fps, 3),
                    "klasse": normalize_label(label, tool_classes),
                    "confidence": round(float(confidence), 3),
                    "bbox_x": x1, "bbox_y": y1,
                    "bbox_w": x2 - x1, "bbox_h": y2 - y1,
                })
            annotated = draw_detections(frame, result, tool_classes)
            if video is not None:
                video.write(annotated)
            if screenshot_every > 0 and frame_idx % screenshot_every == 0:
                save_screenshot(out_dir, "grounding_dino_frame", frame_idx,
                                annotated)
    finally:
        # Ook bij een onderbreking de debug-video netjes afsluiten.
        if video is not None:
            video.release()

    write_rows_csv(out_dir, "grounding_dino_detections.csv", rows, CSV_COLUMNS)
    summary = write_run_summary(
        out_dir, "grounding_dino", model=model_id, device=device,
        video=cfg["video_path"], frames_read=frames.frames_read,
        frames_processed=len(timer.times_ms),
        elapsed_s=time.perf_counter() - started, rows=rows, timer=timer,
        csv_file="grounding_dino_detections.csv",
        extra={"conf_threshold": box_threshold, "frame_stride": stride,
               "tekstprompt": prompt})
    print(f"{frames.frames_read} frames verwerkt, {len(rows)} detections, "
          f"{summary['inference_fps']} fps -> {out_dir}")


if __name__ == "__main__":
    args = parse_args()
    main(args.config, args.device)
