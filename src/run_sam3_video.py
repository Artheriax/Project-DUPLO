"""SAM3 (Segment Anything 3) op een video; logt tool-segmentaties naar CSV.

SAM3 segmenteert elk object dat bij een tekstprompt hoort (open-vocabulary),
zonder finetuning. De gewichten van ``facebook/sam3`` staan achter een
Hugging Face-toegangsaanvraag: vraag die aan op
https://huggingface.co/facebook/sam3 en log daarna in met ``hf auth login``.
Op Windows is daarnaast het community-pakket ``triton-windows`` nodig, omdat
sam3 ``triton`` importeert en het officiële pakket geen Windows-wheels heeft.
"""
import time
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import numpy as np
from PIL import Image

from duplo.benchmark import InferenceTimer, write_run_summary
from duplo.cli import parse_args
from duplo.config import load_config, resolve_device
from duplo.gpu import ensure_cuda_torch
from duplo.output import prepare_output_dir, save_screenshot, write_rows_csv
from duplo.video import VideoFrames, video_fps
from duplo.video_writer import DebugVideoWriter

CSV_COLUMNS = ["video", "frame", "timestamp", "klasse", "confidence",
               "bbox_x", "bbox_y", "bbox_w", "bbox_h", "mask_area"]
BPE_FILENAME = "bpe_simple_vocab_16e6.txt.gz"
BPE_URL = ("https://raw.githubusercontent.com/facebookresearch/sam3/main/"
           f"sam3/assets/{BPE_FILENAME}")


def get_bpe_path(cfg: dict, package_dir: Path) -> str:
    """Pad naar het BPE-vocabulaire dat de tekstencoder van SAM3 nodig heeft.

    Het pip-pakket bevat dit bestand niet, terwijl sam3 het op een ander pad
    zoekt; daarom halen we het eenmalig uit de sam3-repo (net als het
    handmodel in run_handpose.py).
    """
    local_path = Path(cfg.get("sam3_bpe_path", f"models/{BPE_FILENAME}"))
    for candidate in (local_path, package_dir / "assets" / BPE_FILENAME):
        if candidate.exists() and candidate.stat().st_size > 0:
            return str(candidate)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"BPE-vocabulaire downloaden naar {local_path}...")
    # Eerst naar een tijdelijk bestand downloaden, zodat een afgebroken
    # download geen half bestand op de definitieve plek achterlaat.
    temp_path = local_path.with_name(f"{local_path.name}.part")
    try:
        urlretrieve(BPE_URL, temp_path)
        temp_path.replace(local_path)
    finally:
        temp_path.unlink(missing_ok=True)
    return str(local_path)


def build_processor(cfg: dict, device: str):
    """SAM3-model en processor; pas importeren na een CUDA-installatie."""
    try:
        import sam3
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model
    except ImportError as exc:
        raise SystemExit(
            "Het sam3-pakket (of een afhankelijkheid zoals triton) is niet "
            "beschikbaar. Installeer 'sam3==0.1.4' (verlaagt numpy naar <2.4) "
            "en op Windows 'triton-windows' (het officiële triton-pakket heeft "
            f"geen Windows-wheels). Details: {exc}"
        ) from exc
    try:
        model = build_sam3_image_model(
            bpe_path=get_bpe_path(cfg, Path(sam3.__file__).parent),
            checkpoint_path=cfg.get("sam3_checkpoint") or None,
            device="cuda" if device == "gpu" else "cpu")
    except Exception as exc:
        raise SystemExit(
            "SAM3 kon niet worden geladen. De checkpoints van facebook/sam3 "
            "zijn alleen na een goedgekeurde Hugging Face-toegangsaanvraag te "
            "downloaden; vraag toegang aan, log in met 'hf auth login' of zet "
            f"een lokaal 'sam3_checkpoint' in de config. Details: {exc}"
        ) from exc
    return Sam3Processor(
        model, confidence_threshold=cfg["conf_threshold"],
        device="cuda" if device == "gpu" else "cpu")


def mask_pixels(mask) -> np.ndarray:
    """Masker als 2D-boolean array voor de overlay en het oppervlak.

    SAM3 levert het masker als CUDA-tensor (soms met een extra as); numpy
    heeft een array op de CPU nodig.
    """
    if hasattr(mask, "detach"):  # torch.Tensor op de GPU
        mask = mask.detach().cpu().numpy()
    array = np.asarray(mask)
    return array.reshape(array.shape[-2], array.shape[-1]) > 0


def draw_segmentations(frame, segmentations):
    """Kopie van het frame met maskers, bboxes en labels voor de debug-video."""
    annotated = frame.copy()
    overlay = annotated.copy()
    for mask, box, klasse, confidence in segmentations:
        overlay[mask] = (0, 200, 0)
        x1, y1, x2, y2 = (int(round(float(value))) for value in box)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, f"{klasse} {confidence:.2f}",
                    (x1, max(y1 - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 2)
    return cv2.addWeighted(overlay, 0.4, annotated, 0.6, 0)


def main(cfg_path: str, device: str | None = None) -> None:
    cfg = load_config(cfg_path)
    out_dir = prepare_output_dir(cfg, "sam3")

    device = resolve_device(device, cfg)
    if device == "gpu":
        ensure_cuda_torch()
    processor = build_processor(cfg, device)
    prompts = list(cfg.get("sam3_prompts") or cfg["tool_classes"])
    print(f"SAM3 draait op {device.upper()} met prompts: {', '.join(prompts)}")

    fps = video_fps(cfg["video_path"])
    stride = cfg.get("frame_stride", 1)
    screenshot_every = cfg.get("screenshot_every", 150)
    timer = InferenceTimer()
    video = (DebugVideoWriter(out_dir / "sam3_debug.mp4", fps / stride)
             if cfg.get("save_sam3_video", True) else None)
    rows = []
    frames = VideoFrames(cfg["video_path"], stride=stride, desc="SAM3")
    started = time.perf_counter()
    try:
        for frame_idx, frame in frames:
            # SAM3 verwacht RGB en leest de beeldmaten alleen goed uit een
            # PIL-image; bij een numpy-array gaat het model van CHW uit.
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            segmentations = []
            with timer:
                state = processor.set_image(image)
                for prompt in prompts:
                    output = processor.set_text_prompt(prompt, state)
                    for mask, box, confidence in zip(output["masks"],
                                                     output["boxes"],
                                                     output["scores"]):
                        segmentations.append((mask_pixels(mask), box,
                                              prompt, float(confidence)))
            for mask, box, klasse, confidence in segmentations:
                x1, y1, x2, y2 = (int(round(float(value))) for value in box)
                rows.append({
                    "video": cfg["video_path"], "frame": frame_idx,
                    "timestamp": round(frame_idx / fps, 3),
                    "klasse": klasse,
                    "confidence": round(confidence, 3),
                    "bbox_x": x1, "bbox_y": y1,
                    "bbox_w": x2 - x1, "bbox_h": y2 - y1,
                    "mask_area": int(mask.sum()),
                })
            annotated = draw_segmentations(frame, segmentations)
            if video is not None:
                video.write(annotated)
            if screenshot_every > 0 and frame_idx % screenshot_every == 0:
                save_screenshot(out_dir, "sam3_frame", frame_idx, annotated)
    finally:
        # Ook bij een onderbreking de debug-video netjes afsluiten.
        if video is not None:
            video.release()

    write_rows_csv(out_dir, "sam3_detections.csv", rows, CSV_COLUMNS)
    summary = write_run_summary(
        out_dir, "sam3", model=cfg.get("sam3_model", "facebook/sam3"),
        device=device, video=cfg["video_path"],
        frames_read=frames.frames_read, frames_processed=len(timer.times_ms),
        elapsed_s=time.perf_counter() - started, rows=rows, timer=timer,
        csv_file="sam3_detections.csv",
        extra={"conf_threshold": cfg["conf_threshold"], "frame_stride": stride,
               "tekstprompt": prompts})
    print(f"{frames.frames_read} frames verwerkt, {len(rows)} segmentaties, "
          f"{summary['inference_fps']} fps -> {out_dir}")


if __name__ == "__main__":
    args = parse_args()
    main(args.config, args.device)
