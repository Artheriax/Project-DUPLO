from pathlib import Path
from datetime import datetime
from time import perf_counter
import hashlib
import json
import platform
import re
import sys

import cv2
import numpy as np
import torch
import transformers
from PIL import Image
from transformers import AutoProcessor, SmolVLMForConditionalGeneration


MODEL = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
VIDEO = Path("/Users/doahismail/Downloads/testvidDUPLOdm.mp4")
START = 223.0
END = 230.0
N_FRAMES = 8

LABELS = {
    "A1": {
        "nl": "Gereedschap pakken of wisselen",
        "en": "A person picks up or switches a hand tool.",
    },
    "A2": {
        "nl": "Product positioneren",
        "en": "A person repositions or rotates the product.",
    },
    "A3": {
        "nl": "Losschroeven",
        "en": "A person loosens a screw using a screwdriver.",
    },
    "A4": {
        "nl": "Kabel of connector loskoppelen",
        "en": "A person disconnects a cable connector.",
    },
    "A5": {
        "nl": "Component verwijderen",
        "en": "A person removes a component from the product.",
    },
}


def main():
    torch.set_num_threads(4)
    torch.manual_seed(0)

    if not VIDEO.is_file():
        raise FileNotFoundError(f"Video niet gevonden: {VIDEO}")

    print("Beelden uit 03:43–03:50 lezen...", flush=True)
    cap = cv2.VideoCapture(str(VIDEO))
    images = []

    try:
        if not cap.isOpened():
            raise RuntimeError("De video kan niet worden geopend.")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if not np.isfinite(fps) or fps <= 0:
            raise RuntimeError("Ongeldige FPS.")

        first = int(START * fps)
        last = int(END * fps) - 1

        if first < 0 or last >= total_frames or last < first:
            raise RuntimeError("Het fragment valt buiten de video.")

        indices = np.linspace(first, last, N_FRAMES, dtype=int)

        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Beeld {index} kon niet worden gelezen.")

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            images.append(Image.fromarray(rgb))
    finally:
        cap.release()

    print(f"Leesbare beelden: {len(images)} van {N_FRAMES}", flush=True)

    prompt = (
        "These images are sampled in chronological order from one "
        "seven-second video clip. Select the single action that best "
        "describes the action performed during this clip.\n\n"
        + "\n".join(
            f"{code}: {label['en']}" for code, label in LABELS.items()
        )
        + "\n\nReply with only one code: A1, A2, A3, A4, or A5."
    )

    print("Processor en model uit lokale cache laden...", flush=True)
    processor = AutoProcessor.from_pretrained(
        MODEL, local_files_only=True
    )
    model = SmolVLMForConditionalGeneration.from_pretrained(
        MODEL,
        local_files_only=True,
        use_safetensors=True,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    ).to("cpu").eval()

    content = []
    for index in indices:
        content.append({
            "type": "text",
            "text": f"Time within clip: {index / fps - START:.2f} seconds.",
        })
        content.append({"type": "image"})
    content.append({"type": "text", "text": prompt})

    messages = [{"role": "user", "content": content}]
    chat_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    print("Beelden voorbereiden...", flush=True)

    processor.do_image_splitting = False
    processor.image_size = {"longest_edge": 384}

    inputs = processor(
        text=chat_text,
        images=images,
        return_tensors="pt",
    )

    print("SmolVLM2 rekent op CPU; wacht op de uitslag...", flush=True)
    started = perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=16,
            use_cache=True,
        )
    seconds = perf_counter() - started

    # Alleen de nieuw gegenereerde tekst uitlezen, zonder de vraag.
    new_tokens = generated[:, inputs["input_ids"].shape[1]:]
    answer = processor.batch_decode(
        new_tokens, skip_special_tokens=True
    )[0].strip()

    # Een afwijkend antwoord niet stilzwijgend omzetten naar een label.
    match = re.fullmatch(r"(A[1-5])[.!]?", answer)
    prediction = match.group(1) if match else None
    label_nl = LABELS[prediction]["nl"] if prediction else None

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = Path(__file__).resolve().parent / "outputs" / f"smolvlm_{stamp}"
    output.mkdir(parents=True, exist_ok=False)

    # Bewaar ook de geselecteerde bronbeelden voor controle achteraf.
    frames_dir = output / "frames"
    frames_dir.mkdir()
    for number, image in enumerate(images):
        image.save(frames_dir / f"frame_{number:02d}.png")

    with VIDEO.open("rb") as handle:
        video_hash = hashlib.file_digest(handle, "sha256").hexdigest()

    result = {
        "model": MODEL,
        "model_revision": getattr(model.config, "_commit_hash", None),
        "video": str(VIDEO),
        "video_sha256": video_hash,
        "start_seconds": START,
        "end_seconds": END,
        "fps": fps,
        "number_of_frames": N_FRAMES,
        "frame_indices": indices.tolist(),
        "frame_times_seconds": [float(i / fps) for i in indices],
        "input_method": "Chronological multi-image input, without audio",
        "preprocessing": {
            "do_image_splitting": False,
            "longest_edge": 384,
        },
        "labels": LABELS,
        "prompt": prompt,
        "formatted_chat": chat_text,
        "device": str(next(model.parameters()).device),
        "dtype": str(next(model.parameters()).dtype),
        "threads": torch.get_num_threads(),
        "attention": "eager",
        "do_sample": False,
        "max_new_tokens": 16,
        "raw_answer": answer,
        "predicted_code": prediction,
        "predicted_label_nl": label_nl,
        "generation_seconds": seconds,
        "timing_scope": "model.generate only; excludes loading and preprocessing",
        "scores": None,
        "scores_note": "Text generation; no five-class scores calculated.",
        "python": sys.version,
        "macOS": platform.mac_ver(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "input_shapes": {key: list(value.shape) for key, value in inputs.items()},
        "limitations": [
            "One clip; not an accuracy evaluation.",
            "Eight frames versus 32 in the X-CLIP test.",
            "Five choices; no other/unknown category.",
            "No independently confirmed reference label.",
        ],
    }

    (output / "resultaat.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = (
        "SmolVLM2 — technische proef\n"
        "Fragment: 03:43–03:50\n"
        f"Ruw modelantwoord: {answer}\n"
        f"Actie: {prediction} — {label_nl}\n"
        f"Rekentijd generatie: {seconds:.1f} seconden\n"
        "CPU, float32, 8 beelden in tijdsvolgorde.\n"
        "Dit is geen meting van nauwkeurigheid.\n"
    )
    (output / "samenvatting.txt").write_text(summary, encoding="utf-8")

    print("\n" + summary)
    if prediction is None:
        print("Het antwoord volgt niet het gevraagde formaat; controleer het.")
    print("Resultaten opgeslagen in:", output)


if __name__ == "__main__":
    main()