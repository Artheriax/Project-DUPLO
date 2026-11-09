"""DUPLO: lokale X-CLIP CPU-proef; slaat iedere nieuwe run apart op."""
import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

MODEL = 'microsoft/xclip-base-patch16-zero-shot'
LABELS = [
    ('A1', 'Gereedschap pakken of wisselen', 'A person picks up or switches a hand tool.'),
    ('A2', 'Product positioneren', 'A person repositions or rotates the product.'),
    ('A3', 'Losschroeven', 'A person loosens a screw using a screwdriver.'),
    ('A4', 'Kabel of connector loskoppelen', 'A person disconnects a cable connector.'),
    ('A5', 'Component verwijderen', 'A person removes a component from the product.'),
]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', default=str(Path.home() / 'Downloads/testvidDUPLOdm.mp4'))
    parser.add_argument('--start', type=float, default=223)
    parser.add_argument('--end', type=float, default=230)
    parser.add_argument('--revision', default=None, help='Optioneel: exact lokaal gecachet modelcommit')
    args = parser.parse_args()
    if not 0 <= args.start < args.end:
        parser.error('Gebruik 0 <= start < end (seconden).')
    path = Path(args.video).expanduser()
    if not path.is_file():
        parser.error(f'Video niet gevonden: {path}')

    import cv2
    import numpy as np
    import torch
    from transformers import XCLIPModel, XCLIPProcessor
    torch.set_num_threads(4)
    options = {'local_files_only': True}
    if args.revision:
        options['revision'] = args.revision
    print('Model laden uit lokale cache...', flush=True)
    processor = XCLIPProcessor.from_pretrained(MODEL, **options)
    model = XCLIPModel.from_pretrained(MODEL, use_safetensors=True, **options).to('cpu').eval()
    count = model.config.vision_config.num_frames
    cap = cv2.VideoCapture(str(path))
    frames = []
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not cap.isOpened() or fps <= 0:
            raise RuntimeError('Video niet leesbaar.')
        if total > 0 and args.end * fps > total:
            raise RuntimeError('Eindtijd ligt voorbij het einde volgens de videometadata.')
        indices = np.linspace(int(args.start * fps), int(args.end * fps) - 1, count, dtype=int)
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f'Frame {index} niet leesbaar.')
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()

    # Zelfde gezamenlijke processor-aanroep als de oorspronkelijke terminalproef.
    # De gemelde padding-/tensorwaarschuwingen kunnen daarom opnieuw verschijnen.
    inputs = processor(text=[x[2] for x in LABELS], videos=frames,
                       return_tensors='pt', padding=True)
    print('Model rekent op CPU...', flush=True)
    start_time = time.perf_counter()
    with torch.inference_mode():
        output = model(**inputs)
    seconds = time.perf_counter() - start_time
    scores = output.logits_per_video[0].cpu().tolist()
    if not all(np.isfinite(scores)):
        raise RuntimeError('Model gaf ongeldige scores.')
    rows = sorted([dict(code=x[0], label_nl=x[1], prompt_en=x[2], score=s)
                   for x, s in zip(LABELS, scores)], key=lambda x: x['score'], reverse=True)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    report = dict(
        provenance='Nieuwe uitvoering van test_xclip.py op de computer van de uitvoerder',
        recorded_at=datetime.now().astimezone().isoformat(), model=MODEL,
        model_revision=getattr(model.config, '_commit_hash', None),
        requested_revision=args.revision, device='cpu', torch_threads=4,
        dtype=str(next(model.parameters()).dtype),
        python=sys.version, macos=platform.mac_ver()[0], architecture=platform.machine(),
        packages={p: version(p) for p in ['torch', 'torchvision', 'transformers', 'numpy', 'opencv-python']},
        video_filename=path.name, video_sha256=digest.hexdigest(),
        start_seconds=args.start, end_seconds_exclusive=args.end,
        fps=fps, frame_indices=indices.tolist(), frame_count=len(frames),
        inference_seconds=seconds, timing_scope='Alleen model-forward; exclusief laden, decoderen en preprocessing',
        scores=rows, predicted_label=rows[0]['code'], ground_truth=None,
        limitation='Gesloten keuze uit vijf acties; geen gekalibreerde kansen en geen accuracymeting.',
    )
    folder = Path(__file__).resolve().parent / 'outputs' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    folder.mkdir(parents=True, exist_ok=False)
    (folder / 'resultaat.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    with (folder / 'scores.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=['code', 'label_nl', 'prompt_en', 'score'])
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"{row['code']} — {row['label_nl']}: {row['score']:.3f}")
    print(f'Rekentijd model: {seconds:.1f} seconden')
    print(f'Resultaten opgeslagen in: {folder}')

if __name__ == '__main__':
    main()
