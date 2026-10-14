"""Detectoren als zelfstandige pipelinecomponenten (YOLO, Grounding DINO en SAM3)."""
from dataclasses import dataclass, field
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import numpy as np

BPE_FILENAME = "bpe_simple_vocab_16e6.txt.gz"
BPE_URL = ("https://raw.githubusercontent.com/facebookresearch/sam3/main/"
           f"sam3/assets/{BPE_FILENAME}")


def _load_yolo_model(YOLO, cfg: dict, model_key: str, torch_device: str):
    """Laad een YOLO-model, optioneel als TensorRT-engine (opt-in).

    ``yolo_backend: tensorrt`` zoekt naast het ``.pt``-bestand een ``.engine``
    (zelfde naam) en exporteert die eenmalig met FP16 bij dezelfde standaard-
    imgsz. Elke fout (geen TensorRT, geen CUDA-toolkit, Windows-beperking)
    valt stil terug op het ``.pt``-bestand, zodat een run nooit breekt.
    Drempels en imgsz blijven ongewijzigd: alleen de kernel-implementatie
    verandert.
    """
    model_path = str(cfg[model_key])
    backend = str(cfg.get("yolo_backend", "torch")).strip().lower()
    if backend != "tensorrt" or torch_device != "cuda":
        return YOLO(model_path)
    try:
        engine_path = Path(model_path).with_suffix(".engine")
        if engine_path.exists() and engine_path.stat().st_size > 0:
            print(f"TensorRT-engine laden: {engine_path}")
            return YOLO(str(engine_path))
        print(f"TensorRT-engine exporteren naar {engine_path} "
              "(eenmalig, kan enkele minuten duren)...")
        base = YOLO(model_path)
        exported = base.export(format="engine", half=True, device=0)
        # export() geeft het engine-pad terug (str of lijst); laad het resultaat.
        if isinstance(exported, (list, tuple)):
            exported = exported[0] if exported else str(engine_path)
        print(f"TensorRT-engine klaar: {exported}")
        return YOLO(str(exported))
    except Exception as exc:
        print(f"Waarschuwing: TensorRT-fallback naar {model_path} ({exc}). "
              "Installeer 'tensorrt' + CUDA-toolkit om de engine te gebruiken.")
        return YOLO(model_path)


@dataclass(frozen=True)
class Detection:
    """Eén objectdetectie met bbox in pixelcoördinaten (x1, y1, x2, y2).

    ``mask`` is alleen gevuld bij segmentatiemodellen; het masker is 2D
    (hoogte × breedte) en wordt niet in de vergelijking van detecties
    meegenomen.
    """

    bbox: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str
    mask: np.ndarray | None = field(default=None, compare=False)


class YoloDetector:
    """Wrapper rond een ultralytics YOLO-model.

    Los te vervangen component: de rest van de pipeline werkt met Detection
    objecten en is niet afhankelijk van ultralytics. Met ``model_key`` is een
    ander YOLO-model uit de config te kiezen (bijvoorbeeld
    ``yolo_world_model``); ``classes`` zet dan de open-vocabulary klassen
    (YOLO-World) via de tekstprompt.

    Snelheid (detectie-neutraal: zelfde imgsz/drempels/gewichten):
    ``yolo_fuse`` voegt Conv+BN samen, ``yolo_half`` gebruikt FP16 op CUDA en
    ``yolo_backend: tensorrt`` laadt een eenmalig geëxporteerde ``.engine``
    (valt automatisch terug op ``.pt`` als TensorRT ontbreekt).
    """

    def __init__(self, cfg: dict, device: str = "cpu",
                 model_key: str = "yolo_model",
                 classes: list[str] | None = None) -> None:
        # Pas importeren na een eventuele CUDA-torch-installatie, anders draait
        # een reeds geladen CPU-build.
        from ultralytics import YOLO

        from duplo.gpu import setup_cuda_optimizations

        setup_cuda_optimizations(device)
        self.device = "cuda" if device == "gpu" else "cpu"
        self.conf_threshold = cfg["conf_threshold"]
        self.use_half = bool(cfg.get("yolo_half", True)) and self.device == "cuda"
        self.model = _load_yolo_model(YOLO, cfg, model_key, self.device)
        if cfg.get("yolo_fuse", True):
            try:
                self.model.fuse(verbose=False)
            except Exception as exc:
                print(f"Waarschuwing: YOLO-fuse overgeslagen ({exc}).")
        if classes:
            # YOLO-World vervangt hiermee de COCO-klassen door de tekstprompt.
            self.model.set_classes(list(classes))
        names = self.model.names
        self.class_names = (dict(enumerate(names)) if isinstance(names, list)
                            else dict(names))

    def detect(self, frame: np.ndarray,
               conf: float | None = None) -> list[Detection]:
        """Detecteer objecten; conf overschrijft tijdelijk de config-drempel."""
        result = self.model.predict(
            frame, conf=self.conf_threshold if conf is None else conf,
            device=self.device, verbose=False,
            quantize=16 if self.use_half else None, augment=False,
        )[0]
        detections = []
        for box in result.boxes:
            x1, y1, x2, y2 = (float(value) for value in box.xyxy[0])
            class_id = int(box.cls)
            detections.append(Detection(
                bbox=(x1, y1, x2, y2),
                confidence=float(box.conf),
                class_id=class_id,
                class_name=self.class_names[class_id],
            ))
        return detections


def normalize_label(label: str, tool_classes: list[str]) -> str:
    """Koppel een Grounding DINO-label aan de promptklasse die erin voorkomt.

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


class GroundingDinoDetector:
    """Wrapper rond Grounding DINO (Hugging Face transformers).

    Open-vocabulary: de gereedschapsnamen uit ``tool_classes`` (of een eigen
    ``grounding_dino_prompt``) gaan als tekstprompt mee, zodat geen finetuning
    nodig is. De rest van de pipeline werkt met Detection-objecten.
    """

    def __init__(self, cfg: dict, device: str = "cpu") -> None:
        from transformers import (AutoModelForZeroShotObjectDetection,
                                   AutoProcessor)

        from duplo.gpu import setup_cuda_optimizations

        setup_cuda_optimizations(device)
        self.model_id = cfg.get("grounding_dino_model",
                                "IDEA-Research/grounding-dino-base")
        self.torch_device = "cuda" if device == "gpu" else "cpu"
        self.use_amp = bool(cfg.get("autocast_fp16", True)) and self.torch_device == "cuda"
        self.tool_classes = [str(tool).strip().lower()
                             for tool in cfg["tool_classes"]]
        self.prompt = (cfg.get("grounding_dino_prompt")
                       or ". ".join(self.tool_classes) + ".")
        self.box_threshold = cfg.get("conf_threshold", 0.25)
        self.text_threshold = cfg.get("grounding_dino_text_threshold", 0.25)
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        # SDPA-aandacht is sneller op RTX-kaarten; valt terug op standaard.
        # (Grounding DINO ondersteunt geen SDPA: dan volgt automatisch eager.)
        try:
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
                self.model_id, attn_implementation="sdpa")
        except Exception:
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
                self.model_id)
        self.model.to(self.torch_device).eval()
        # De tekstprompt is statisch: tokeniseer eenmalig en hergebruik per
        # frame, zodat alleen het beeld nog verwerkt hoeft te worden.
        self._text_cache = None
        try:
            text_only = self.processor(text=self.prompt, return_tensors="pt")
            self._text_cache = {
                key: value for key, value in text_only.items()
                if key in ("input_ids", "attention_mask")}
            if not self._text_cache:
                self._text_cache = None
        except Exception as exc:
            print(f"Waarschuwing: prompt-cache overgeslagen ({exc}).")
            self._text_cache = None
        self.class_names = {index: name
                            for index, name in enumerate(self.tool_classes)}

    def detect(self, frame: np.ndarray,
               conf: float | None = None) -> list[Detection]:
        """Detecteer gereedschappen; conf overschrijft tijdelijk de drempel."""
        import torch
        from contextlib import nullcontext

        # De processor verwacht RGB-beelden; OpenCV levert BGR.
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self._text_cache is not None:
            try:
                inputs = self.processor(images=image, return_tensors="pt")
                for key, value in self._text_cache.items():
                    if key not in inputs:
                        inputs[key] = value
                inputs = inputs.to(self.torch_device)
            except Exception:
                # Val terug op de oorspronkelijke aanroep (beeld + tekst).
                inputs = self.processor(
                    images=image, text=self.prompt,
                    return_tensors="pt").to(self.torch_device)
        else:
            inputs = self.processor(images=image, text=self.prompt,
                                    return_tensors="pt").to(self.torch_device)
        amp = (torch.autocast("cuda", dtype=torch.float16)
               if self.use_amp else nullcontext())
        with torch.inference_mode(), amp:
            outputs = self.model(**inputs)
        result = self.processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            threshold=self.box_threshold if conf is None else conf,
            text_threshold=self.text_threshold,
            target_sizes=[image.shape[:2]])[0]
        detections = []
        # `text_labels` bevat de stringnamen; `labels` wordt integer-id's en
        # geeft een FutureWarning (transformers >= 4.51).
        label_key = "text_labels" if "text_labels" in result else "labels"
        for box, confidence, label in zip(result["boxes"], result["scores"],
                                          result[label_key]):
            x1, y1, x2, y2 = (float(value) for value in box)
            name = normalize_label(label, self.tool_classes)
            detections.append(Detection(
                bbox=(x1, y1, x2, y2),
                confidence=float(confidence),
                class_id=self._class_id(name),
                class_name=name,
            ))
        return detections

    def _class_id(self, name: str) -> int:
        """Klasse-index van een promptklasse; onbekende labels krijgen een eigen id."""
        for index, class_name in self.class_names.items():
            if class_name == name:
                return index
        index = len(self.class_names)
        self.class_names[index] = name
        return index


def get_bpe_path(cfg: dict, package_dir: Path) -> str:
    """Pad naar het BPE-vocabulaire dat de tekstencoder van SAM3 nodig heeft.

    Sommige versies van het pip-pakket bevatten dit bestand niet, terwijl sam3
    het op een ander pad zoekt; daarom halen we het eenmalig uit de sam3-repo
    (net als het handmodel in run_handpose.py).
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


def mask_pixels(mask) -> np.ndarray:
    """Masker als 2D-boolean array voor de overlay en het oppervlak.

    SAM3 levert het masker als CUDA-tensor (soms met een extra as); numpy
    heeft een array op de CPU nodig.
    """
    if hasattr(mask, "detach"):  # torch.Tensor op de GPU
        mask = mask.detach().cpu().numpy()
    array = np.asarray(mask)
    return array.reshape(array.shape[-2], array.shape[-1]) > 0


class Sam3Detector:
    """Wrapper rond het SAM3-beeldmodel met een tekstprompt per toolklasse.

    SAM3 segmenteert elk object dat bij de prompt hoort (open-vocabulary) en
    levert naast de bbox ook het masker mee. De gewichten van
    ``facebook/sam3`` vragen een goedgekeurde Hugging Face-toegangsaanvraag en
    op Windows het pakket ``triton-windows``; zie de README.
    """

    def __init__(self, cfg: dict, device: str = "cpu") -> None:
        try:
            import sam3
            from sam3.model.sam3_image_processor import Sam3Processor
            from sam3.model_builder import build_sam3_image_model
        except ImportError as exc:
            raise SystemExit(
                "Het sam3-pakket (of een afhankelijkheid zoals triton) is niet "
                "beschikbaar. Installeer 'sam3' (verlaagt numpy naar <2.4) en "
                "op Windows 'triton-windows' (het officiële triton-pakket "
                f"heeft geen Windows-wheels). Details: {exc}"
            ) from exc
        from duplo.gpu import setup_cuda_optimizations

        setup_cuda_optimizations(device)
        self.model_id = cfg.get("sam3_model", "facebook/sam3")
        self.device = "cuda" if device == "gpu" else "cpu"
        self.use_amp = bool(cfg.get("autocast_fp16", True)) and self.device == "cuda"
        self.conf_threshold = cfg.get("conf_threshold", 0.5)
        self.prompts = [str(prompt) for prompt in
                        (cfg.get("sam3_prompts") or cfg["tool_classes"])]
        self.class_names = {index: name
                            for index, name in enumerate(self.prompts)}
        try:
            self.model = build_sam3_image_model(
                bpe_path=get_bpe_path(cfg, Path(sam3.__file__).parent),
                checkpoint_path=cfg.get("sam3_checkpoint") or None,
                device=self.device)
        except Exception as exc:
            raise SystemExit(
                "SAM3 kon niet worden geladen. De checkpoints van "
                "facebook/sam3 zijn alleen na een goedgekeurde Hugging "
                "Face-toegangsaanvraag te downloaden; vraag toegang aan, log "
                "in met 'hf auth login' of zet een lokaal 'sam3_checkpoint' "
                f"in de config. Details: {exc}"
            ) from exc
        self.processor = Sam3Processor(
            self.model, confidence_threshold=self.conf_threshold,
            device=self.device)

    def detect(self, frame: np.ndarray,
               conf: float | None = None) -> list[Detection]:
        """Detecteer gereedschappen; conf overschrijft tijdelijk de drempel."""
        import torch
        from contextlib import nullcontext
        from PIL import Image

        # SAM3 verwacht RGB en leest de beeldmaten alleen goed uit een
        # PIL-image; bij een numpy-array gaat het model van CHW uit.
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if conf is not None:
            self.processor.set_confidence_threshold(conf)
        detections = []
        amp = (torch.autocast("cuda", dtype=torch.float16)
               if self.use_amp else nullcontext())
        try:
            with torch.inference_mode(), amp:
                state = self.processor.set_image(image)
                for class_id, prompt in enumerate(self.prompts):
                    output = self.processor.set_text_prompt(prompt, state)
                detections.extend(
                    Detection(
                        bbox=tuple(float(value) for value in box),
                        confidence=float(score),
                        class_id=class_id,
                        class_name=prompt,
                        mask=mask_pixels(mask),
                    )
                    for mask, box, score in zip(output["masks"],
                                                output["boxes"],
                                                output["scores"])
                )
        finally:
            if conf is not None:
                self.processor.set_confidence_threshold(self.conf_threshold)
        return detections
