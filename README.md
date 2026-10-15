# Project DUPLO

Proof of concept voor het automatisch analyseren van DUPLO-disassemblyvideo's. De scope van het project is **tool recognition** (gereedschappen herkennen en volgen) en **human action recognition** (menselijke handelingen herkennen). Het finetunen van YOLO tot een generiek disassemblymodel is te moeilijk en te tijdrovend gebleken en valt buiten dit project; in plaats daarvan gebruiken we pretrained modellen voor tooldetectie en richten we ons eigen modelleerwerk op de actieherkenning. Zie [Documentation/Opdracht.md](Documentation/Opdracht.md) en [Documentation/PLAN.md](Documentation/PLAN.md) voor het volledige kader.

## Projectstructuur

- `configs/` – instellingen per experiment (video, model, drempels, device)
- `data/` – invoervideo's (niet in git)
- `Documentation/` – [Opdracht.md](Documentation/Opdracht.md) (opdrachtomschrijving), [PLAN.md](Documentation/PLAN.md) (technisch plan) en [Modelvergelijking.md](Documentation/Modelvergelijking.md) (testrapport van de modelvergelijking)
- `models/` – gedownloade modelgewichten: `yolo26x.pt` en `hand_landmarker.task` (niet in git)
- `outputs/` – resultaten per video, per stap een eigen submap (`yolo/`, `handpose/`, `tracking/`, `rfdetr/`, `rfdetr_seg/`, `grounding_dino/`, `yolo_world/`, `yolo_seg/`, `sam3/`, `sam31/` en `comparison/`) met CSV's, voorbeeldframes en debug-video
- `src/duplo/` – gedeelde componenten: CLI, config, GPU-setup, video-inlezen, output, detectie, ByteTrack-tracking en de benchmark-samenvatting per modelstap
- `src/run_yolo_video.py` – YOLO-pipeline (detectie, in dit project gericht op gereedschappen)
- `src/run_handpose.py` – MediaPipe-pipeline (handlandmarks)
- `src/run_tracking.py` – YOLO + ByteTrack-pipeline (objecten volgen over frames)
- `src/run_rfdetr_video.py` – RF-DETR-detectie (COCO-gewichten)
- `src/run_rfdetr_seg_video.py` – RF-DETR-segmentatie (maskers)
- `src/run_grounding_dino_video.py` – Grounding DINO (open-vocabulary, tekstprompt met toolklassen)
- `src/run_yolo_world_video.py` – YOLO-World (open-vocabulary, tekstprompt met toolklassen)
- `src/run_yolo_seg_video.py` – YOLO26-segmentatie (maskers)
- `src/run_sam3_video.py` – SAM3-segmentatie (open-vocabulary, vereist Hugging Face-toegang)
- `src/run_sam31_video.py` – SAM 3.1-segmentatie via Ultralytics (open-vocabulary, `sam3.1_multiplex.pt`, vereist Hugging Face-toegang)
- `src/compare_models.py` – vergelijkt de modeltesten op tooldetectie en snelheid
- `run_yolo_test.bat` / `run_handpose_test.bat` / `run_tracking_test.bat` / `run_rfdetr_test.bat` / `run_rfdetr_seg_test.bat` / `run_grounding_dino_test.bat` / `run_yolo_world_test.bat` / `run_yolo_seg_test.bat` / `run_sam3_test.bat` / `run_sam31_test.bat` / `run_compare_models.bat` – startscripts voor Windows

## Eerste YOLO-test

Dubbelklik op `run_yolo_test.bat` vanuit de projectmap. Het script maakt `.venv` aan als die nog niet bestaat, installeert of controleert de packages uit `requirements.txt`, controleert de imports en voert daarna de YOLO-test uit op `data/Disassembly1.mp4`.

De eerste run downloadt automatisch het pretrained model `models/yolo26x.pt` als dat nog ontbreekt. De resultaten komen in `outputs/Disassembly1/yolo/`, met detections in `yolo_detections.csv`, voorbeeldframes als JPG-bestanden en een `yolo_debug.mp4` met de getekende bboxes en labels (uit te zetten met `save_yolo_video: false` in de config). Daarnaast schrijft de stap `yolo_summary.json` met de rekentijd per frame, zodat YOLO in `compare_models.py` als baseline meekan.

## Modelvergelijking: toolrecognition

Voor de toolrecognition zijn naast YOLO zes alternatieven toegevoegd met dezelfde opzet (video in, CSV + voorbeeldframes + debug-video uit). Elk model schrijft in een eigen submap van `outputs/Disassembly1/` en levert naast de CSV ook een `<model>_summary.json` met frames, detections en rekentijd:

| Startscript | Modelstap | Soort | Submap |
| --- | --- | --- | --- |
| `run_yolo_test.bat` | YOLO26x (baseline) | gesloten vocabulaire (80 COCO-klassen) | `yolo/` |
| `run_rfdetr_test.bat` | RF-DETR Large | gesloten vocabulaire (80 COCO-klassen) | `rfdetr/` |
| `run_grounding_dino_test.bat` | Grounding DINO base | open-vocabulary (tekstprompt) | `grounding_dino/` |
| `run_yolo_world_test.bat` | YOLO-World v2 X | open-vocabulary (tekstprompt) | `yolo_world/` |
| `run_yolo_seg_test.bat` | YOLO26x-seg | segmentatie | `yolo_seg/` |
| `run_rfdetr_seg_test.bat` | RF-DETR Seg 2XLarge | segmentatie | `rfdetr_seg/` |
| `run_sam3_test.bat` | SAM3 | segmentatie, open-vocabulary | `sam3/` |
| `run_sam31_test.bat` | SAM 3.1 (Object Multiplex-checkpoint) | segmentatie, open-vocabulary | `sam31/` |

De doelgereedschappen staan als `tool_classes` in `configs/baseline.yaml` (schroevendraaier, steeksleutel, tang, snijder, hamer). De open-vocabulary modellen krijgen die lijst als tekstprompt; de andere modellen gebruiken hun eigen COCO-klassen en kunnen dus alleen gereedschappen vinden die in COCO zitten. De segmentatiestappen loggen ook `mask_area` (maskeroppervlak in pixels).

De nieuwe stappen gebruiken dezelfde kolomopzet als de YOLO-test (`video`, `frame`, `klasse`, `confidence`, `bbox_x`, `bbox_y`, `bbox_w`, `bbox_h`), aangevuld met `timestamp` en bij segmentatie met `mask_area` (maskeroppervlak in pixels), zodat de resultaten onderling te vergelijken zijn.

**SAM3:** de gewichten van `facebook/sam3` staan achter een Hugging Face-toegangsaanvraag: vraag toegang aan op https://huggingface.co/facebook/sam3 en log daarna in met `hf auth login`. Op Windows is daarnaast `triton-windows` nodig (sam3 importeert `triton`; het officiële pakket heeft geen Windows-wheels) — beide packages staan in `requirements.txt`. Het pip-pakket mist het BPE-vocabulaire voor de tekstencoder; [run_sam3_video.py](src/run_sam3_video.py) haalt dat eenmalig uit de sam3-repo en zet het in `models/`.

**SAM 3.1:** de gewichten van `facebook/sam3.1` (`sam3.1_multiplex.pt`, Object Multiplex-checkpoint) staan achter een Hugging Face-toegangsaanvraag: vraag toegang aan op https://huggingface.co/facebook/sam3.1 en log daarna in met `huggingface-cli login` (of `hf auth login`). Daarna downloadt Ultralytics het checkpoint automatisch bij `run_sam31_test.bat`, of zet een lokaal bestand via `sam31_model` in `configs/baseline.yaml` (standaard `models/sam3.1_multiplex.pt`, niet in git — handmatig downloaden mag ook). De test gebruikt Ultralytics `SAM3SemanticPredictor` met de `tool_classes` als tekstprompt (per-frame beeldinferentie met maskers); Object Multiplex-videotracking wordt door Ultralytics nog niet ondersteund. Voor het native videomodel (gezamenlijk multi-object tracken met gedeeld geheugen, ~7x sneller bij veel objecten) is de originele repo https://github.com/facebookresearch/sam3 nodig (Python 3.12+, CUDA 12.6+, `sam3.1_video_predictor_example.ipynb`); zie ook `RELEASE_SAM3p1.md` daar. Bij `TypeError: 'SimpleTokenizer' object is not callable` is het verkeerde `clip`-pakket geïnstalleerd: draai dan `pip uninstall clip -y` en `pip install git+https://github.com/ultralytics/CLIP.git`.

De SAM 3.1-checkpoints (Object Multiplex, `sam3.1_multiplex.pt`) verbeteren volgens Meta vooral het *videomodel* (~7x sneller bij veel objecten); Ultralytics laadt hetzelfde checkpoint ook voor beeldinferentie, dus `run_sam31_test.bat` vergelijkt de actuele SAM 3.1-beelddetector met de SAM3-test (`sam3.pt`). De repositoryversie van de SAM3-code pint `numpy<2` en vraagt `pkg_resources` (`setuptools<81`); `requirements.txt` gebruikt daarom bewust de PyPI-release (`sam3==0.1.4`, werkt met numpy 2.x op Python 3.13) en niet de git-versie — gebruik voor die git-versie een aparte venv.

### Vergelijken

Draai eerst de testen die je wilt vergelijken en daarna:

```bat
run_compare_models.bat
```

`src/compare_models.py` leest alle modelstappen in `outputs/Disassembly1/` (CSV + summary) en schrijft `outputs/Disassembly1/comparison/model_comparison.csv` en `model_comparison.md` met daarin per model: gevonden toolklassen, tool-detecties, aandeel frames met een tool, gemiddelde confidence, detecties per frame, rekentijd en doorvoer. De rangschikking is een expliciet weegbare heuristic:

```text
score = 100 x (coverage x dekking + presence x tool-frames + speed x min(1, doorvoer/target_fps))
```

De gewichten (`comparison_weights`) en `comparison_target_fps` (de doorvoer die nodig is om real-time te draaien: bronfps / frame_stride) staan in `configs/baseline.yaml`.

Het volledige testrapport met de resultaten van alle modellen staat in [Documentation/Modelvergelijking.md](Documentation/Modelvergelijking.md).

Let op de kanttekeningen in het rapport: er is geen geannoteerde ground truth, dus accuracy (precision/recall) kan niet worden gemeten; dekking en tool-frames zijn proxies en bevoordelen open-vocabulary modellen, die hun promptklassen per definitie kunnen terugvinden. Confidence is niet tussen modelfamilies te vergelijken. Gebruik de debug-video's en voorbeeldframes per stap voor het kwalitatieve oordeel.

## Eerste MediaPipe-test

Dubbelklik op `run_handpose_test.bat` om MediaPipe handpose op dezelfde video uit te voeren. De handlandmarks komen in `outputs/Disassembly1/handpose/hand_landmarks.csv` en geannoteerde voorbeeldframes worden als JPG-bestanden opgeslagen. Daarnaast schrijft de stap een `handpose_debug.mp4` met de getekende handlandmarks (uit te zetten met `save_handpose_video: false` in de config).

## ByteTrack-tracking

Dubbelklik op `run_tracking_test.bat` om YOLO + ByteTrack op dezelfde video uit te voeren. Elke YOLO-detectie krijgt een unieke track-ID die tussen frames stabiel blijft, ook wanneer een object korte tijd aan het zicht onttrokken is (occlusie). De resultaten komen in `outputs/Disassembly1/tracking/`:

- `tracking_history.csv` – per frame één rij per track met `track_id`, `class_id`, `class_name`, `confidence`, `bbox_x1`..`bbox_y2`, `center_x`/`center_y` (bbox en center in pixels), `velocity_x`/`velocity_y` (verplaatsing in pixels per videoframe), `track_age` (frames sinds het ontstaan van de track), `frames_since_detection` en `status`. Rijen met status `occluded` hebben geen bbox of confidence: het object was tijdelijk niet zichtbaar, maar de track blijft bestaan om de occlusie te overbruggen. Rijen met status `tentative` zijn tracks die net gestart zijn en nog bevestigd moeten worden.
- `tracking_debug.mp4` – debug-video met YOLO-bboxes, klassennamen, confidence, track-ID's en trajectories. Occluded tracks worden grijs weergegeven op hun voorspelde positie. De video wordt als H.264 (yuv420p) geschreven, zodat hij in VS Code, browsers en deelplatforms afspeelt; OpenCV's standaard mp4v-formaat doet dat niet.

De trackerinstellingen staan in `configs/baseline.yaml`: `track_high_thresh` (drempel voor de eerste associatiestap), `track_low_thresh` (ondergrens waarmee YOLO-detecties worden meegenomen; lage scores worden in de tweede associatiestap gebruikt om occlusies te overbruggen), `new_track_thresh` (minimale score om een nieuwe track te starten), `track_buffer` (hoeveel frames een track zonder detectie blijft bestaan), `match_thresh` en `fuse_score` (associatie) en `track_trajectory_length` (lengte van de getekende trajectory). Associatie gebeurt met beweging (Kalman-filter) en IoU; ByteTrack heeft geen appearance-embedder en draait op de CPU.

## Draaien op de GPU

De scripts accepteren een `--device` argument met de waarden `cpu` of `gpu`:

```bat
run_yolo_test.bat --device gpu
run_handpose_test.bat --device gpu
run_tracking_test.bat --device gpu
run_rfdetr_test.bat --device gpu
run_rfdetr_seg_test.bat --device gpu
run_grounding_dino_test.bat --device gpu
run_yolo_world_test.bat --device gpu
run_yolo_seg_test.bat --device gpu
run_sam3_test.bat --device gpu
run_sam31_test.bat --device gpu
```

Of direct via Python:

```bat
.venv\Scripts\python.exe src\run_yolo_video.py --config configs\baseline.yaml --device gpu
.venv\Scripts\python.exe src\run_handpose.py --config configs\baseline.yaml --device gpu
.venv\Scripts\python.exe src\run_tracking.py --config configs\baseline.yaml --device gpu
.venv\Scripts\python.exe src\run_rfdetr_video.py --config configs\baseline.yaml --device gpu
```

Zonder argument gebruiken de scripts de waarde van `device` in `configs\baseline.yaml` (daar staat `gpu`; met `--device cpu` dwing je een CPU-run af).

**YOLO:** bij `--device gpu` controleert het script of torch CUDA ondersteunt en installeert het automatisch de CUDA-build van torch (`--index-url https://download.pytorch.org/whl/cu130`) als dat nog niet het geval is. De eerste keer kan dit enkele minuten duren (download van ca. 2 GB). De ByteTrack-tracker in `run_tracking.py` draait op de CPU en heeft geen CUDA nodig; alleen de YOLO-detectie gebruikt de GPU.

**MediaPipe:** in de huidige Windows-build van mediapipe is de GPU-delegate niet beschikbaar. `--device gpu` wordt geaccepteerd, maar het script meldt dit en valt terug op CPU.

## Voortgang

Tijdens het draaien wordt de voortgang direct getoond: in een console een balk die op één regel bijwerkt, en bij vastgelegde uitvoer (logbestand of IDE-paneel) voortgangsregels bij elke 10% of elke 5 seconden.
