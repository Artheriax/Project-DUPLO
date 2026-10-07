# Project DUPLO

## Projectstructuur

- `configs/` – instellingen per experiment (video, model, drempels, device)
- `data/` – invoervideo's (niet in git)
- `models/` – gedownloade modelgewichten: `yolo11x.pt` en `hand_landmarker.task` (niet in git)
- `outputs/` – resultaten per video, per stap een eigen submap (`yolo/`, `handpose/`, `tracking/`) met CSV's, voorbeeldframes en debug-video
- `src/duplo/` – gedeelde componenten: CLI, config, GPU-setup, video-inlezen, output, detectie en DeepSORT-tracking
- `src/run_yolo_video.py` – YOLO-pipeline (objectdetectie)
- `src/run_handpose.py` – MediaPipe-pipeline (handlandmarks)
- `src/run_tracking.py` – YOLO + DeepSORT-pipeline (objecten volgen over frames)
- `run_yolo_test.bat` / `run_handpose_test.bat` / `run_tracking_test.bat` – startscripts voor Windows

## Eerste YOLO-test

Dubbelklik op `run_yolo_test.bat` vanuit de projectmap. Het script maakt `.venv` aan als die nog niet bestaat, installeert of controleert de packages uit `requirements.txt`, controleert de imports en voert daarna de YOLO-test uit op `data/Disassembly1.mp4`.

De eerste run downloadt automatisch het pretrained model `models/yolo11x.pt` als dat nog ontbreekt. De resultaten komen in `outputs/Disassembly1/yolo/`, met detections in `yolo_detections.csv` en voorbeeldframes als JPG-bestanden.

## Eerste MediaPipe-test

Dubbelklik op `run_handpose_test.bat` om MediaPipe handpose op dezelfde video uit te voeren. De handlandmarks komen in `outputs/Disassembly1/handpose/hand_landmarks.csv` en geannoteerde voorbeeldframes worden als JPG-bestanden opgeslagen.

## DeepSORT-tracking

Dubbelklik op `run_tracking_test.bat` om YOLO + DeepSORT op dezelfde video uit te voeren. Elke YOLO-detectie krijgt een unieke track-ID die tussen frames stabiel blijft, ook wanneer een object korte tijd aan het zicht onttrokken is (occlusie). De resultaten komen in `outputs/Disassembly1/tracking/`:

- `tracking_history.csv` – per frame één rij per track met `track_id`, `class_id`, `class_name`, `confidence`, `bbox_x1`..`bbox_y2`, `center_x`/`center_y` (bbox en center in pixels), `velocity_x`/`velocity_y` (verplaatsing in pixels per videoframe), `track_age` (frames sinds het ontstaan van de track), `frames_since_detection` en `status`. Rijen met status `occluded` hebben geen bbox of confidence: het object was tijdelijk niet zichtbaar, maar de track blijft bestaan om de occlusie te overbruggen.
- `tracking_debug.mp4` – debug-video met YOLO-bboxes, klassennamen, confidence, track-ID's en trajectories. Occluded tracks worden grijs weergegeven op hun voorspelde positie. De video wordt als H.264 (yuv420p) geschreven, zodat hij in VS Code, browsers en deelplatforms afspeelt; OpenCV's standaard mp4v-formaat doet dat niet.

De trackerinstellingen staan in `configs/baseline.yaml`: `track_max_age` (hoeveel frames een occlusie overbrugd mag worden), `track_n_init` (aantal opeenvolgende detecties voordat een track bevestigd is), `track_max_cosine_distance` (drempel voor appearance-associatie) en `track_trajectory_length` (lengte van de getekende trajectory). Associatie gebeurt met zowel beweging (Kalman-filter) als appearance (embedding van de objectcrop).

## Draaien op de GPU

De scripts accepteren een `--device` argument met de waarden `cpu` of `gpu`:

```bat
run_yolo_test.bat --device gpu
run_handpose_test.bat --device gpu
run_tracking_test.bat --device gpu
```

Of direct via Python:

```bat
.venv\Scripts\python.exe src\run_yolo_video.py --config configs\baseline.yaml --device gpu
.venv\Scripts\python.exe src\run_handpose.py --config configs\baseline.yaml --device gpu
.venv\Scripts\python.exe src\run_tracking.py --config configs\baseline.yaml --device gpu
```

Zonder argument gebruiken de scripts de waarde van `device` in `configs\baseline.yaml` (standaard `cpu`).

**YOLO:** bij `--device gpu` controleert het script of torch CUDA ondersteunt en installeert het automatisch de CUDA-build van torch (`--index-url https://download.pytorch.org/whl/cu130`) als dat nog niet het geval is. De eerste keer kan dit enkele minuten duren (download van ca. 2 GB). Bij `run_tracking.py` gebruikt DeepSORT dezelfde CUDA-build ook voor de appearance-embedder.

**MediaPipe:** in de huidige Windows-build van mediapipe is de GPU-delegate niet beschikbaar. `--device gpu` wordt geaccepteerd, maar het script meldt dit en valt terug op CPU.

## Voortgang

Tijdens het draaien wordt de voortgang direct getoond: in een console een balk die op één regel bijwerkt, en bij vastgelegde uitvoer (logbestand of IDE-paneel) voortgangsregels bij elke 10% of elke 5 seconden.

