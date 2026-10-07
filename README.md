# Project DUPLO

## Eerste YOLO-test

Dubbelklik op `run_yolo_test.bat` vanuit de projectmap. Het script maakt `.venv` aan als die nog niet bestaat, installeert of controleert de packages uit `requirements.txt`, controleert de imports en voert daarna de YOLO-test uit op `data/Disassembly1.mp4`.

De eerste run downloadt automatisch het pretrained model `yolo11x.pt` als dat nog ontbreekt. De resultaten komen in `outputs/Disassembly1/`, met detections in `yolo_detections.csv` en voorbeeldframes als JPG-bestanden.

## Eerste MediaPipe-test

Dubbelklik op `run_handpose_test.bat` om MediaPipe handpose op dezelfde video uit te voeren. De handlandmarks komen in `outputs/Disassembly1/hand_landmarks.csv` en geannoteerde voorbeeldframes worden als JPG-bestanden opgeslagen.

## Draaien op de GPU

Beide scripts accepteren een `--device` argument met de waarden `cpu` of `gpu`:

```bat
run_yolo_test.bat --device gpu
run_handpose_test.bat --device gpu
```

Of direct via Python:

```bat
.venv\Scripts\python.exe src\run_yolo_video.py --config configs\baseline.yaml --device gpu
.venv\Scripts\python.exe src\run_handpose.py --config configs\baseline.yaml --device gpu
```

Zonder argument gebruiken de scripts de waarde van `device` in `configs\baseline.yaml` (standaard `cpu`).

**YOLO:** bij `--device gpu` controleert het script of torch CUDA ondersteunt en installeert het automatisch de CUDA-build van torch (`--index-url https://download.pytorch.org/whl/cu130`) als dat nog niet het geval is. De eerste keer kan dit enkele minuten duren (download van ca. 2 GB).

**MediaPipe:** in de huidige Windows-build van mediapipe is de GPU-delegate niet beschikbaar. `--device gpu` wordt geaccepteerd, maar het script meldt dit en valt terug op CPU.

