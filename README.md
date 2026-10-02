# Project DUPLO

## Eerste YOLO-test

Dubbelklik op `run_yolo_test.bat` vanuit de projectmap. Het script maakt `.venv` aan als die nog niet bestaat, installeert of controleert de packages uit `requirements.txt`, controleert de imports en voert daarna de YOLO-test uit op `data/Disassembly1.mp4`.

De eerste run downloadt automatisch het pretrained model `yolo11n.pt`. De resultaten komen in `outputs/Disassembly1/`, met detections in `yolo_detections.csv` en voorbeeldframes als JPG-bestanden.

## Eerste MediaPipe-test

Dubbelklik op `run_handpose_test.bat` om MediaPipe handpose op dezelfde video uit te voeren. De handlandmarks komen in `outputs/Disassembly1/hand_landmarks.csv` en geannoteerde voorbeeldframes worden als JPG-bestanden opgeslagen.

