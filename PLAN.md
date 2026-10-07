Het project:
We bouwen een video-analyse pipeline voor DUPLO-disassembly.

DOEL:
Detecteer objecten met YOLO26X, volg dezelfde objecten over meerdere frames met DeepSORT en detecteer handposities met MediaPipe Hands. De tracking- en handinformatie moet per frame worden opgeslagen zodat deze later gebruikt kan worden voor temporele actieherkenning.

PIPELINE:

Video
→ frame extraction
→ YOLO26X object detection
→ YOLO bounding boxes/classes/confidence
→ DeepSORT multi-object tracking
→ unieke track IDs + object history
→ MediaPipe Hands
→ 21 hand landmarks per hand
→ gecombineerde temporele dataset
→ later action recognition

STAP 1 — VIDEO:
Lees een video frame voor frame en behoud het oorspronkelijke frame number en timestamp.

STAP 2 — YOLO26X:
Voer YOLO26X objectdetectie uit op ieder frame.

Per detectie moeten minimaal worden opgeslagen:

- frame_number
- timestamp
- class_id
- class_name
- confidence
- bounding_box: x1, y1, x2, y2
Gebruik voorlopig het pretrained model. Er is nog geen fine-tuning.

STAP 3 — DEEPSORT:
Geef de YOLO-detecties door aan DeepSORT.

DeepSORT moet:

- bestaande objecten tussen frames aan elkaar koppelen;
- een unieke track_id geven;
- rekening houden met beweging;
- appearance information gebruiken voor associatie;
- tijdelijke occlusion zo goed mogelijk kunnen overbruggen.
Voorbeeld:

Frame 100:
screw → track_id 17

Frame 101:
screw → track_id 17

Frame 102:
screw → track_id 17

Frame 103:
screw → track_id 17

De track_id moet dus stabiel blijven zolang de tracker kan bepalen dat het om hetzelfde object gaat.

STAP 4 — OBJECT HISTORY:
Bewaar per track_id een tijdreeks.

Bijvoorbeeld:

track_id: 17
class: screw

frame 100 → bbox + center + confidence
frame 101 → bbox + center + confidence
frame 102 → bbox + center + confidence

Bereken indien mogelijk ook:

- center_x
- center_y
- velocity / displacement tussen frames
- track age
- aantal frames sinds laatste detectie
- zichtbaarheid/status
STAP 5 — MEDIAPIPE HANDS:
Gebruik MediaPipe Hands onafhankelijk van DeepSORT om beide handen te detecteren.

Bewaar per frame:

- left/right hand
- 21 landmarks
- landmark x/y/z
- hand confidence indien beschikbaar
De handdata moet dezelfde frame_number/timestamp gebruiken als de YOLO/DeepSORT-data zodat alles later gecombineerd kan worden.

STAP 6 — OUTPUT:
Maak een gestructureerde output waarin object tracking en hand landmarks aan dezelfde frames gekoppeld kunnen worden.

Bijvoorbeeld:

frame_number
timestamp
track_id
class_name
confidence
bbox_x1
bbox_y1
bbox_x2
bbox_y2
center_x
center_y
...

en daarnaast de MediaPipe landmarks.

Sla de tracking history bij voorkeur op in een machine-readable formaat zoals JSON/JSONL of CSV, zodat deze later opnieuw gebruikt kan worden zonder de video opnieuw te verwerken.

STAP 7 — VISUALISATIE:
Maak een optionele debug-outputvideo waarin zichtbaar is:

- YOLO bounding boxes
- class names
- confidence
- DeepSORT track IDs
- MediaPipe hand landmarks
- eventueel trajectory/history van tracked objects
Dit is belangrijk om visueel te controleren of tracking correct werkt.

BELANGRIJK:
De eerste implementatie moet nog GEEN action-recognition model trainen.

Het doel van deze fase is alleen:

1. Objecten detecteren.
2. Objecten door de tijd volgen.
3. Stabiele object IDs verkrijgen.
4. Object movement/history verzamelen.
5. Hand landmarks verzamelen.
6. Alles synchroniseren per frame.
7. De resultaten visueel kunnen controleren.
Daarna kunnen we deze temporele informatie gebruiken voor een latere action-recognition/temporal-action-detection fase.

ARCHITECTUUR:

YOLO26X = WHAT is present?
DeepSORT = WHICH object is the same object over time?
MediaPipe Hands = WHERE are the hands and how do they move?
Tracking history = HOW do objects move over time?
Action recognition (later) = WHAT ACTION happened?

Ontwerp de code modulair zodat YOLO, DeepSORT en MediaPipe afzonderlijk vervangen of getest kunnen worden.