# Technisch plan – DUPLO Human Action Recognition

## 1. Doel van het project

We bouwen een modulaire video-analysepipeline voor DUPLO-disassembly waarmee we onderzoeken in hoeverre disassemblyprocessen automatisch uit video kunnen worden herkend en gestructureerd.

De pipeline moet uiteindelijk informatie kunnen verzamelen over:

* welke objecten/componenten aanwezig zijn;
* welke gereedschappen worden gebruikt;
* welke objecten door de tijd gevolgd worden;
* waar en hoe de handen bewegen;
* welke handelingen plaatsvinden;
* wanneer een handeling begint en eindigt.

Deze informatie moet later bruikbaar zijn voor onder andere **eDIM/reDIM, Disassembly Graphs, tool taxonomy en toekomstige work instructions**.

Het project is een proof of concept. Het doel is daarom niet om direct een volledig productierijp systeem te bouwen, maar om experimenteel vast te stellen wat technisch haalbaar is en waar de beperkingen liggen.

---

# 2. Overall workflow

```text
Video
  │
  ▼
Frame extraction
  │
  ├───────────────────────┐
  ▼                       ▼
YOLO object detection   MediaPipe Hands
  │                       │
  ▼                       ▼
Bounding boxes          21 landmarks/hand
  │                       │
  ▼                       │
Object tracking           │
(ByteTrack)               │
  │                       │
  ▼                       ▼
Object IDs + history + hand movement
             │
             ▼
      Gecombineerde
      temporele data
             │
             ▼
      Action recognition
             │
             ▼
  Disassembly action labels
             │
             ▼
 eDIM / Disassembly Graph /
 Tool taxonomy / work instructions
```

De pipeline wordt in fases opgebouwd. Eerst wordt onderzocht wat met bestaande pretrained modellen mogelijk is. Daarna wordt de dataset uitgebreid en worden relevante modellen gefinetuned.

---

# 3. Fase 1 – Technische baseline

## 3.1 YOLO objectdetectie

Gebruik YOLO26X met pretrained weights als eerste baseline.

Per frame worden minimaal opgeslagen:

* frame_number
* timestamp
* class_id
* class_name
* confidence
* bounding box: x1, y1, x2, y2

De pretrained versie wordt gebruikt om te onderzoeken welke informatie al zonder training uit de beschikbare video's kan worden gehaald.

### Belangrijke beperking

Het pretrained model is getraind op algemene objectklassen en bevat niet noodzakelijk de objecten die voor DUPLO relevant zijn.

Voorbeelden:

* schroeven;
* specifieke schroevendraaiers;
* connectoren;
* kabels;
* specifieke productonderdelen.

Wanneer een object niet door YOLO wordt gedetecteerd, kan ByteTrack dit object ook niet volgen.

De baseline wordt daarom vooral gebruikt om de **gap tussen een generiek pretrained model en de DUPLO-specifieke toepassing** te bepalen.

---

# 4. Fase 2 – Object tracking

Wanneer YOLO een object detecteert, worden de detecties doorgegeven aan een multi-object tracker.

Er wordt gebruikgemaakt van **ByteTrack**, de tracker die ook in de Ultralytics-tooling is opgenomen.

ByteTrack moet:

* dezelfde objecten over meerdere frames koppelen;
* een unieke track_id geven;
* bewegingsinformatie gebruiken (Kalman-filter);
* detecties met een lage confidence in een tweede associatiestap gebruiken;
* tijdelijke occlusion zo goed mogelijk kunnen overbruggen.

Voorbeeld:

```text
Frame 100:
screw → track_id 17

Frame 101:
screw → track_id 17

Frame 102:
screw → track_id 17

Frame 103:
screw → track_id 17
```

Hierdoor ontstaat een objectgeschiedenis in plaats van alleen losse frame-detecties.

Omdat DUPLO-disassembly veel beweging en occlusion bevat, kan ByteTrack later worden vergeleken met alternatieven zoals **BoT-SORT**.

---

# 5. Fase 3 – Object history

Per track_id wordt een tijdreeks opgeslagen.

Bijvoorbeeld:

```text
track_id: 17
class: screw

frame 100 → bbox + center + confidence
frame 101 → bbox + center + confidence
frame 102 → bbox + center + confidence
frame 103 → bbox + center + confidence
```

Waar mogelijk worden aanvullende eigenschappen berekend:

* center_x
* center_y
* displacement
* velocity
* track age
* frames sinds laatste detectie
* visibility/status
* trajectory

Hiermee kan later worden onderzocht of bewegingen van objecten kunnen helpen bij het herkennen van disassembly actions.

---

# 6. Fase 4 – Handdetectie en handtracking

MediaPipe Hands wordt onafhankelijk van YOLO/ByteTrack gebruikt.

Per frame worden opgeslagen:

* left/right hand;
* 21 landmarks;
* landmark x/y/z;
* hand confidence indien beschikbaar.

De handdata krijgt dezelfde:

* frame_number;
* timestamp;

als de objectdetecties en trackingdata.

Hierdoor kunnen bijvoorbeeld de volgende gebeurtenissen later worden geanalyseerd:

```text
hand approaches object
        ↓
hand touches object
        ↓
object starts moving
        ↓
object moves away
```

MediaPipe levert hiermee belangrijke informatie voor actieherkenning zonder dat hiervoor direct een zwaar pose-estimationmodel nodig is.

---

# 7. Fase 5 – Definitie van acties en labels

Voordat een action-recognitionmodel wordt getraind, wordt een beperkte actietaxonomie opgesteld.

De voorgestelde startcategorieën zijn gebaseerd op de opdracht:

| Actie                      | Mogelijke eDIM-koppeling     |
| -------------------------- | ---------------------------- |
| Unscrewing                 | Manipulation / Disconnection |
| Tool change                | Tool change                  |
| Pick up                    | Manipulation                 |
| Remove component           | Removing                     |
| Disconnect cable/connector | Disconnection                |
| Position product           | Positioning                  |

De uiteindelijke labels worden afgestemd met de opdrachtgever.

Daarnaast wordt onderzocht welke informatie nodig is om een actie volledig te beschrijven.

Bijvoorbeeld:

```text
REMOVE
component = battery
tool = screwdriver
start = 12.4s
end = 16.8s
```

Hierdoor wordt niet alleen "remove" herkend, maar ontstaat informatie die later daadwerkelijk naar DUPLO-structuren kan worden vertaald.

---

# 8. Fase 6 – Dataset en annotatie

De beschikbare DUPLO-video's worden gebruikt om een kleine gelabelde dataset op te bouwen.

De dataset bestaat uit meerdere informatielagen:

### Objectannotaties

Bounding boxes voor relevante:

* componenten;
* gereedschappen;
* connectoren;
* andere objecten die relevant zijn voor de acties.

### Actieannotaties

Acties krijgen:

* action label;
* start frame/timestamp;
* eind frame/timestamp.

### Eventueel

Relaties tussen:

* hand;
* object;
* gereedschap;
* actie.

Bijvoorbeeld:

```text
hand → screwdriver → screw
```

Dit maakt het mogelijk om later complexere acties te herkennen.

---

# 9. Fase 7 – YOLO fine-tuning

Zodra voldoende relevante frames zijn geannoteerd, wordt YOLO gefinetuned op de eigen DUPLO-objectklassen.

De pretrained COCO-klassen worden niet één-op-één overgenomen. De nieuwe dataset bepaalt welke klassen het uiteindelijke model moet herkennen.

Bijvoorbeeld:

```text
0 = screwdriver
1 = screw
2 = connector
3 = cable
4 = component_A
5 = component_B
```

Objecten die niet relevant zijn voor de toepassing hoeven niet als klasse in het uiteindelijke model te worden opgenomen.

Dit kan bijvoorbeeld voorkomen dat een hand als `person` wordt gedetecteerd, omdat `person` geen relevante klasse hoeft te zijn wanneer MediaPipe Hands specifiek voor handinformatie wordt gebruikt.

De pretrained YOLO-weights worden gebruikt als startpunt voor transfer learning.

---

# 10. Fase 8 – Action recognition

Pas nadat voldoende temporele data en actielabels beschikbaar zijn, wordt onderzocht welke methode geschikt is voor daadwerkelijke actieherkenning.

Mogelijke kandidaten zijn:

* ST-GCN op MediaPipe keypoints;
* temporal action detection;
* video classification;
* een eigen lichtgewicht temporal model;
* een combinatie van object trajectories en hand landmarks.

De keuze wordt gebaseerd op:

* hoeveelheid beschikbare data;
* benodigde nauwkeurigheid;
* computationele eisen;
* interpreteerbaarheid;
* benodigde temporele informatie.

Het action-recognitionmodel hoeft dus niet noodzakelijk end-to-end op de volledige video te werken.

Een mogelijke input is bijvoorbeeld:

```text
Hand landmarks
       +
Object track history
       +
Object classes
       +
Tool information
       ↓
Temporal model
       ↓
"Remove component"
```

---

# 11. Fase 9 – Output

De pipeline moet uiteindelijk een machine-readable dataset kunnen produceren.

Bijvoorbeeld:

```text
timestamp
frame_number
track_id
object_class
confidence
bbox
center
velocity
hand_landmarks
action
action_start
action_end
```

Hiermee kan een gedetecteerde gebeurtenis bijvoorbeeld worden weergegeven als:

```text
12.4s – 16.8s

ACTION:
Remove component

OBJECT:
Battery

TOOL:
Screwdriver

HAND:
Right hand

CONFIDENCE:
0.91
```

Deze informatie kan later worden vertaald naar DUPLO-outputs.

---

# 12. Relatie met DUPLO

De uiteindelijke pipeline moet niet alleen een computer vision-demo opleveren, maar informatie produceren die aansluit op de bestaande DUPLO-structuur.

### Disassembly Graph

Gedetecteerde:

* componenten;
* acties;
* tools;
* volgorde;

kunnen later worden vertaald naar relaties en processtappen.

### eDIM / reDIM

Acties en eventueel hun duur kunnen worden gebruikt als input voor:

* identifying;
* manipulation;
* positioning;
* disconnection;
* removing;
* tool change.

### Tool taxonomy

Gedetecteerde gereedschappen en tool changes kunnen worden gekoppeld aan de bestaande toolcategorieën.

### Work instructions

De gedetecteerde sequence:

```text
pick up screwdriver
→ unscrew
→ remove component
→ disconnect cable
→ position component
```

kan later dienen als basis voor automatisch of semi-automatisch gegenereerde werkinstructies.

---

# 13. Evaluatie

De evaluatie wordt aangepast aan het type component.

### Objectdetectie

* Precision
* Recall
* F1-score
* mAP

### Tracking

* ID consistency
* ID switches
* tracking accuracy
* eventueel standaard MOT-metrics indien relevant

### Action recognition

Afhankelijk van de gekozen methode:

* Precision
* Recall
* F1-score
* confusion matrix
* temporal localization/segmentation quality

### Praktische evaluatie

Daarnaast wordt gekeken naar:

* processing time;
* visuele kwaliteit van de tracking;
* failure cases;
* occlusion;
* kleine objecten;
* false positives;
* false negatives.

Wanneer de dataset te klein is voor betrouwbare statistische conclusies, wordt een kwalitatieve demonstratie gecombineerd met een kritische analyse van de beperkingen.

---

# 14. Belangrijke uitgangspunten

De eerste versie hoeft niet direct het volledige systeem te bevatten.

De ontwikkeling verloopt stapsgewijs:

```text
FASE 1
Pretrained YOLO + MediaPipe
        ↓
Wat kunnen we zonder training?

FASE 2
Object tracking
        ↓
Kunnen we objecten door de tijd volgen?

FASE 3
Annotatie + dataset
        ↓
Welke objecten en acties zijn relevant?

FASE 4
YOLO fine-tuning
        ↓
Kunnen we DUPLO-specifieke objecten herkennen?

FASE 5
Temporal/action recognition
        ↓
Kunnen we disassembly actions herkennen?

FASE 6
DUPLO mapping
        ↓
Kunnen we de resultaten koppelen aan
eDIM / Disassembly Graph / tool taxonomy?
```

Het project blijft daarmee een proof of concept en hoeft niet alle bovenstaande fases volledig uit te voeren wanneer de beschikbare data daarvoor onvoldoende is.

---

# 15. Kern van de architectuur

**YOLO26X**

> WHAT is present?

**ByteTrack / tracking**

> WHICH detected object is the same object over time?

**MediaPipe Hands**

> WHERE are the hands and how do they move?

**Object history**

> HOW do objects move over time?

**Action recognition**

> WHAT ACTION happened?

**DUPLO mapping**

> HOW does that action translate into eDIM, Disassembly Graphs and work instructions?

De code wordt modulair ontworpen zodat YOLO, tracker, MediaPipe en het latere action-recognitionmodel afzonderlijk vervangen en geëvalueerd kunnen worden.