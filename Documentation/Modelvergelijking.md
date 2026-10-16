# Testrapport – modelvergelijking voor toolrecognition

*Uitgevoerd op 14 oktober 2026 op `data/Disassembly1.mp4` met de pipelines in `src/` en de instellingen uit `configs/baseline.yaml`. Het bijbehorende cijfermateriaal staat in `outputs/Disassembly1/comparison/` (niet in git; opnieuw te genereren met `run_compare_models.bat`).*

> **Update oktober 2026 (cleanup):** SAM3 had een bug (`Sam3Detector.detect()` gebruikte alleen de laatste prompt; gefixt in `src/duplo/detection.py`) — oude SAM3-cijfers undercounten en moeten opnieuw worden gedraaid. `mask_area` is nu overal een pixel-count op frame-resolutie, YOLO-CSV's hebben `timestamp`, tracking/handpose schrijven `*_summary.json` en `compare_models.py` rapporteert tracking apart (`tracking_comparison.csv`). SAM 3.1 en de vier `*_tracking`-varianten ontbreken nog in de cijfers hieronder; draai `run_sam31_test.bat` + tracking-scripts en `run_compare_models.bat` om dit rapport aan te vullen.

## 1. Doel

Fase 7 van het technisch plan (zie [PLAN.md](PLAN.md)) draait om **tool recognition**: kan een pretrained model de gereedschappen in de DUPLO-disassemblyvideo's herkennen zonder eigen training? De baseline gebruikt YOLO26x met COCO-gewichten; dit onderzoek zet daar zeven alternatieven naast en beantwoordt de vraag welk model voor deze taak het beste past:

* **RF-DETR** (detectie) en **RF-DETR Seg** (segmentatie) – gesloten vocabulaire, COCO;
* **Grounding DINO** en **YOLO-World** – open-vocabulary: de gereedschapsnamen gaan als tekstprompt mee;
* **YOLO26x-seg** (segmentatie) – gesloten vocabulaire, COCO;
* **SAM3** en **SAM 3.1** – open-vocabulary segmentatie op basis van een tekstprompt.

## 2. Testopzet

| Onderdeel | Waarde |
| --- | --- |
| Video | `data/Disassembly1.mp4` (top-down opname van het demonteren van een telefoon) |
| Videoformaat | 9639 frames · 30 fps · 1280×720 · 321 s |
| `frame_stride` | 2 → **4820 verwerkte frames** per model |
| `conf_threshold` | 0,25 voor alle modellen |
| Grounding DINO `text_threshold` | 0,25 |
| Doelgereedschappen (`tool_classes`) | `screwdriver`, `wrench`, `pliers`, `cutter`, `hammer` |
| Voorbeeldframes | elke 150 verwerkte frames (`screenshot_every`) |
| Hardware | NVIDIA GeForce RTX 5080 (16 GB), Windows |
| Software | Python 3.13.15 · torch 2.14.1+cu130 · ultralytics 8.4.174 · rfdetr 1.11.2 · transformers 5.19.0 · supervision 0.30.9 · sam3 0.1.4 · triton-windows 3.8.0.post29 |

Elke modelstap doorloopt dezelfde pipeline: video inlezen (`duplo.video.VideoFrames`), per frame inferentie, resultaten wegschrijven en de rekentijd meten (`duplo.benchmark`). Per stap verschijnt een eigen submap in `outputs/Disassembly1/` met:

* `<model>_detections.csv` – één rij per detectie: `video`, `frame`, `timestamp`, `klasse`, `confidence`, `bbox_x/y/w/h` en bij segmentatie `mask_area` (maskeroppervlak in pixels);
* `<model>_debug.mp4` – H.264-video met bboxes, labels (en maskers);
* `<model>_frame_*.jpg` – voorbeeldframes voor de kwalitatieve controle;
* `<model>_summary.json` – frames, detecties, klassen en rekentijd.

## 3. Geteste modellen

| Stap | Gewichten | Soort | Omvang | Licentie |
| --- | --- | --- | --- | --- |
| `yolo` | `models/yolo26x.pt` (COCO) | detectie, gesloten vocabulaire | 59,0 M | AGPL-3.0 (Ultralytics) |
| `rfdetr` | `RFDETRLarge` (COCO) | detectie, gesloten vocabulaire | 33,9 M · 704×704 | Apache-2.0 (XL/2XL vallen onder PML) |
| `grounding_dino` | `IDEA-Research/grounding-dino-base` | detectie, open-vocabulary | 232,8 M | Apache-2.0 |
| `yolo_world` | `models/yolov8x-worldv2.pt` | detectie, open-vocabulary (CLIP-tekstprompt) | 72,9 M | AGPL-3.0 (Ultralytics) |
| `yolo_seg` | `models/yolo26x-seg.pt` (COCO) | segmentatie, gesloten vocabulaire | 70,7 M | AGPL-3.0 (Ultralytics) |
| `rfdetr_seg` | `RFDETRSeg2XLarge` (COCO) | segmentatie, gesloten vocabulaire | 38,6 M · 768×768 | Apache-2.0 |
| `sam3` | `facebook/sam3` (`sam3.pt`) | segmentatie, open-vocabulary | 859,9 M | Meta-licentie (gewichten achter HF-goedkeuring) |

De gewichten van RF-DETR en SAM3 worden bij de eerste run automatisch gedownload; YOLO-gewichten komen uit de Ultralytics-releases. SAM3 vraagt daarnaast om een goedgekeurde Hugging Face-toegangsaanvraag en `hf auth login` (zie [README](../README.md)).

## 4. Werkwijze en metrieken

Er is **geen geannoteerde ground truth**, dus precision/recall/mAP zijn niet te meten. De vergelijking gebruikt daarom meetbare proxies die samen iets zeggen over bruikbaarheid voor toolrecognition:

| Metriek | Betekenis |
| --- | --- |
| Tool-detecties | aantal detecties waarvan de klasse in `tool_classes` valt |
| Dekking | aantal doelklassen met ≥ 1 detectie (van de 5) |
| Tool-frames | aandeel verwerkte frames met ≥ 1 tool-detectie (temporele aanwezigheid) |
| Gem. conf. | gemiddelde confidence van de tool-detecties (alleen binnen één modelfamilie vergelijkbaar) |
| Det./frame | alle detecties per verwerkt frame (ook niet-tools; maat voor ruis/overkill) |
| Inferentie (ms) | mediaan van de modelaanroep per frame |
| Doorvoer (fps) | verwerkte frames per seconde wandkloktijd, inclusief video-inlezen, tekenen en debug-video |
| Score | heuristische rangschikking, zie formule hieronder |

```text
score = 100 × (0,4 × dekking + 0,3 × tool-frames + 0,3 × min(1, doorvoer/15))
```

De gewichten staan in `configs/baseline.yaml` (`comparison_weights`) en zijn bewust aanpasbaar: 15 fps is de doorvoer die nodig is om met `frame_stride: 2` real-time mee te lopen met de 30fps-video. Dekking en tool-frames wegen samen zwaarder dan snelheid (70/30), omdat een model dat de tools niet vindt ook snel niet nuttig is.

## 5. Resultaten

### 5.1 Score en hoofmetrieken

| Model | Modelnaam | Toolklassen (dekking) | Tool-detecties | Tool-frames | Gem. conf. | Det./frame | Inferentie (ms) | Doorvoer (fps) | Score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| yolo_world | YOLO-World v2 X | **4/5** | 1 965 | 35,2 % | 0,38 | 0,41 | **13,7** | **61,6** | **72,6** |
| grounding_dino | Grounding DINO base | 3/5 | 7 637 | 98,1 % | 0,43 | 1,59 | 164,5 | 5,9 | 65,3 |
| sam3 | SAM3 (`facebook/sam3`) | 3/5 | **8 971** | **99,9 %** | **0,82** | 1,86 | 492,7 | 2,0 | 57,9 |
| rfdetr | RF-DETR Large | 0/5 | 0 | 0 % | n.v.t. | 3,10 | 12,4 | 67,9 | 30,0 |
| rfdetr_seg | RF-DETR Seg 2XLarge | 0/5 | 0 | 0 % | n.v.t. | 2,76 | 23,4 | 33,1 | 30,0 |
| yolo | YOLO26x (baseline) | 0/5 | 0 | 0 % | n.v.t. | 2,50 | 15,0 | 53,9 | 30,0 |
| yolo_seg | YOLO26x-seg | 0/5 | 0 | 0 % | n.v.t. | 1,98 | 18,5 | 41,8 | 30,0 |

Totale wandkloktijd per volledige video (5 min): RF-DETR 71 s, YOLO-World 78 s, YOLO 89 s, YOLO-seg 115 s, RF-DETR-seg 146 s, Grounding DINO 812 s (13,5 min) en SAM3 2422 s (40,4 min).

#### Hardware

Getest op: RTX 5080 (16 GB VRAM), Ryzen 9 9950X (16 cores, 32 threads, 4,5 GHz), 64GB DDR5-6000 RAM.

### 5.2 Detecties per gereedschapsklasse

| Gereedschap | grounding_dino | sam3 | yolo_world | yolo | yolo_seg | rfdetr | rfdetr_seg |
| --- | --- | --- | --- | --- | --- | --- | --- |
| screwdriver | 7 598 | **8 826** | 1 927 | 0 | 0 | 0 | 0 |
| pliers | 30 | 144 | 30 | 0 | 0 | 0 | 0 |
| hammer | 9 | 0 | 5 | 0 | 0 | 0 | 0 |
| wrench | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| cutter | 0 | 1 | 0 | 0 | 0 | 0 | 0 |

Alleen de drie open-vocabulary modellen vinden gereedschappen; de COCO-modellen scoren 0/5. Dat is geen detectiefout maar een **vocabulairebeperking**: RF-DETR en de YOLO-modellen labelen de schroevendraaier consequent als `knife` of `scissors`. Die klassen komen in de ruwe data veel voor (RF-DETR: 690× `knife`, 163× `scissors`; YOLO-seg: 58× `scissors`), maar tellen niet mee omdat ze niet in `tool_classes` staan.

De segmentatiemodellen loggen ook `mask_area` (maskeroppervlak in pixels). Die getallen zeggen vooral *wat* er gemaskeerd wordt: YOLO-seg (gem. 48 566 px) en RF-DETR-seg (35 386 px) maskeren vrijwel uitsluitend de persoon/handen en de telefoon, terwijl SAM3 (gem. 3 113 px) de gereedschappen zelf maskeert — een klein, precies oppervlak dat past bij een schroevendraaier.

### 5.3 Meest gedetecteerde klassen (top-3)

| Model | Top-3 klassen |
| --- | --- |
| yolo_world | screwdriver (1 927), pliers (30), hammer (5) |
| grounding_dino | screwdriver (7 598), pliers (30), `##drive` (11)* |
| sam3 | screwdriver (8 826), pliers (144), cutter (1) |
| rfdetr | person (8 436), cell phone (4 546), knife (690) |
| rfdetr_seg | person (5 866), cell phone (5 758), knife (695) |
| yolo | person (6 435), cell phone (2 706), book (1 379) |
| yolo_seg | person (7 505), cell phone (1 698), remote (192) |

\* `##drive` is een deelwoord-artefact van Grounding DINO's labeldecodering (zoals `screw`, `##r`, `screwdrive`); `run_grounding_dino_video.py` koppelt zulke fragmenten aan de juiste promptklasse, waardoor de schroevendraaier in de CSV correct is weggeschreven.

## 6. Kwalitatieve controle

De cijfers zijn gecontroleerd tegen de opgeslagen voorbeeldframes en debug-video's:

* **Frame 3600** – YOLO-World zet een strak kader om de echte schroevendraaier (`screwdriver 0,28`); RF-DETR zet vrijwel hetzelfde kader om hetzelfde object, maar noemt het `knife` (0,27). Het vocabulaireprobleem is dus zichtbaar in één frame.
* **Frame 6000** – Grounding DINO vindt twee echte schroevendraaiers (0,53 en 0,21); SAM3 segmenteert dezelfde twee objecten met maskers op 0,96 en 0,92. De maskers liggen precies op de gereedschappen.
* **Frame 9150** – SAM3 vindt de schroevendraaier opnieuw (0,97), maar produceert ook twee kleine false positives op de hand (0,34 en 0,43).
* De zeldzame klassen zijn ruis: YOLO-Worlds `wrench` (3×) en `hammer` (5×) liggen op de schroevendraaier of hand, DINO's `pliers`/`hammer` (30/9) zijn kortstondig en laag-confident, en SAM3's `pliers` (144, mediaan 0,36) liggen grotendeels op de handen. Het betrouwbare signaal is bij alle drie **`screwdriver`**.
* De continue `person`- en `cell phone`-detecties van de COCO-modellen grijpen de handen/arm van de monteur en de telefoon die in beeld ligt; ze zijn voor toolrecognition niet bruikbaar.

## 7. Interpretatie en advies

1. **YOLO-World v2 X** is de beste keuze voor de pipeline zoals die nu draait: het gereedschap dat in deze video echt voorkomt (de schroevendraaier) krijgt meteen de juiste naam, het vindt 4 van de 5 doelklassen, en met 61,6 fps (pipeline) is het ruim snel genoeg om real-time te draaien en om er direct ByteTrack-tracking op te zetten.
2. **SAM3** levert de hoogste kwaliteit per detectie: mediaan-confidence 0,96, maskers die de gereedschappen precies volgen en een tool-detectie in 99,9 % van de frames. Met ± 2 fps is het niet real-time, maar het is zeer geschikt voor keyframe-analyse, offline verwerking en als **auto-labeler** om een toolannotatie te maken voor fine-tuning of voor precision/recall-meting.
3. **Grounding DINO** zit er tussenin: 98,1 % tool-frames en redelijke dekking, maar lagere confidence, meer verspreide kaders en 5,9 fps.
4. **RF-DETR en de YOLO26-modellen (detectie en segmentatie) vallen af zonder finetuning**: ze zijn zeer snel (12–23 ms), maar missen de juiste klassen. Wil het project deze modellen gebruiken, dan is finetuning op de toolklassenlijst uit PLAN.md fase 7 nodig — met SAM3 (of Grounding DINO) als labelbron.

## 8. Beperkingen

* **Geen ground truth**: dekking en tool-frames zijn proxies; absolute precision/recall zijn niet bekend. Een geannoteerde steekproef (bijvoorbeeld 200 frames) is nodig om de modellen echt te rangschikken op kwaliteit.
* **Vocabulaire-bias**: open-vocabulary modellen krijgen de doelklassen als prompt en kunnen die daardoor "altijd" terugvinden; gesloten-vocabulaire modellen kunnen per definitie niet winnen op dekking. De vergelijking meet dus eerder haalbaarheid dan accuratesse.
* **Confidence is niet vergelijkbaar tussen families**: 0,38 bij YOLO-World is niet hetzelfde als 0,38 bij Grounding DINO.
* **Kleine aantallen zijn ruis**: klassen met < ± 50 detecties moeten als onzeker worden beschouwd (zie de kwalitatieve controle).
* **Eén video, één scène**: alles is gemeten op één top-down opname van één product; generalisatie naar andere video's is niet getest.
* **Snelheid is hardware-afhankelijk**: gemeten op een RTX 5080 zonder batching; een tragere GPU verandert de score via de doorvoerterm.
* **SAM3 draait per frame met vijf prompts**; een selectievere prompt of het videomodel (SAM 3.1, `sam3.1_multiplex.pt`) zou de doorvoer kunnen verbeteren.

## 9. Reproduceren

```bat
run_yolo_test.bat               :: baseline (optioneel, zit al in de vergelijking)
run_rfdetr_test.bat
run_rfdetr_seg_test.bat
run_grounding_dino_test.bat
run_yolo_world_test.bat
run_yolo_seg_test.bat
run_sam3_test.bat               :: vereist HF-toegang tot facebook/sam3 + 'hf auth login'
run_compare_models.bat          :: schrijft outputs\Disassembly1\comparison\model_comparison.{csv,md}
```

Direct via Python werkt ook, bijvoorbeeld `.venv\Scripts\python.exe src\run_yolo_world_video.py --config configs\baseline.yaml --device gpu`. De modelkeuzes, prompts, drempels en scoringsgewichten staan in `configs/baseline.yaml`; `src/run_sam3_video.py` haalt het ontbrekende BPE-vocabulaire van SAM3 eenmalig uit de sam3-repo en zet het in `models/`.

Wil je een ander gereedschap of een andere prompt testen, pas dan `tool_classes` aan (de open-vocabulary modellen gebruiken die lijst automatisch) en draai de betreffende stappen opnieuw.

## 10. Vervolgstappen

1. **Steekproef annoteren** (200–500 frames) om precision/recall en een echte F1 per model te berekenen; de opgeslagen voorbeeldframes en debug-video's zijn daar een goed startpunt voor.
2. **Auto-labeling**: SAM3-maskers gebruiken om toolannotaties te genereren en daarmee YOLO26/RF-DETR op de toolklassen te finetunen (fase 7 van het plan).
3. **Tracking**: ByteTrack op de YOLO-World-detecties zetten (`run_tracking.py` gebruikt nu `YoloDetector`) om een toolgeschiedenis per track te krijgen, zoals in fase 2 en 3 van het plan.
4. **SAM 3.1 videomodel** proberen voor geprompt tracken van gereedschappen over de hele video (multiplex-checkpoint, aparte venv).
5. **Uitbreiden naar meerdere video's en camerahoeken** om de conclusie robuuster te maken, en naar connectorherkenning zodra de actieherkenning er staat.
6. **Koppeling naar DUPLO-outputs**: de toolklassen en confidence per frame zijn direct bruikbaar als input voor de Disassembly Graph en de tool-taxonomie.

## 11. Bronnen

* RF-DETR (Roboflow, ICLR 2026): <https://github.com/roboflow/rf-detr>
* Grounding DINO (IDEA-Research): <https://github.com/IDEA-Research/GroundingDINO> · modelcard <https://huggingface.co/IDEA-Research/grounding-dino-base>
* YOLO-World (AILab-CVC): <https://github.com/AILab-CVC/YOLO-World> · Ultralytics-documentatie: <https://docs.ultralytics.com/models/yolo-world/>
* SAM 3 (Meta): <https://github.com/facebookresearch/sam3> · gewichten <https://huggingface.co/facebook/sam3>
* Ultralytics YOLO26: <https://docs.ultralytics.com/>
