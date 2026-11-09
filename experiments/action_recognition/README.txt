DUPLO — Technical action-recognition trials

PURPOSE
I tested whether two pretrained models could process a short disassembly
clip on my Mac without additional training. These are technical feasibility
trials, not an accuracy benchmark or an integrated production pipeline.

RECORDED SETUP
Apple M2, 8 GiB RAM; macOS 13.5; Python 3.11.18.
Both trials used CPU, float32 and four PyTorch threads.
Video: testvidDUPLOdm.mp4, 30 fps; interval [223, 230) seconds (03:43–03:50).
Both result files record the same video SHA-256:
230620064e5c3925a5204ffa62dbb9e324663ac43214166e54cb7ba6e29bb50a
Audio was not used. The video and model weights are not included.

ACTION OPTIONS
A1 — Gereedschap pakken of wisselen
     A person picks up or switches a hand tool.
A2 — Product positioneren
     A person repositions or rotates the product.
A3 — Losschroeven
     A person loosens a screw using a screwdriver.
A4 — Kabel of connector loskoppelen
     A person disconnects a cable connector.
A5 — Component verwijderen
     A person removes a component from the product.
The models received the English descriptions; Dutch labels are for reporting.

X-CLIP
Model: microsoft/xclip-base-patch16-zero-shot
Recorded revision: 6746d6a5202bf24300cd9bf36f457f174483d41b
32 uniformly sampled RGB frames; evaluation mode and inference_mode.
PyTorch 2.2.2; torchvision 0.17.2; Transformers 4.44.2.
Result directory: outputs/20261109_130011_893066/
Files: resultaat.json and scores.csv.
Ranking (raw logits, rounded; NOT confidence percentages):
  A5 — Component verwijderen:          22.676
  A3 — Losschroeven:                   21.353
  A4 — Kabel of connector loskoppelen: 21.186
  A1 — Gereedschap pakken of wisselen: 21.062
  A2 — Product positioneren:           19.067
Model-forward time: 6.870657 seconds (6.9 seconds rounded).
An earlier terminal trial took 7.2 seconds. That is a separate execution;
the result files included here refer to the 6.9-second execution.

SMOLVLM2
Model: HuggingFaceTB/SmolVLM2-500M-Video-Instruct
Recorded revision: 7b375e1b73b11138ff12fe22c8f2822d8fe03467
Eight uniformly sampled RGB images, supplied chronologically with timestamps
through the multi-image interface, rather than a native video input.
PyTorch 2.2.2; torchvision 0.17.2; Transformers 4.51.3.
CPU float32, eager attention, do_sample=False, max_new_tokens=16.
The prompt requests a single code from A1–A5. The complete prompt and
formatted chat are preserved in resultaat.json.
The script sets image_size longest_edge=384 and disables image splitting.
The recorded final pixel tensor is [1, 8, 3, 512, 512]. The 384 setting
must therefore not be reported as the final tensor resolution.
Result directory: outputs/smolvlm_20261109_132732_090047/
Files: resultaat.json and samenvatting.txt.
Raw model answer: A2.
Parsed prediction: A2 — Product positioneren.
Generation time: 14.059952 seconds (14.1 seconds rounded).
No five-class scores or confidence percentages were calculated.

INTERPRETATION AND LIMITATIONS
Both trials completed on my Mac. Their predictions differ from my provisional
interpretation of the clip as A3 (loosening screws). That interpretation has
not been independently verified. No ground-truth dataset was established.
One clip does not establish accuracy, precision, recall or macro-F1.
The models had only five options; no other/unknown option was offered.
X-CLIP used 32 frames and SmolVLM2 used eight. Their input processing and
output types also differ, so this is not a controlled model comparison.
Reported times exclude model loading, video decoding and preprocessing;
they are not end-to-end timings or proof of real-time performance.
The results do not prove performance on full videos or other products.
Future prompt experiments should retain these original results and use
separate development and evaluation examples.

REPOSITORY CONTENTS
test_xclip.py                    X-CLIP trial script
test_smolvlm.py                  SmolVLM2 trial script
requirements_xclip.txt           Five direct X-CLIP dependency pins
environment_xclip_mac.txt        pip freeze from the X-CLIP environment
environment_smolvlm_mac.txt      pip freeze from the SmolVLM2 environment
outputs/                        Recorded trial results listed above
README.txt                      This explanation
The environment snapshots document the tested Mac environments; they are
not guaranteed installation recipes for every operating system or GPU.
The scripts use different Transformers versions; keep their environments
separate. The main DUPLO environment does not need to be changed to read
the results. Model environments and weights are not stored in this folder.

READING AND OPTIONAL RERUNS
No installation is needed to read the CSV, JSON and text results.
For reruns, obtain the source video and install the appropriate dependencies
in separate environments. Both scripts use local_files_only=True: the
processor, tokenizer and safetensors weights must already be in the local
Hugging Face cache. The scripts do not download missing model files.

X-CLIP accepts --video, --start, --end and --revision. Its default video path
is Downloads/testvidDUPLOdm.mp4 under the current user's home directory.
SmolVLM2 currently stores the original absolute video path in the VIDEO
constant near the top of test_smolvlm.py. Change that constant for a different
machine. START and END are also constants; the prompt and printed text are
currently written for the seven-second 03:43–03:50 clip.

On the original Mac, from the Project-DUPLO root, the existing environments
can run the copied scripts using:
/Users/doahismail/Documents/duplo-technisch-test/.venv/bin/python experiments/action_recognition/test_xclip.py
/Users/doahismail/Documents/duplo-technisch-test/.venv-smolvlm/bin/python experiments/action_recognition/test_smolvlm.py
These interpreter paths are specific to the original Mac, not team-wide commands.

Each run creates a new timestamped directory next to the scripts under
outputs/. Previous results are not overwritten. SmolVLM2 also saves eight
source-frame PNGs in a frames/ subfolder. Those images are omitted from the
sharing package; creating a new run will save them locally again.
Original result metadata is preserved, including the original local video
path. The scripts have not been changed as part of the sharing review.
Recorded revisions identify the checkpoints used. X-CLIP can pin a cached
revision with --revision; SmolVLM2 currently does not explicitly pin a revision.

REVIEW
Both scripts passed Python syntax checks. The X-CLIP CSV scores match its
JSON scores; both JSON files record the same source-video hash and interval.
The stored frame counts and rounded timings agree with the reported trials.
This file review did not rerun either model or independently inspect the video.
