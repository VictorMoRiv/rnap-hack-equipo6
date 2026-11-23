# Casa Norte: The Interface That Follows Your Intent

**RNAP Hack 2026 · Track 5 · Redes Neuronales y Aprendizaje Profundo · November 2026**

A Spanish-language home assistant that understands what the user wants from a single sentence, extracts the key data (time, place, person, item) and shows the screen card that completes the task, instead of answering with long text.

> Casa Norte is a fictional company created for the course challenge.

---

## The Problem

Casa Norte builds a home assistant in Spanish for families in northern Mexico. The assistant has always answered with text. Users ask it to set alarms or add items to the shopping list, and it replies with long sentences they must read and then act on in another app. In user tests, a simple card with a button would have let people finish the task in half the time.

## The Challenge

Can the system understand what the user wants from one sentence, pick out the key data (time, place, person, item) and show the screen component that completes the task?

## What This Project Does

1. The user types an informal sentence, e.g. *"despiértame a las seis y cuarto"*.
2. The model detects the **intent** (e.g. set an alarm).
3. The model extracts the **slots** (e.g. time = 06:15).
4. The interface shows the matching **card** with the data already filled in and hides the rest.
5. An action on the card produces a **real change** in the app state (e.g. the alarm is added to the alarm list). If the user edits a value on the card, the app state changes too.

> Phase 1 covered steps 1–3. Phase 2 adds the fine-tuned Transformer and the Gradio interface v1 (steps 4–5).

## Approach

Three levels are compared on the same test set, as required by the event rules:

| Level | Intent detection | Key data (slots) |
|---|---|---|
| **Baseline (trained by the team)** | TF-IDF (word 1–2 and character 3–5 grams) + Logistic Regression | Word-level token classifier (`DictVectorizer` + Logistic Regression) |
| **Pre-trained, no fine-tuning** | Multilingual zero-shot NLI (`MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`) | Not extracted by this model |
| **Phase 2: fine-tuned Transformer** | BETO (`dccuchile/bert-base-spanish-wwm-cased`) + intent head | BETO + BIO slot head (shared encoder) |

The fine-tuned Transformer uses one Spanish encoder with two heads: an 8-way intent classifier and a 39-label BIO slot tagger (19 slot types × B/I + O).

## Intents

Trained only on **8 classes** selected from the 60 MASSIVE intents, because each one maps to a card with a real action:

| Class | Card / action |
|---|---|
| `alarm_set` | Alarm card: time and label → create alarm |
| `alarm_query` | Alarm list: view active alarms |
| `weather_query` | Weather card: place and date |
| `lists_createoradd` | Shopping list: add an item |
| `lists_remove` | Shopping list: remove an item |
| `calendar_set` | Calendar card: event or reminder |
| `play_music` | Music card: now playing |
| `control_luces` | Lights card: room, color or brightness |

Two dataset details:

- `control_luces` **groups** five MASSIVE intents (`iot_hue_lighton`, `iot_hue_lightoff`, `iot_hue_lightup`, `iot_hue_lightdim`, `iot_hue_lightchange`) because `iot_hue_lighton` alone has only 22 training sentences.
- MASSIVE has no `reminder` intent. Reminders are covered by `calendar_set` ("pon un recordatorio para llamar a Pablo mañana") and `alarm_set`.

Sentences outside these intents are handled as follows: the fine-tuned model uses a 0.55 confidence threshold and the Gradio app adds a conservative keyword guard for obvious out-of-scope requests. Neither is a learned `out_of_scope` class, so a dedicated class and better calibration remain future work.

## Data

- **MASSIVE**, locale `es-ES`, version 1.1: 16,521 sentences, 60 intents and 55 slot types ([Hugging Face](https://huggingface.co/datasets/AmazonScience/massive)).
- A local copy of the filtered data is in `data/` (`massive_es-ES_train.jsonl`, `massive_es-ES_validation.jsonl`, `massive_es-ES_test.jsonl`).
- The official partitions are kept: **train = 3,127**, **validation = 565**, **test = 837** sentences for the 8 classes. `validation` is used only to choose hyperparameters; `test` is used once, for the final metrics.
- `split_manifest.json` stores the sentence ids of each partition to prove there is no leakage.
- **Extra test set:** 30 Mexican-style sentences written by the team (`data/mexican_test.jsonl`, 25 in-domain + 5 out-of-scope). They are **not** used for training.

## Evaluation

Results on the MASSIVE `test` split (837 sentences):

| Metric | Baseline (TF-IDF + LogReg) | Zero-shot (mDeBERTa) | BETO fine-tuned |
|---|---|---|---|
| Intent accuracy | 0.9558 | 0.5436 | **0.9725** |
| Intent macro-F1 | 0.9292 | 0.5706 | **0.9602** |
| Slot span-F1 | **0.7079*** | — (not extracted) | 0.6753 |
| Slot precision | 0.8447* | — | 0.8150 |
| Slot recall | 0.6093* | — | 0.5765 |
| Runtime | 3 min 22 s (train + test) | 40 min (test only, CPU) | 46 min 24 s (train + test, CPU) |
| Accuracy on 25 Mexican in-domain sentences | 0.920 | — | **0.960** |
| Out-of-scope rejected (5) | 0 / 5 | — | 0 / 5** |

\* Re-evaluated on the 19 canonical slot types used by the Transformer. Phase 1 reported 0.6659 over all 55 MASSIVE slot types, so the numbers are not directly comparable.

\*\* The interface adds a keyword guard that catches the five out-of-scope demo sentences; the model alone does not reject them.

BETO improves intent accuracy (+0.017) and macro-F1 (+0.031), and transfers better to the team's Mexican sentences. The feature-engineered baseline is still slightly better at exact slot boundaries, so the interface combines BETO's intent and slots with deterministic normalization and fallbacks. The zero-shot model never saw MASSIVE; its 0.54 accuracy shows that label wording and domain shift matter a lot. Full analysis, confusion matrix, learning curves and error cases are in `notebooks/technical_report.md`.

## Interface

Built with **Gradio** in `app/app.py`: one card per intent (alarm, weather, shopping list, calendar, music, lights) appears when detected and hides the rest, with editable fields and a button that changes the app state.

- The intent and slots come from the fine-tuned BETO model in `models/beto_joint/`.
- Alarms, events, the shopping list, the light state and "now playing" are kept per session with `gr.State`.
- Time and date values are normalized ("seis y media" → 06:30, "el viernes" → next Friday).
- Weather is a clearly labeled demo (no external API); music opens a YouTube search.
- Out-of-scope text gets a clarification card, and confidence below 0.55 asks the user to rephrase.

```bash
pip install -r requirements.txt
python notebooks/fase2_transformer.py   # trains and saves models/beto_joint (CPU: ~46 min)
python app/app.py                       # http://127.0.0.1:7860
```

Final version: Hugging Face Spaces (free CPU) or the team laptop. Live demo link: pending.

**Planned extra:** a cutaway "home simulator" in the style of The Sims inside the interface: little characters, lights that turn on, and an alarm on the bed that reacts to the user's commands.

## Repository Structure

```
.
├── app/app.py               # Gradio interface v1
├── casa_norte/              # shared NLU code: data parsing, normalization, model
├── data/                    # MASSIVE es-ES copy + 30 Mexican sentences
├── models/beto_joint/       # fine-tuned weights (ignored by git, 420 MB)
├── notebooks/
│   ├── casa_norte_8intenciones.py   # Phase 1 baseline + zero-shot
│   ├── fase2_transformer.py         # BETO fine-tune, metrics, figures, errors
│   ├── eval_baseline_mexican.py      # baseline on the 30 Mexican sentences
│   ├── figs/                        # confusion matrix, per-class F1, curves
│   ├── metrics_fase2.json           # Phase 2 metrics
│   ├── metrics_baseline.json        # baseline on test + Mexican
│   ├── error_cases.json             # 8 test error cases
│   └── technical_report.md          # sections 1–5
├── PROGRESS.md              # checkpoint log (CP1–CP4)
├── README.md
├── RNAP-ET4-D2.pdf          # English D2 deliverable
├── requirements.txt
└── split_manifest.json      # ids of the train/validation/test splits
```

## How to Run

```bash
pip install -r requirements.txt
python notebooks/casa_norte_8intenciones.py   # Phase 1 baseline + zero-shot
python notebooks/fase2_transformer.py         # Phase 2 BETO fine-tune (CPU: ~46 min)
python app/app.py                             # interface v1
```

`requirements.txt` pins `datasets==2.19.0` because MASSIVE still ships a loading script and `datasets` 3+ removed script support. `fase2_transformer.py` reads the JSONL files already in `data/`, so it does not need to download the dataset. Model weights are not committed (420 MB); the training script regenerates them.

## Judging Criteria (Track 5)

- **Results:** right intent and right data on the judge's unseen sentences.
- **Interface:** the card changes with the intent and its action works.
- **Robustness:** informal Mexican Spanish and sentences outside the chosen intents.
- **Clarity:** the user understands what the assistant understood.

## Event Rules Followed

- Clean train / validation / test split; the test set is used only for the final metrics.
- Only public, free data or data created by the team; synthetic data is labeled as such.
- AI tools were used for writing code; every member can explain the whole project.
- Scope kept small: one complete flow over several half-finished features.

## Timeline

| Date | Checkpoint |
|---|---|
| Thu, Nov 5, 2026 | CP1 |
| Mon, Nov 9, 2026 (extended) | CP2 · Phase 1 baseline |
| Mon, Nov 16, 2026 (extended) | CP3–CP4 · Transformer + interface v1 |
| Thu, Nov 19, 2026 | Final presentation in English + live judge test |

## Authors

Travis Contreras, Julio Esquivel, Víctor Morales, Angel Tamez · RNAP Hack 2026 · Redes Neuronales y Aprendizaje Profundo

## License / Data Credits

MASSIVE dataset by Amazon Science (CC BY 4.0).
