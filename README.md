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

> Phase 1 covered steps 1–3. Phase 2 added the fine-tuned Transformer and Gradio v1. The final pass hardens data hygiene, adds labeled synthetic training phrases, calibrates out-of-scope behavior, exposes attention and ships the commercial Gradio v2 interface.

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

Sentences outside these intents are handled with a confidence threshold stored in the model bundle (0.65 in the final training config; the currently served `models/beto_joint/` bundle uses 0.55) plus a conservative keyword guard for obvious out-of-scope requests such as transfers, jokes or food delivery. Neither is a learned `out_of_scope` class; the app says "No reconozco esa acción todavía" instead of forcing a wrong card.

## Data

- **MASSIVE**, locale `es-ES`, version 1.1: 16,521 sentences, 60 intents and 55 slot types ([Hugging Face](https://huggingface.co/datasets/AmazonScience/massive)).
- A local copy of the filtered data is in `data/` (`massive_es-ES_train.jsonl`, `massive_es-ES_validation.jsonl`, `massive_es-ES_test.jsonl`).
- The official partitions are kept by id; additionally, repeated surface forms are removed from train and validation before fitting. Deduplicated sizes for the 8 classes: **train = 3,005**, **validation = 554**, **test = 837**. `validation` is used only to choose hyperparameters; `test` is used once, for the final metrics.
- `split_manifest.json` stores the sentence ids of each partition to prove there is no id leakage; the scripts also assert that no normalized text crosses the partition boundary.
- **Extra test set:** 38 Mexican-style sentences written by the team (`data/mexican_test.jsonl`, 30 in-domain + 8 out-of-scope). They are **not** used for training.
- **Synthetic augmentation:** 64 train-only phrases in `data/synthetic_norte_train.jsonl`, each marked `synthetic: true` (63 survive the held-out text check). They cover northern/Mexican times such as *seis y cuarto*, *siete y media*, *ocho menos diez*, relative alarms (*en 20 minutos*), multi-item lists (*un doce*) and *calor hoy en Monterrey*. No synthetic row appears in validation or test.

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

Built with **Gradio** in `app/app.py` (v2): each intent has its own `gr.Group(visible=False)` panel; only the detected one is shown, with editable fields and a button that changes the app state.

- The intent and slots come from a local bundle. Loading is offline and measured below 3 s on the demo CPU; the live app currently serves `models/beto_joint/` because the retrain to `modelo_casa_norte/` was stopped.
- Alarms, events, the shopping list, the light state and "now playing" are kept per session with real `gr.State` objects; the panel can be cleared or marked complete.
- The banner states the flow explicitly: text → intent + slots → editable card → state change.
- Time and date normalization covers informal forms: *seis y cuarto* → 06:15, *ocho menos diez* → 07:50, *en 20 minutos* → current time + 20 min, *el viernes* → next Friday.
- Items are split deterministically by commas, *y*, *e* or *and*; place, person and list values are cleaned from leading prepositions and articles.
- Weather is a clearly labeled demo (no external API); music opens a YouTube search.
- Out-of-scope text or confidence below the configured threshold shows the same clean "No reconozco esa acción todavía" card; no guessed card is forced.
- A **Técnico** tab renders the last-layer attention map with `plot_attention_map()`; the header shows intent, confidence and per-request inference time.
- Six quick examples include the OOS trap *"transfiere 500 pesos a mi hermana"*.

```bash
pip install -r requirements.txt
python notebooks/baseline_casa_norte.py  # exact TF-IDF + LogReg baseline + latency
python notebooks/fase2_transformer.py    # trains and saves modelo_casa_norte/
python app/app.py                        # http://0.0.0.0:7860
```

Current version: local service on the team laptop at `http://localhost:7860` (HTTP 200). Public Hugging Face Spaces deployment: pending.

**Planned extra:** a cutaway "home simulator" in the style of The Sims inside the interface: little characters, lights that turn on, and an alarm on the bed that reacts to the user's commands.

## Repository Structure

```
.
├── app/app.py               # Gradio interface v2: groups, state, technical tab
├── casa_norte/              # shared NLU code: data parsing, normalization, model, attention
├── data/                    # MASSIVE es-ES + 38 Mexican + 64 synthetic train phrases
├── modelo_casa_norte/       # target final offline bundle; retrain stopped, not created yet (ignored)
├── models/beto_joint/       # Phase 2 weights (ignored by git, 420 MB)
├── notebooks/
│   ├── casa_norte_8intenciones.py   # Phase 1 baseline + zero-shot
│   ├── baseline_casa_norte.py       # exact TF-IDF + LogReg baseline + latency
│   ├── fase2_transformer.py         # BETO fine-tune, metrics, figures, attention, errors
│   ├── entrenamiento.ipynb          # production cells: audit → baseline → train → explain → app
│   ├── eval_baseline_mexican.py      # baseline on the Mexican set
│   ├── figs/                        # confusion matrix, F1, curves, attention heatmaps
│   ├── metrics_fase2.json           # final transformer metrics
│   ├── metrics_baseline_prod.json   # exact baseline metrics + latency + leakage check
│   ├── error_cases.json             # 8 test error cases
│   └── technical_report.md          # sections 1–6
├── PROGRESS.md              # checkpoint log (CP1–CP5)
├── README.md
├── EVIDENCIA_APRENDIZAJE_ACT12_TROUBLESHOOTING.md
├── RNAP-ET4-D3-ACT12-TROUBLESHOOTING-EQUIPO6.pdf  # English D3 / Act. 12 evidence
├── RNAP-ET4-D2.pdf          # English D2 deliverable
├── requirements.txt
└── split_manifest.json      # ids of the train/validation/test splits
```

## How to Run

```bash
pip install -r requirements.txt
python notebooks/baseline_casa_norte.py       # exact baseline, validation-only C, latency
python notebooks/fase2_transformer.py         # final BETO fine-tune (CPU: ~1 h, 4 epochs)
python app/app.py                             # interface v2
```

`requirements.txt` is intentionally small for Hugging Face Spaces: torch, transformers, scikit-learn, numpy, matplotlib and gradio. All data is local JSONL, so `datasets`, `scipy`, `sentencepiece` and `pandas` are no longer required. Model weights are not committed (420 MB); the training script regenerates them in `modelo_casa_norte/`.

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
