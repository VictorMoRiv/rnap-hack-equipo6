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

> Phase 1 (this checkpoint) covers steps 1–3. The interface (steps 4–5) is built in Phase 2.

## Approach

Two levels of the course are compared on the same test set, as required by the event rules:

| Level | Intent detection | Key data (slots) |
|---|---|---|
| **Baseline (trained by the team)** | TF-IDF (word 1–2 and character 3–5 grams) + Logistic Regression | Word-level token classifier (`DictVectorizer` + Logistic Regression) |
| **Pre-trained, no fine-tuning** | Multilingual zero-shot NLI (`MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`) | Not extracted by this model |

Phase 2 replaces the baseline with a fine-tuned Spanish Transformer (BETO or multilingual DistilBERT) with two heads: intent classification and BIO slot tagging.

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

Sentences outside these intents are handled as follows: if the classifier's confidence is below 0.50, the system returns a clarification reply instead of forcing a card. A dedicated `out_of_scope` class and a better calibration are Phase 2 work.

## Data

- **MASSIVE**, locale `es-ES`, version 1.1: 16,521 sentences, 60 intents and 55 slot types ([Hugging Face](https://huggingface.co/datasets/AmazonScience/massive)).
- A local copy of the filtered data is in `data/` (`massive_es-ES_train.jsonl`, `massive_es-ES_validation.jsonl`, `massive_es-ES_test.jsonl`).
- The official partitions are kept: **train = 3,127**, **validation = 565**, **test = 837** sentences for the 8 classes. `validation` is used only to choose hyperparameters; `test` is used once, for the final metrics.
- `split_manifest.json` stores the sentence ids of each partition to prove there is no leakage.
- **Extra test set:** 30 Mexican-style sentences written by the team. They are **not** used for training; evaluation on them is pending (Phase 2).

## Evaluation

Results on the MASSIVE `test` split (837 sentences):

| Metric | Baseline (TF-IDF + LogReg) | Pre-trained zero-shot |
|---|---|---|
| Intent accuracy | **0.9558** | **0.5436** |
| Intent macro-F1 | **0.9292** | **0.5706** |
| Slot span-F1 | **0.6659** | — (not extracted) |
| Runtime | 3 min 22 s (train + test) | 40 min (test only, CPU) |
| Accuracy on 30 Mexican sentences | pending | pending |

The TF-IDF baseline is strong in-domain: it saw the training vocabulary and the eight intents. The zero-shot model never saw MASSIVE; its 0.54 accuracy shows that label wording and domain shift matter a lot. It stays in the comparison as the "pre-trained without fine-tuning" level and as a candidate for out-of-scope sentences, not as the final model.

## Interface

Planned for Phase 2 with **Gradio**: one card per intent (alarm, weather, shopping list, calendar, music, lights) that appears when detected and hides the rest, with editable fields and a button that changes the app state.

- During development: run from Google Colab with `share=True`.
- Final version: Hugging Face Spaces (free CPU) or the team laptop.
- Live demo link: pending.

**Planned extra:** a cutaway "home simulator" in the style of The Sims inside the interface: little characters, lights that turn on, and an alarm on the bed that reacts to the user's commands.

## Repository Structure

```
.
├── app/                     # interface (Phase 2)
├── data/                    # local copy of the filtered MASSIVE es-ES files
├── notebooks/
│   ├── casa_norte_8intenciones.py   # training + evaluation script
│   └── technical_report.md          # sections 1–3 of the report
├── PROGRESS.md              # checkpoint log (CP1–CP4)
├── README.md
├── RNAP-ET4-D2.pdf          # English D2 deliverable
├── requirements.txt
└── split_manifest.json      # ids of the train/validation/test splits
```

## How to Run

```bash
pip install -r requirements.txt
python notebooks/casa_norte_8intenciones.py
```

`requirements.txt` pins `datasets==2.19.0` because MASSIVE still ships a loading script and `datasets` 3+ removed script support. The script downloads the dataset through Hugging Face; the copy in `data/` can be used instead.

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
| Thu, Nov 12, 2026 | CP3–CP4 · Transformer + interface v1 |
| Thu, Nov 19, 2026 | Final presentation in English + live judge test |

## Authors

Travis Contreras, Julio Esquivel, Víctor Morales, Angel Tamez · RNAP Hack 2026 · Redes Neuronales y Aprendizaje Profundo

## License / Data Credits

MASSIVE dataset by Amazon Science (CC BY 4.0).
