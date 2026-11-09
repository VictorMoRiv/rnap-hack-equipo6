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

## Approach

Two levels of the course are compared, as required by the event rules:

| Level | Intent detection | Key data (slots) |
|---|---|---|
| **Baseline (from scratch)** | TF-IDF |  MASSIVE |
| **Pre-trained Transformer** | Clasificador zero-shot multilingüe | MASSIVE |


## Intents

Trained only on a selected subset of **7 intents** from MASSIVE (instead of all 60):

- _TODO: list the 7 intents (e.g. alarm, weather, shopping list, calendar, ...)_

Sentences outside these intents are handled gracefully (_TODO: describe fallback behavior_).

## Data

- **MASSIVE**, locale `es-ES`, 60 intents with labeled slots: https://huggingface.co/datasets/AmazonScience/massive
- **Extra test set:** 30 Mexican-style sentences written by the author, used to test robustness with informal Mexican Spanish.
- Data is split into **train / validation / test** before training. The test set is used only for the final metrics.

## Evaluation

| Metric | Baseline | Transformer |
|---|---|---|
| Intent accuracy (MASSIVE test) | _TODO_ | _TODO_ |
| Intent F1 (MASSIVE test) | _TODO_ | _TODO_ |
| Slots extracted correctly (%) | _TODO_ | _TODO_ |
| Intent accuracy (30 Mexican-style sentences) | _TODO_ | _TODO_ |

## Interface

Built with **Gradio**. Each intent has its own card (e.g. alarm, weather, shopping list, calendar) that appears when detected and hides the others.

- During development: run from Google Colab with `share=True`.
- Final version: Hugging Face Spaces (free CPU) or local laptop.
- _TODO: add live demo link / screenshots._

**Extra feature:** a cutaway "home simulator" in the style of The Sims inside the interface: little characters, lights that turn on, and an alarm on the bed that reacts to the user's commands.

## Repository Structure

```
.
├── data/            # MASSIVE + Frases de practica
├── notebooks/       # VS (baseline, fine-tuning, evaluation)
├── src/             # Modelo y interfaz del codigo
├── app.py           # App
├── requirements.txt
└── README.md
```

## How to Run

```bash
pip install -r requirements.txt
python app.py
```


## Judging Criteria (Track 5)

- **Results:** right intent and right data on the judge's unseen sentences.
- **Interface:** the card changes with the intent and its action works.
- **Robustness:** informal Mexican Spanish and sentences outside the chosen intents.
- **Clarity:** the user understands what the assistant understood.

## Event Rules Followed

- Clean train / validation / test split; test set used only for final metrics.
- Only public, free data or data created by the author; synthetic data is labeled as such.
- AI tools were used for writing code; the author can explain the whole project.
- Scope kept small: one complete flow over several half-finished features.

## Timeline

| Date | Checkpoint |
|---|---|
| Thu, Nov 5, 2026 | CP1 |
| Thu, Nov 8, 2026 | CP2 |
| Thu, Nov 12, 2026 | CP3-CP4 |
| Thu, Nov 19, 2026 | Final presentation in English + live judge test |

## Authors

Travis Contreras, Julio Esquivel, Víctor Morales, Angel Tamez  · RNAP Hack 2026 · Redes Neuronales y Aprendizaje Profundo

## License / Data Credits

MASSIVE dataset by Amazon Science (CC BY 4.0).
