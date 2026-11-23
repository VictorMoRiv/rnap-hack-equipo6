# Evidencia de Aprendizaje · Act. 12 — Troubleshooting Report

**Course:** Redes Neuronales y Aprendizaje Profundo (RNAP) · ET4<br>
**Team:** Equipo 6 — Víctor Morales Rivera, Travis Patricio Contreras García, Julio Alexander Esquivel Guerra, Angel Patricio Tamez Martinez<br>
**Project:** Casa Norte — *The Interface That Follows Your Intent*<br>
**Date:** 2026-11-22<br>

---

## 1. Purpose

This report documents the technical incidents found while productionizing the Casa Norte intent-and-slot model and its Gradio interface. For each incident it records the symptom, diagnosis, root cause, corrective action, evidence and observed result. The report is written in technical English because it is the English learning evidence for Activity 12.

## 2. Environment

| Item | Value |
|---|---|
| OS | Arch Linux, CPU-only laptop |
| Python | 3.12, project virtual environment `.venv/` |
| Deep learning | `torch 2.15.0+cpu`, `transformers 5.23.0` |
| Interface | `gradio 6.32.0` |
| Model | `dccuchile/bert-base-spanish-wwm-cased` (BETO), joint intent + BIO slot heads |
| Served bundle | `models/beto_joint/` |
| Demo | `http://localhost:7860`, systemd user unit `casa-norte.service` |

---

## 3. Incident 1 — SDPA attention backend cannot emit attention weights

**Symptom.** The first attention-map smoke test failed with an error equivalent to: `sdpa attention does not support output_attentions=True`. Training and prediction worked, but the explainability tab could not render a heatmap.

**Diagnosis.** In `transformers 5.23.0` the default attention implementation is `sdpa`, which is optimized for speed but does not return raw attention tensors. The plotting routine needs those tensors; the classifier and slot tagger do not.

**Root cause.** Capability mismatch between the fast inference backend and the diagnostic feature, not a defect in the fine-tuned weights.

**Corrective action.**
- Set `DEFAULT_ATTN = "sdpa"` for normal training and inference.
- In `plot_attention_map()`, temporarily switch the encoder to `eager`, run one forward pass with `output_attentions=True`, and restore the previous implementation in a `finally` block.
- Save `output_attentions=True` in the bundle config so the diagnostic path is reproducible.

**Evidence.** `casa_norte/model.py:11`, `casa_norte/model.py:167`, `notebooks/figs/attention_app.png`, and `render_attention()` in `app/app.py`. The smoke test produced an attention matrix with shape `(11, 11)` for an 11-token sentence.

**Result.** Fast path preserved for prediction; attention visualization works on demand without permanently slowing the app.

---

## 4. Incident 2 — Text leakage through repeated surface forms

**Symptom.** Validation scores looked suspiciously high and some Spanish sentences appeared in more than one partition even though the ids were disjoint.

**Diagnosis.** The official MASSIVE split separates sentence ids, but it does not guarantee that identical normalized text is absent from train and validation. A short utterance such as a time request can be repeated with a different id.

**Root cause.** Id-level separation is necessary but insufficient for short, formulaic utterances.

**Corrective action.**
- Added `deduplicate_splits(train, val, test)` in `casa_norte/nlu.py:181`.
- The function normalizes text and removes overlapping train/validation and train/test surface forms before fitting.
- Reran the exact baseline with validation-only `C` selection.

**Evidence.** Deduplicated sizes: train **3,005**, validation **554**, test **837**. `split_manifest.json` still proves id-level separation. `notebooks/metrics_baseline_prod.json` records `C=100`, validation macro-F1 **0.9179**, test accuracy **0.9462**, test macro-F1 **0.9086**, mean/p95 latency **0.857 / 1.010 ms**.

**Result.** Evaluation is leakage-free and the reported baseline metrics are defensible.

---

## 5. Incident 3 — CPU-only training cost and slow attention backend

**Symptom.** The original 3-epoch BETO fine-tune took **46 min 24 s** on CPU. A later 4-epoch run using the `eager` backend was too slow and was stopped after repeated smoke tests; the user explicitly asked not to keep improving the model because it took too long.

**Diagnosis.** There is no CUDA device available. The `eager` backend materializes full attention matrices at every step, multiplying an already expensive CPU fine-tune.

**Root cause.** Hardware limitation amplified by a diagnostic attention implementation used as the default during one training attempt.

**Corrective action.**
- Switched the default backend back to `sdpa`.
- Used small `--limit 32 --epochs 1` smoke tests to validate the full pipeline before any long run.
- Made bundle loading offline and fast with local encoder config and meta-device materialization.
- Kept the already trained `models/beto_joint/` bundle as the served model and documented that no final 4-epoch metrics are claimed.

**Evidence.** `requirements.txt` contains CPU-only packages; `casa_norte/model.py:54` implements local loading; smoke tests measured **2.52–2.88 s** load and about **58 ms** per prediction; `notebooks/metrics_fase2.json` and `PROGRESS.md:57` record the original 46 min 24 s run; `PROGRESS.md:118` records that the retrain was stopped.

**Result.** The demo remains responsive and the evidence set does not claim results from a run that did not finish.

---

## 6. Incident 4 — Offline startup stalled after reboot

**Symptom.** After a machine reboot, `systemd --user is-active casa-norte.service` reported `active`, but `curl http://localhost:7860` returned HTTP code `000`. Logs showed `Warning: You are sending unauthenticated requests to the HF Hub` and no `Running on local URL` line.

**Diagnosis.** The old bundle `models/beto_joint/` contained `tokenizer/`, `model.pt` and `casa_norte_config.json`, but no `encoder_config/` directory. `load_bundle()` therefore fell back to `AutoModel.from_pretrained(config["model_name"])`, which attempted a network request during app startup.

**Root cause.** Incomplete offline bundle: the tokenizer and weights were local, but the encoder configuration was only referenced by its Hub model name.

**Corrective action.**
- Wrote the cached BETO `config.json` into `models/beto_joint/encoder_config/config.json`.
- Kept `AutoConfig.from_pretrained(config_dir, local_files_only=True)` in `casa_norte/model.py:64`.
- Stopped and restarted the user service with the corrected bundle.

**Evidence.** `models/beto_joint/encoder_config/config.json`, `casa_norte/model.py:62-72`, final health check: HTTP **200** and service state `active`.

**Result.** Startup is deterministic and does not require HF Hub access.

---

## 7. Incident 5 — Softmax overconfidence on out-of-scope requests

**Symptom.** Requests such as `transfiere 500 pesos a mi hermana` were forced into one of the eight trained intents. The model has no `out_of_scope` class, so softmax always produces a winner.

**Diagnosis.** A closed-set classifier cannot express “none of the above”. Confidence alone was insufficient because some irrelevant sentences received high scores.

**Root cause.** Model scope mismatch between the 8 supported cards and unrestricted user input.

**Corrective action.**
- Added a confidence threshold from the model bundle.
- Added a conservative keyword guard for obvious out-of-scope domains such as transfers, jokes and food delivery.
- Both paths now render the same clean card: *“No reconozco esa acción todavía.”*

**Evidence.** `out_of_scope()` and `understand()` in `app/app.py`, `notebooks/technical_report.md:131`, `data/mexican_test.jsonl` (8 out-of-scope rows).

**Result.** The demo no longer forces a wrong card during live tests. A learned OOS class remains future work.

---

## 8. Preventive actions adopted

1. Every bundle must contain `tokenizer/`, `encoder_config/`, `model.pt` and `casa_norte_config.json`; loading must use `local_files_only=True`.
2. Deduplicate by normalized text before fitting, not only by id.
3. Run a 32-example smoke test before any full CPU training job.
4. Use `sdpa` by default; enable `eager` only inside the attention-plotting context.
5. Record every metric in a JSON artifact and never claim results from a cancelled run.
6. Verify the demo with both `curl` HTTP status and service state after every restart.

## 9. Conclusion

The incidents were resolved with changes that are visible in the repository: a local-only model bundle, leakage-controlled data splits, a fast default attention backend with on-demand explainability, a clean out-of-scope path, and a verified running demo. The main remaining limitation is the absence of a learned out-of-scope class and of the cancelled 4-epoch transformer export; both are documented instead of being hidden.

## 10. Evidence index

| Artifact | Supports |
|---|---|
| `RNAP-ET4-D2.pdf` | English grammar evidence (D2) |
| `notebooks/technical_report.md` | Full model, evaluation and productionization analysis |
| `notebooks/metrics_fase2.json` | Original trained BETO metrics |
| `notebooks/metrics_baseline_prod.json` | Exact deduplicated baseline and latency |
| `notebooks/figs/attention_app.png` | Attention-map output |
| `split_manifest.json` | Id-level split evidence |
| `casa_norte/model.py` | Offline loading and attention backend fix |
| `casa_norte/nlu.py` | Deduplication and normalization fixes |
| `app/app.py` | Interface, state actions and out-of-scope handling |
| `PROGRESS.md` | Chronological record of CP1–CP5 |
| Live `http://localhost:7860` | Final verified demo, HTTP 200 |
