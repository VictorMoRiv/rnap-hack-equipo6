#
## CP1 - 2026-11-05

- **What we have done**:
  - Created the public repository with the required structure (README.md, PROGRESS.md, notebooks/, app/, data/, requirements.txt).
  - Wrote the first version of README.md with the Track 5 brief and the team members.

- **What we haven't done yet**:
  - Downloading and splitting the MASSIVE dataset.
  - The baseline model and the zero-shot comparison.

- **Next step**:
  - Download MASSIVE es-ES, select the 8 intents, split train/dev/test and build the TF-IDF baseline.

- **Who did what**:
  Victor: created the repository and the initial project structure.
  Julio: wrote the README.md and organized the team.
  Travis: started the model code.
  Tamez: started the English D2 grammar table.

#
## CP2 - 2026-11-08

- **What we have done**:
  - Created the public repository with the required structure (README.md, PROGRESS.md, notebooks/, app/, data/, requirements.txt).
  - Downloaded MASSIVE es-ES v1.1 and filtered 8 intents: alarm_set, alarm_query, weather_query, lists_createoradd, lists_remove, calendar_set, play_music and control_luces.
  - Split the data before training into train/dev/test using the official partitions (3,127 / 565 / 837 sentences) and saved the ids in split_manifest.json.
  - Built the TF-IDF + logistic regression baseline (word 1-2 and character 3-5 grams).
  - Implemented the multilingual zero-shot classifier without fine-tuning and evaluated it on the same test set.
  - Measured both models on the test set (837 sentences): TF-IDF accuracy 0.9558 and macro-F1 0.9292; zero-shot accuracy 0.5436 and macro-F1 0.5706.
  - Wrote the training script notebooks/casa_norte_8intenciones.py and the technical report notebooks/technical_report.md with sections 1-3 (problem and data, text representation, and the comparative table with metrics and processing time).
  - Completed the English D2 grammar table (Present Perfect, Second Conditional, Perfect Conditional).

- **What we haven't done yet**:
  - Fine-tuning the Transformer (BETO or multilingual DistilBERT) for intent and slot extraction.
  - Five or more error cases with their possible cause (Phase 2).
  - The Gradio interface with one card per intent and a working action.
  - The 30 Mexican-style sentences and the out-of-scope behavior.

- **Next step**:
  - Update the README with the measured metrics and the list of 8 intents.
  - Start Phase 2: fine-tune BETO/DistilBERT, add slots and build interface v1.

- **Who did what**:
  Travis: wrote the model code (TF-IDF baseline, zero-shot evaluation and slot extractor).
  Julio: wrote the README.md, coordinated the team and helped with the code.
  Tamez: completed the English D2 homework (grammar table).
  Victor: set up the repository and GitHub structure, added the data split and the training script, and helped with the code.

#
## CP3 - 2026-11-16

- **What we have done**:
  - Froze the team's 30 Mexican-style sentences (`data/mexican_test.jsonl`, 25 in-domain + 5 out-of-scope) before any Phase 2 training.
  - Re-evaluated the Phase 1 baseline on those sentences: intent accuracy 0.920, macro-F1 0.9214, slot span-F1 0.5366.
  - Fine-tuned BETO (`dccuchile/bert-base-spanish-wwm-cased`) with two heads: 8-way intent classification and BIO slot tagging (19 canonical slot types + O).
  - Trained 3 epochs on the official train split (3,127 sentences), selected the best epoch on validation and evaluated once on test: intent accuracy 0.9725, macro-F1 0.9602, slot span-F1 0.6753 (CPU, 46 min 24 s).
  - Evaluated on the 30 Mexican sentences: intent accuracy 0.960 (24/25 in-domain), macro-F1 0.9643, slot span-F1 0.7600.
  - Saved metrics, confusion matrix, per-class F1, training curves and 8 error cases in `notebooks/`.
  - Wrote sections 4 and 5 of `notebooks/technical_report.md`: Transformer solution, configuration, hyperparameter effects, test and Mexican results, and 5+ error cases with causes.

- **What we haven't done yet**:
  - A learned `out_of_scope` class; the model alone rejects 0/5 OOS sentences.
  - Slot recall is 0.5765; the feature baseline is still slightly better at exact boundaries (0.7079).
  - Commits CP3/CP4 and the English deliverables.

- **Next step**:
  - Build interface v1 on top of the fine-tuned model (CP4).
  - Prepare the Troubleshooting Report and D3.

- **Who did what**:
  - Victor: fine-tune, evaluation, figures, error analysis and report sections 4–5, with AI pair-programming.
  - Travis: baseline and dataset review.
  - Julio: coordination and README review.
  - Tamez: English deliverables.


#
## CP4 - 2026-11-16

- **What we have done**:
  - Built the Gradio interface v1 in `app/app.py`, connected to `models/beto_joint`.
  - One card per intent (alarm, weather, shopping list, calendar, music, lights); the detected card appears and the others stay hidden.
  - Editable fields: time, date, event, place, item, song/artist/playlist, room/action/color.
  - Working actions with per-session state: save alarm, add/remove from the shopping list, save event, update lights, set "now playing" and open a YouTube search.
  - Normalization: "seis y media" → 06:30, "el viernes" → next Friday, list item extraction, light action from verbs, adjacent-slot merging.
  - Out-of-scope guard and confidence threshold (0.55) with a clean "No reconozco esa acción todavía" card; low confidence no longer forces a guessed card.
  - Smoke-tested the app (launch, intent cards, fields and actions) on the team laptop.

- **What we haven't done yet**:
  - Public deployment (Hugging Face Spaces or share link).
  - Learned OOS class and better slot recall.
  - Final presentation in English and the judge's live test.

- **Next step**:
  - Deploy the app, record the final demo, and prepare the final presentation.
  - Commit CP3 and CP4 during class time.

- **Who did what**:
   - Victor: interface v1, NLU normalization and integration with the model, with AI pair-programming.
   - Travis: model review.
   - Julio: coordination and demo script.
   - Tamez: English deliverables.

#
## CP5 - 2026-11-22

- **What we have done**:
  - Audited train/validation/test leakage by id and normalized surface form; deduplicated sizes are 3,005 / 554 / 837, with `split_manifest.json` still proving id-level separation.
  - Added 64 labeled synthetic train-only phrases (63 survive the held-out text check) and expanded the Mexican test set to 38 sentences: 30 in-domain + 8 out-of-scope, never used for training.
  - Rebuilt the exact TF-IDF (1,2) + Logistic Regression baseline with validation-only `C=100`, latency measurement and `notebooks/metrics_baseline_prod.json`.
  - Refactored `casa_norte/model.py` for offline bundle loading, fast `sdpa` attention and on-demand `eager` attention maps; smoke test measured 2.52–2.88 s load and ~58 ms prediction.
  - Rebuilt the Gradio app as v2: 8 groups, 18 editable fields, real per-session state, out-of-scope card, clarity banner, six examples and a Técnico attention tab.
  - Relaunched the local service at `http://localhost:7860` (HTTP 200) and added production cells to `notebooks/entrenamiento.ipynb` plus Section 6 of `notebooks/technical_report.md`.
  - Reduced `requirements.txt` to the six packages needed by Hugging Face Spaces.
  - Wrote the English Activity 12 troubleshooting evidence (`EVIDENCIA_APRENDIZAJE_ACT12_TROUBLESHOOTING.md` and `RNAP-ET4-D3-ACT12-TROUBLESHOOTING-EQUIPO6.pdf`) with five incidents, fixes and an evidence index.

- **What we haven't done yet**:
  - The 4-epoch transformer retrain that writes `modelo_casa_norte/`, its calibrated threshold sweep and its final attention map; stopped at the user's request to keep the model work short.
  - Public deployment, a learned `out_of_scope` class and better slot recall.

- **Next step**:
  - Present the working local demo and the current evidence in technical English; keep the app running for the judge's test.

- **Who did what**:
  - Victor: production code, data audit, baseline rerun, app v2, local service and report Section 6, with AI pair-programming.
  - Travis: baseline and model review.
  - Julio: coordination and demo script.
  - Tamez: English deliverables.
