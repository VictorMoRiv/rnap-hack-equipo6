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
## CP3 - 2026-11-12

- **What we have done**: 

- **What we haven't done yet**: 

- **Next step**: 

- **Who did what**:
  
  Travis:

  Julio:
  
  Tamez:
  
  Victor:


#
## CP4 - 2026-11-12

- **What we have done**: 

- **What we haven't done yet**: 

- **Next step**: 

- **Who did what**:
  
  Travis:

  Julio:
  
  Tamez:
  
  Victor:
