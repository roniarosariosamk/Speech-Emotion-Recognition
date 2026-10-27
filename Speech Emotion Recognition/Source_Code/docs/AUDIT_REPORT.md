# Project audit and corrections

Audit date: 25 Oct 2026. Source of truth: the files in `Source_Code/`. The RAVDESS audio was **not** part of the audited archive, so nothing could be re-trained; the audit therefore (a) read every source file, (b) loaded every saved checkpoint, (c) recomputed all metrics from the saved confusion matrices, (d) ran tests and an Experiment 4 smoke run on synthetic audio.

## Critical problems found and fixed
| # | Problem | Evidence | Fix |
|---|---|---|---|
| 1 | `predict.py` loaded `models/ser_model_best.keras`, which is the **Experiment 1** network (341,064 params), not the best model | checkpoint inspection | Rewritten: loads the selected final model (Experiment 3) with its own `norm.npz`, selectable with `--experiment`, input validation, 8 finite probabilities checked |
| 2 | Experiment 4 was never configured: `model_exp4.py` had **identical** dropout values to Experiment 3; `train_exp4.py` / `run_experiment_04.py` did not exist | `diff ser/model.py ser/model_exp4.py` showed only imports differed | `model_exp4.py` now has 0.35/0.40/0.35/0.50; new `train_exp4.py` imports `model_exp4` explicitly; new `run_experiment_04.py` writes only to `experiment_04` folders and refuses to overwrite |
| 3 | `build_report.py` / `build_ppt.py` read `results/metrics.json` (Experiment 1: Macro F1 0.4292) and described an architecture that no longer exists (3 Conv blocks, BiLSTM 96, Dense 128); `diagram_architecture.png` showed the old model | script and diagram code | Both generators rewritten to read the verified final summary; architecture diagram generated from the real model configuration |
| 4 | Top-level README stated that the report/PPT were synthetic smoke-test output | README text | README rewritten |
| 5 | `evaluate.compute_metrics` called `f1_score` without `labels=`: a class absent from both truth and predictions would silently drop out of the macro average | code review + failing unit test | `labels=range(n_classes)` added to macro / weighted / micro F1. Existing real-data numbers verified unchanged (Exp 3 recomputed: Macro F1 0.5306963) |
| 6 | `from tensorflow import keras` fails in some installs (`keras.saving` missing) while `model_exp4.py` used `import keras` | import error in a clean environment | `ser/model.py`, `ser/train.py` use `import keras` (same Keras 3 API, same layer registration, checkpoints load identically) |
| 7 | Unreadable / silent audio gave obscure library errors | code review | `load_audio` raises clear `ValueError`s |
| 8 | Re-running an old experiment script would silently overwrite finished results | scripts had no guard | `--overwrite` guard on `run_pipeline.py`, `run_experiment_02/03/04.py` |
| 9 | Tests covered 5 cases only | `tests/test_pipeline.py` | `tests/test_project.py` adds 20 tests (config, labels, discovery, features, both models, save/load, inference, errors, metrics, isolation) |

## Verified and found correct (no change needed)
- Label order `neutral, calm, happy, sad, angry, fearful, disgust, surprised` (ids 0-7) used identically by data, models, metrics and inference.
- Split: train actors 1-16 (960 clips, 2,880 after 2 augmented copies), val 17-20 (240), test 21-24 (240); overlap asserted in code and in tests; validation and test are never augmented.
- Normaliser fitted on augmented *training* features only; same `mu/sd` applied to val, test and inference.
- Feature tensor (130, 120): 40 MFCC + delta + delta-delta, sr 22,050, FFT 2,048, hop 512, 128 mels.
- SVM baseline: scaler inside the pipeline (fit on training only), C/gamma chosen on validation actors only via `PredefinedSplit` on Macro F1, refit on training only, evaluated once on test.
- Experiment 3 checkpoint: loads, input (None,130,120), output 8 softmax probabilities, 79,080 parameters; Exp 2 has the identical architecture (Exp 3 differs in checkpoint criterion and epoch limit, not in layers).
- Stored metrics for Exp 1-3 are internally consistent: accuracy, Macro F1, weighted F1 recomputed from confusion matrices match to 1e-9.
- Harmless Keras notice when loading `AttentionPool` ("build() was called ... un-built state"): the model saves, loads and reproduces predictions, so the layer was deliberately left unchanged.

## Methodological caveats (reported honestly in the report)
- Experiments 1-3 were compared while test results were visible; the final model choice rule is validation-only and agrees (Exp 3 has the best validation *and* test Macro F1), but the test score remains slightly optimistic.
- Four test actors / 240 clips: differences of 1-2 F1 points are not statistically meaningful.

## Not done / not possible in this environment
- Nothing outstanding for training: Experiment 4 was trained on real RAVDESS data in the final build (see below).
- Inference smoke test used a *synthetic* clip (technical check only; its predicted label is meaningless). Replace with a real RAVDESS clip, e.g. `python predict.py data\RAVDESS\Actor_21\03-01-05-01-01-01-21.wav`.

## Final build addendum: Experiment 4 on real data
- Data: RAVDESS speech obtained from a public GitHub mirror (Zenodo was unreachable from the build sandbox); 1,440 clips, 24 actors x 60, 96 neutral + 192 per other emotion, mono 48 kHz, filenames per the official scheme. Not checksum-verified against Zenodo, but the SVM baseline reproduced **exactly** (identical confusion matrix, Macro F1 0.4464177), which would be very unlikely with different audio. One pair of byte-identical clips exists inside Actor 07 (training split), so it cannot leak between splits.
- Experiment 4 (one run, seed 42): val Macro F1 0.4794, test accuracy 0.5000, test Macro F1 0.4783, weighted F1 0.4769, best epoch 26 of 38. Worse than Experiment 3 -> **Experiment 3 stays selected** (validation-only rule).
- Experiments 1-3 artefacts re-checked: all 36 files byte-identical to the original upload.
