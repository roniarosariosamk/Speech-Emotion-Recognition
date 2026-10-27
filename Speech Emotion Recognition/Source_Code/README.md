# Speech Emotion Recognition - MFCC + CNN-BiLSTM-Attention

Individual major project (Machine Learning Internship): *Develop a machine learning system for emotion recognition in speech using MFCC features and a deep learning model.*

The system classifies a speech clip into **8 emotions** - neutral, calm, happy, sad, angry, fearful, disgust, surprised - using MFCC (+ delta + delta-delta) features and a compact **CNN -> BiLSTM -> temporal attention** network, and compares it with an RBF-SVM baseline. Evaluation is **actor-independent** and centred on **Macro F1**.

## Results (RAVDESS, held-out test actors 21-24, 240 clips)

| Model | Val Macro F1 | Test accuracy | Test Macro F1 | Test weighted F1 |
|---|---|---|---|---|
| Experiment 1 - first (wide) CNN-BiLSTM-Attention, 341,064 params | 0.4484 | 0.4708 | 0.4292 | 0.4323 |
| Experiment 2 - compact network, 79,080 params | 0.4873 | 0.5375 | 0.5231 | 0.5343 |
| **Experiment 3 - compact network, checkpoint by val Macro F1 (selected)** | **0.5161** | **0.5667** | **0.5307** | **0.5384** |
| Experiment 4 - same network, stronger dropout (0.35/0.40/0.35/0.50), 1 run | 0.4794 | 0.5000 | 0.4783 | 0.4769 |
| SVM baseline (RBF, C=100, gamma=0.001) | - | 0.4667 | 0.4464 | 0.4624 |

Selected final model: **Experiment 3** (`models/experiment_03/ser_model_best.keras` + `norm.npz`), chosen by the highest *validation* Macro F1. Macro precision 0.5743, macro recall 0.5742. Full tables: `results/final/final_summary.md` / `.json`.

Per-class test F1: surprised 0.806, angry 0.716, calm 0.648, disgust 0.647, neutral 0.415, happy 0.400, fearful 0.381, sad 0.233.

> **Experiment 4** (dropout 0.35 / 0.40 / 0.35 / 0.50 instead of 0.30 / 0.35 / 0.30 / 0.50) was trained once on the real RAVDESS data (best epoch 26, stopped at 38, 204 s on one CPU core). It scored lower than Experiment 3 on both validation (0.4794 vs 0.5161) and test Macro F1 (0.4783 vs 0.5307), so **Experiment 3 remains the final model**. These are single runs with one seed; the gap is consistent with the extra regularisation not helping here but is not proof, since run-to-run variation was not measured.
> The SVM baseline reproduced exactly on the dataset copy used for this run (identical confusion matrix), confirming the data and pipeline match the original experiments.

## Features
- Reproducible pipeline: fixed seed, central config (`ser/config.py`), 25 automated tests.
- Actor-independent split with leakage assertions; augmentation on training data only; normaliser fitted on training data only.
- MFCC + delta + delta-delta features, shape (130, 120) per 3 s clip.
- CNN-BiLSTM-Attention model with custom attention layer; checkpoint and early stopping on validation Macro F1.
- RBF-SVM baseline tuned on validation actors only.
- Inference script with input validation, any experiment selectable.
- Automatic result verification (metrics recomputed from confusion matrices), model selection, report PDF and presentation generated from the real metrics.

## Dataset
RAVDESS speech (Livingstone & Russo, 2018; CC BY-NC-SA 4.0): 1,440 clips, 24 actors. Download `Audio_Speech_Actors_01-24.zip` from https://zenodo.org/records/1188976 and unzip into `data\RAVDESS\` (folders `Actor_01` ... `Actor_24`), or let the code download it with `--download`.

## Installation (Windows PowerShell)
```powershell
cd "C:\Users\ronia\OneDrive\Documents\Emotion Recognition\Speech Emotion Recognition\Source_Code"
python -m venv .venv                      # skip if .venv already exists
.\.venv\Scripts\Activate.ps1              # if blocked: Set-ExecutionPolicy -Scope Process Bypass
pip install -r requirements.txt
```
Developed with Python 3.13, TensorFlow 2.21.0, Keras 3.15.1, librosa 1.0.0, scikit-learn.
**Use Python 3.10-3.13**: TensorFlow 2.21 has no wheels for Python 3.14. Check your environment with `python check_env.py`; to create a 3.13 environment use `py -3.13 -m venv .venv`. Without TensorFlow the model-dependent tests are skipped, not failed, so a "passed" result must show **0 skipped** to count as a full run.

## Project structure
```
Source_Code/
  ser/            config.py data.py features.py model.py evaluate.py train.py   (Experiments 1-3)
                  model_exp4.py train_exp4.py                                    (Experiment 4)
                  final.py diagrams.py                                           (verification, selection, figures)
  run_pipeline.py run_experiment_02.py run_experiment_03.py run_experiment_04.py
  summarize_results.py predict.py build_report.py build_ppt.py
  tests/          test_pipeline.py test_project.py
  models/         ser_model_best.keras + norm.npz (Exp 1), experiment_02/, experiment_03/, experiment_04/ (after training)
  results/        metrics + figures per experiment, final/ (selected model, comparison, summary)
  docs/           AUDIT_REPORT.md VIVA_GUIDE.md GITHUB_SYNC.md
  ..\             Speech_Emotion_Recognition_Report.pdf / _Presentation.pptx are written here (project root)
```

## Finish the project in ONE command (trains Experiment 4 on real RAVDESS)
```powershell
python finalize_project.py --name "Ronia Rosario K" --roll "YOUR_ID"
```
Put `Audio_Speech_Actors_01-24.zip` in `data\` first (or unzip into `data\RAVDESS`; `--download` fetches it if online). The script trains Experiment 4 (about 5-10 min), re-verifies all metrics, selects the final model by validation Macro F1 (Experiment 3 stays if Experiment 4 is not better), rebuilds the PDF report and PowerPoint with the real numbers, and runs `verify_release.py` on a real test-actor clip. The delivered package was built **before** this step, so it reports Experiment 4 as not run.

## How to train
```powershell
python run_experiment_04.py --data data\RAVDESS --epochs 60        # Experiment 4 -> models\experiment_04, results\experiment_04
```
Experiments 1-3 are already trained; their scripts (`run_pipeline.py`, `run_experiment_02.py`, `run_experiment_03.py`) refuse to overwrite existing outputs unless `--overwrite` is given.

## How to evaluate
```powershell
python verify_release.py                  # ONE command: env + 25 tests (0 skips) + metrics + inference -> RELEASE VERIFIED
                                          # (add --audio path\to\real_clip.wav for a real-audio check)
python check_env.py                       # Python / package check
python -m pytest -q                       # automated checks (expect: 25 passed, 0 skipped)
python summarize_results.py               # verifies all metrics, selects the final model, writes results\final\
python summarize_results.py --force experiment_03    # optionally pin a model
```

## How to predict an emotion
```powershell
python predict.py path\to\clip.wav                   # selected final model
python predict.py path\to\clip.wav --all --json      # all 8 probabilities as JSON
python predict.py path\to\clip.wav --experiment experiment_03
```
Input: `.wav`, `.flac`, `.ogg` or `.mp3`, speech, ideally 1-5 s (trimmed to silence-free speech and cropped / padded to 3 s; resampled to 22.05 kHz mono). Silent, empty, corrupt or unsupported files give a clear error. Prediction uses the saved training mean / std - nothing is fitted on the input.

## Build the report and slides
```powershell
python build_report.py --name "Your Name" --roll "ID"     # writes ..\Speech_Emotion_Recognition_Report.pdf
python build_ppt.py --name "Your Name"                    # writes ..\Speech_Emotion_Recognition_Presentation.pptx
```

## Model artifacts
`models\experiment_0X\ser_model_best.keras` (best validation-Macro-F1 checkpoint) and `norm.npz` (training mean `mu` and std `sd`, shape 1x1x120) always belong together. Experiment 1 files are in `models\` itself.

## Known limitations
Acted speech from 24 actors; only 4 test actors (noisy estimate); sad and fearful recall are low (0.156 and 0.250) and neutral is over-predicted; the experiment comparison used test scores as well as validation scores, so the test number is mildly optimistic; no noise-robustness or cross-corpus evaluation; Experiment 4 is a single run. This is a research prototype and not a general emotion detector. See `docs\AUDIT_REPORT.md`.
