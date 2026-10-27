# Speech Emotion Recognition - MFCC + CNN-BiLSTM-Attention
Individual Major Project (Machine Learning Internship)

## Package contents
| Item | File / folder |
|---|---|
| Source code | `Source_Code/` |
| Project report (PDF) | `Speech_Emotion_Recognition_Report.pdf` |
| Presentation (PPT) | `Speech_Emotion_Recognition_Presentation.pptx` |

## Project summary
Recognises 8 emotions (neutral, calm, happy, sad, angry, fearful, disgust, surprised) from speech using
MFCC (+ delta, delta-delta) features and a CNN -> BiLSTM -> Attention deep model, compared with an RBF-SVM baseline.
Evaluation uses F1-score (macro / weighted / micro / per-class) on a speaker-independent split
(actors 1-16 train, 17-20 validation, 21-24 test).

## Final results (RAVDESS, unseen test actors 21-24)
Selected model: **Experiment 3** - test Macro F1 **0.5307**, accuracy 56.67%, weighted F1 0.5384 (validation Macro F1 0.5161); SVM baseline Macro F1 0.4464.
Experiment 4 (stronger dropout, single run) scored lower (test Macro F1 0.4783), so Experiment 3 remains the final model.
All numbers in the PDF report and PowerPoint are generated from the verified metrics (`Source_Code/results/final/final_summary.json`).

## How to run (from inside `Source_Code/`, Windows PowerShell)
```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest -q                                     # 25 automated tests
python summarize_results.py                             # verify metrics + select final model
python predict.py path\to\clip.wav                      # predict an emotion
python run_experiment_04.py --data data\RAVDESS         # optional: run Experiment 4 on the real data
python build_report.py --name "Your Name"               # regenerate PDF report
python build_ppt.py --name "Your Name"                  # regenerate PowerPoint
```
Dataset: download `Audio_Speech_Actors_01-24.zip` from https://zenodo.org/records/1188976 into `Source_Code/data/RAVDESS/`. Full instructions: `Source_Code/README.md`; audit: `Source_Code/docs/AUDIT_REPORT.md`; viva notes: `Source_Code/docs/VIVA_GUIDE.md`.

## Code structure (`Source_Code/`)
- `ser/config.py` - all settings
- `ser/data.py` - download, label parsing, speaker-independent split
- `ser/features.py` - preprocessing, MFCC, augmentation, normalisation
- `ser/model.py` - CNN-BiLSTM-Attention network
- `ser/train.py` - SVM baseline and deep model training
- `ser/evaluate.py` - F1 metrics, confusion matrices, plots
- `run_pipeline.py`, `predict.py`, `build_report.py`, `build_ppt.py`, `tests/`

Dataset: RAVDESS (Livingstone & Russo, 2018), CC BY-NC-SA 4.0.
