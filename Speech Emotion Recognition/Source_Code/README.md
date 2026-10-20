# Speech Emotion Recognition - MFCC + CNN-BiLSTM-Attention

Recognises 8 emotions (neutral, calm, happy, sad, angry, fearful, disgust, surprised) from speech.
Evaluated with **F1-score** (macro / weighted / micro / per-class) under a **speaker-independent** split.

## Quick start
```bash
pip install -r requirements.txt
python run_pipeline.py --download --data data/RAVDESS     # real experiment (downloads ~215 MB from Zenodo)
python build_report.py --name "Your Name" --roll "ID"      # -> Speech_Emotion_Recognition_Report.pdf
python build_ppt.py    --name "Your Name"                  # -> Speech_Emotion_Recognition_Presentation.pptx
python predict.py path/to/clip.wav                         # inference on one file
pytest -q                                                  # unit tests
```
If the Zenodo download is blocked, download `Audio_Speech_Actors_01-24.zip` from
https://zenodo.org/records/1188976, unzip into `data/RAVDESS/`, and omit `--download`.

`python run_pipeline.py --synthetic` runs a fast offline smoke test on generated audio. **Its numbers are not real results.**

## Pipeline
`ser/data.py` download + labels + actor split -> `ser/features.py` resample/trim/normalise/augment + MFCC(+delta,+delta-delta)
-> `ser/train.py` SVM baseline + CNN-BiLSTM-Attention -> `ser/evaluate.py` F1 metrics, confusion matrices, plots.
All settings are in `ser/config.py`. Outputs: `results/metrics.json`, figures, classification reports; `models/ser_model.keras`.

## Split
Actors 1-16 train, 17-20 validation, 21-24 test (no speaker overlap, asserted in code). Augmentation is applied to training clips only.
