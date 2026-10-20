Speech Emotion Recognition - MFCC + CNN-BiLSTM-Attention
Individual Major Project (Machine Learning Internship)

Package contents
Item	File / folder
Source code	Source_Code/
Project report (PDF)	Speech_Emotion_Recognition_Report.pdf
Presentation (PPT)	Speech_Emotion_Recognition_Presentation.pptx
Project summary
Recognises 8 emotions (neutral, calm, happy, sad, angry, fearful, disgust, surprised) from speech using MFCC (+ delta, delta-delta) features and a CNN -> BiLSTM -> Attention deep model, compared with an RBF-SVM baseline. Evaluation uses F1-score (macro / weighted / micro / per-class) on a speaker-independent split (actors 1-16 train, 17-20 validation, 21-24 test).

How to run (from inside Source_Code/)
pip install -r requirements.txt
python run_pipeline.py --download --data data/RAVDESS     # downloads RAVDESS (~215 MB) and trains/evaluates
python build_report.py --name "Your Name" --roll "ID"      # regenerates the PDF report from results/metrics.json
python build_ppt.py --name "Your Name"                     # regenerates the PowerPoint
python predict.py path/to/clip.wav                         # predict emotion for one audio file
pytest -q                                                  # unit tests
If the download is blocked, get Audio_Speech_Actors_01-24.zip from https://zenodo.org/records/1188976, unzip it into Source_Code/data/RAVDESS/, and run without --download.

IMPORTANT - about the results in this package
The PDF and PPT included here were generated from a synthetic smoke-test run (the build environment could not reach the RAVDESS download). They are marked "synthetic - not RAVDESS" and only prove the pipeline works. Run the real experiment above, regenerate the report and PPT, and replace the two files before submitting.

Code structure (Source_Code/)
ser/config.py - all settings
ser/data.py - download, label parsing, speaker-independent split
ser/features.py - preprocessing, MFCC, augmentation, normalisation
ser/model.py - CNN-BiLSTM-Attention network
ser/train.py - SVM baseline and deep model training
ser/evaluate.py - F1 metrics, confusion matrices, plots
run_pipeline.py, predict.py, build_report.py, build_ppt.py, tests/
Dataset: RAVDESS (Livingstone & Russo, 2018), CC BY-NC-SA 4.0.
