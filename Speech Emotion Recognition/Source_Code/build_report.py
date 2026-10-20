"""Generate the project-report PDF from results/metrics.json  (python build_report.py [--name "Your Name"])."""
import argparse
import json
from datetime import date
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether, PageBreak, PageTemplate,
                                Paragraph, Spacer, Table, TableStyle)
from reportlab.platypus.tableofcontents import TableOfContents

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
INDIGO, SLATE, LIGHT, AMBER = colors.HexColor("#4F46E5"), colors.HexColor("#1E293B"), colors.HexColor("#EEF2FF"), colors.HexColor("#F59E0B")

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="Student Name"); ap.add_argument("--roll", default="")
ap.add_argument("--out", default=str(ROOT / "Speech_Emotion_Recognition_Report.pdf"))
args = ap.parse_args()
M = json.loads((RES / "metrics.json").read_text())
SYN = M["synthetic"]
deep, svm, tr = M["deep"], M["svm"], M["training"]

ss = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=ss["BodyText"], fontName="Helvetica", fontSize=10, leading=15, alignment=TA_JUSTIFY, textColor=SLATE, spaceAfter=7)
H1 = ParagraphStyle("h1", parent=ss["Heading1"], fontName="Helvetica-Bold", fontSize=17, leading=21, textColor=INDIGO, spaceBefore=6, spaceAfter=10)
H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=12, leading=15, textColor=SLATE, spaceBefore=8, spaceAfter=5)
CAP = ParagraphStyle("cap", parent=BODY, fontSize=8.5, leading=11, alignment=TA_CENTER, textColor=colors.HexColor("#64748B"), spaceAfter=10)
BUL = ParagraphStyle("bul", parent=BODY, leftIndent=14, bulletIndent=3, spaceAfter=3, alignment=0)
CODE = ParagraphStyle("code", parent=BODY, fontName="Courier", fontSize=8.2, leading=11, backColor=colors.HexColor("#F1F5F9"), borderPadding=6, alignment=0, spaceAfter=10)
WARN = ParagraphStyle("warn", parent=BODY, backColor=colors.HexColor("#FEF3C7"), borderColor=AMBER, borderWidth=1, borderPadding=8, textColor=colors.HexColor("#78350F"), spaceBefore=6, spaceAfter=12, alignment=0)


class Doc(BaseDocTemplate):
    def __init__(self, fn, **kw):
        super().__init__(fn, pagesize=A4, leftMargin=2.2 * cm, rightMargin=2.2 * cm, topMargin=2.2 * cm, bottomMargin=2.0 * cm,
                         title="Speech Emotion Recognition using MFCC and Deep Learning", author=args.name, **kw)
        fr = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="f")
        self.addPageTemplates([PageTemplate("cover", [fr], onPage=self.cover), PageTemplate("body", [fr], onPage=self.chrome)])

    def cover(self, c, d):
        c.saveState(); w, h = A4
        c.setFillColor(INDIGO); c.rect(0, h - 9.5 * cm, w, 9.5 * cm, fill=1, stroke=0)
        c.setFillColor(AMBER); c.rect(0, h - 9.5 * cm, w, 0.25 * cm, fill=1, stroke=0)
        c.restoreState()

    def chrome(self, c, d):
        c.saveState(); w, h = A4
        c.setStrokeColor(INDIGO); c.setLineWidth(1.2); c.line(2.2 * cm, h - 1.6 * cm, w - 2.2 * cm, h - 1.6 * cm)
        c.setFont("Helvetica", 8); c.setFillColor(colors.HexColor("#64748B"))
        c.drawString(2.2 * cm, h - 1.35 * cm, "Speech Emotion Recognition - MFCC + CNN-BiLSTM-Attention")
        c.drawRightString(w - 2.2 * cm, 1.2 * cm, f"Page {d.page}")
        if SYN: c.setFillColor(colors.HexColor("#B45309")); c.drawString(2.2 * cm, 1.2 * cm, "RESULTS FROM SYNTHETIC SMOKE-TEST DATA - NOT RAVDESS")
        c.restoreState()

    def afterFlowable(self, f):
        if isinstance(f, Paragraph) and f.style.name in ("h1", "h2"):
            key = f"h-{id(f)}"; self.canv.bookmarkPage(key)
            self.notify("TOCEntry", (0 if f.style.name == "h1" else 1, f.getPlainText(), self.page, key))


def P(t, s=BODY): return Paragraph(t, s)
def bullets(items): return [Paragraph(i, BUL, bulletText="\u2022") for i in items]
def fig(name, width_cm, caption):
    from PIL import Image as PI
    w, h = PI.open(RES / name).size
    return KeepTogether([Image(str(RES / name), width=width_cm * cm, height=width_cm * cm * h / w), P(caption, CAP)])
def table(data, widths, hl_rows=(), left=False):
    t = Table(data, colWidths=[x * cm for x in widths], repeatRows=1)
    st = [("BACKGROUND", (0, 0), (-1, 0), INDIGO), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
          ("FONTSIZE", (0, 0), (-1, -1), 8.8), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]), ("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#CBD5E1")),
          ("ALIGN", (1, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]
    if left: st.append(("ALIGN", (1, 0), (-1, -1), "LEFT"))
    for r in hl_rows: st += [("FONTNAME", (0, r), (-1, r), "Helvetica-Bold"), ("BACKGROUND", (0, r), (-1, r), colors.HexColor("#E0E7FF"))]
    t.setStyle(TableStyle(st)); return t

f4 = lambda x: f"{x:.4f}"
S = []

# ------------------------------------------------------------------ cover
S += [Spacer(1, 1.3 * cm), P("<font color='white' size=11>INDIVIDUAL MAJOR PROJECT  |  MACHINE LEARNING INTERNSHIP</font>", ParagraphStyle("c0", parent=BODY, alignment=0)),
      Spacer(1, .5 * cm),
      P("<font color='white' size=27><b>Speech Emotion Recognition</b></font>", ParagraphStyle("c1", parent=BODY, leading=33, alignment=0)),
      P("<font color='white' size=15>using MFCC Features and a CNN-BiLSTM-Attention Deep Learning Model</font>", ParagraphStyle("c2", parent=BODY, leading=20, alignment=0)),
      Spacer(1, 7.2 * cm),
      table([["Submitted by", args.name + (f"  ({args.roll})" if args.roll else "")], ["Project", "Emotion recognition in speech - MFCC + deep learning, evaluated with F1-score"],
             ["Dataset", M["dataset"]], ["Framework", "Python, librosa, TensorFlow / Keras, scikit-learn"], ["Date", date.today().strftime("%d %B %Y")]], [3.6, 12.8], left=True),
      Spacer(1, .6 * cm)]
if SYN:
    S.append(P("<b>Note on results.</b> The figures in this copy were produced by the built-in <i>synthetic smoke-test</i> dataset, which only proves that the pipeline runs end to end. "
               "They are <b>not</b> RAVDESS results. Run <font name='Courier'>python run_pipeline.py --download</font> and then <font name='Courier'>python build_report.py</font> to regenerate this report with real numbers.", WARN))
S += [PageBreak()]
S.append(P("Table of Contents", H1))
toc = TableOfContents(); toc.levelStyles = [ParagraphStyle("t1", fontName="Helvetica-Bold", fontSize=10.5, leading=17, leftIndent=0), ParagraphStyle("t2", fontName="Helvetica", fontSize=9.5, leading=14, leftIndent=16)]
S += [toc, PageBreak()]

# ------------------------------------------------------------------ abstract
S += [P("Abstract", H1),
      P("Speech carries emotional information in addition to words. This project builds an end-to-end machine learning system that recognises eight emotions "
        "(neutral, calm, happy, sad, angry, fearful, disgust, surprised) from short speech clips. Audio is resampled, trimmed and normalised, then converted into "
        "<b>Mel-Frequency Cepstral Coefficients (MFCC)</b> with first- and second-order deltas. A <b>CNN-BiLSTM-Attention</b> network learns local spectral patterns, "
        "long-range temporal context and the most informative frames. The evaluation protocol is <b>speaker-independent</b>: the actors used for testing never appear in training. "
        "Performance is measured with <b>F1-score</b> (macro, weighted, micro and per class) and compared with a classical RBF-SVM baseline."),
      P(f"On the held-out test speakers of the <i>{M['dataset']}</i> data the deep model reaches a <b>macro-F1 of {f4(deep['macro_f1'])}</b> (accuracy {f4(deep['accuracy'])}), "
        f"against {f4(svm['macro_f1'])} for the SVM baseline." + (" These values come from synthetic smoke-test audio and are shown only to demonstrate the pipeline." if SYN else ""))]
if SYN: S.append(P("<b>Smoke-test copy:</b> metrics in this document are not real-world performance.", WARN))
S.append(P("<b>Keywords:</b> speech emotion recognition, MFCC, CNN, BiLSTM, attention, RAVDESS, F1-score, speaker-independent evaluation.", BODY))
S.append(PageBreak())

# ------------------------------------------------------------------ 1 intro
S += [P("1. Introduction", H1),
      P("Emotion-aware systems - call-centre analytics, tutoring tools, mental-wellbeing assistants and human-robot interaction - need to understand <i>how</i> something is said and not only <i>what</i> is said. "
        "Speech Emotion Recognition (SER) maps an audio signal to an emotion category. It is difficult because emotion is subjective, expressed differently by each speaker, "
        "and public datasets are small, so models easily memorise voices instead of learning emotion."),
      P("1.1 Problem statement", H2),
      P("<i>Develop a machine learning system for emotion recognition in speech using MFCC features and a deep learning model. The project should include data collection, preprocessing, "
        "model implementation, and evaluation using F1-score.</i>"),
      P("1.2 Objectives and how each is met", H2),
      table([["Requirement", "Where it is delivered"],
             ["Data collection", "ser/data.py - automatic RAVDESS download, filename-based labelling, metadata table (Section 4)"],
             ["Preprocessing", "ser/features.py - resampling, silence trimming, normalisation, fixed-length windows, augmentation (Section 5)"],
             ["MFCC features", "40 MFCC + delta + delta-delta = 120 features x 130 frames per clip (Section 5.2)"],
             ["Deep learning model", "ser/model.py - CNN + BiLSTM + attention pooling (Section 6)"],
             ["Evaluation with F1-score", "ser/evaluate.py - macro / weighted / micro / per-class F1, confusion matrices (Sections 7-8)"],
             ["Deliverables", "Source code, this PDF report, PowerPoint presentation"]], [4.2, 12.2], left=True),
      Spacer(1, .3 * cm), P("1.3 Contributions", H2)]
S += bullets(["A fully reproducible pipeline (single command) with fixed seeds, configuration dataclasses and unit tests.",
              "Speaker-independent train/validation/test split that prevents identity leakage, the most common flaw in SER studies.",
              "A compact CNN-BiLSTM-Attention architecture compared against a tuned classical baseline.",
              "Complete F1-based analysis: per-class scores, confusion matrices and error discussion."])

# ------------------------------------------------------------------ 2 background
S += [P("2. Background", H1),
      P("2.1 MFCC", H2),
      P("MFCCs (Davis and Mermelstein, 1980) summarise the short-time spectral envelope on the perceptual mel scale. The signal is split into overlapping frames, "
        "a power spectrum is computed, passed through a mel filter bank, log-compressed and decorrelated with a discrete cosine transform. The first coefficients describe the "
        "vocal-tract shape, which changes with arousal and emotional state. Delta and delta-delta features add the velocity and acceleration of each coefficient and so capture dynamics."),
      P("2.2 Deep sequence models", H2),
      P("1-D convolutions detect local time-frequency patterns, batch normalisation and dropout stabilise training on small data, a bidirectional LSTM (Hochreiter and Schmidhuber, 1997) models context in both "
        "directions, and additive attention pooling learns a weighted average over frames so that emotionally salient moments (stressed syllables, pauses, pitch peaks) dominate the utterance representation."),
      P("2.3 F1-score", H2),
      P("For class c, precision = TP / (TP + FP), recall = TP / (TP + FN) and F1 = 2PR / (P + R). <b>Macro-F1</b> averages F1 over classes with equal weight (the primary metric here, because it "
        "penalises ignoring minority classes); <b>weighted-F1</b> weights classes by support; <b>micro-F1</b> pools all decisions and equals accuracy in single-label problems.")]

# ------------------------------------------------------------------ 3 method overview
S += [P("3. System Overview", H1), P("The system is a linear pipeline. Every stage is a separate, tested Python module.", BODY),
      fig("diagram_pipeline.png", 16.4, "Figure 1 - End-to-end pipeline."), PageBreak()]

# ------------------------------------------------------------------ 4 data
cc = M["class_counts"]
S += [P("4. Data Collection", H1),
      P("The project uses the <b>Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)</b> (Livingstone and Russo, 2018), freely available for research under CC BY-NC-SA 4.0. "
        "The speech subset contains 1,440 recordings: 24 professional actors (12 female, 12 male) each speaking two lexically matched sentences in eight emotions at two intensities. "
        "The emotion, intensity, statement, repetition and actor are encoded in each filename (e.g. <font name='Courier'>03-01-05-02-01-01-07.wav</font> = speech, angry, strong, actor 7)."),
      P("<b>Collection:</b> <font name='Courier'>python run_pipeline.py --download</font> fetches and unzips the archive from Zenodo; labels are parsed from the filename and stored in a pandas metadata table."),
      table([["Item", "Value"], ["Dataset used in this run", M["dataset"]], ["Clips", str(M["n_clips"])], ["Actors", str(M["n_actors"])], ["Classes", "8"],
             ["Train clips / actors", f"{M['split']['train_clips']} / {len(M['split']['train_actors'])}"], ["Validation clips / actors", f"{M['split']['val_clips']} / {len(M['split']['val_actors'])}"],
             ["Test clips / actors", f"{M['split']['test_clips']} / {len(M['split']['test_actors'])}"]], [7, 9.4]),
      Spacer(1, .3 * cm), fig("class_distribution.png", 12, "Figure 2 - Number of clips per emotion."),
      P("<b>Speaker-independent split.</b> Actors are assigned to exactly one partition (train / validation / test) and an assertion in the code guards against overlap. "
        "A random clip-level split would put the same voice in training and test sets and inflate scores, because the model could recognise the speaker rather than the emotion.")]

# ------------------------------------------------------------------ 5 preprocessing
S += [P("5. Pre-processing and Feature Extraction", H1), P("5.1 Pre-processing steps", H2)]
S += bullets(["<b>Resampling</b> to 22,050 Hz mono (RAVDESS is 48 kHz) - reduces cost, retains speech-relevant bandwidth.",
              "<b>Silence trimming</b> with a 30 dB threshold, then <b>peak normalisation</b> to remove loudness differences between recordings.",
              "<b>Fixed 3-second window</b>: centre crop or zero-pad so every sample yields the same tensor shape.",
              "<b>Data augmentation</b> (training clips only, "
              f"{M['training'].get('aug_copies', 2) if 'aug_copies' in M['training'] else 2} extra copies each): Gaussian noise, pitch shift (+/-2 semitones), time stretch (0.9-1.1x), time shift. "
              f"Training set grows to {M['split']['train_after_aug']} samples. Validation and test data are never augmented.",
              "<b>Standardisation</b>: per-coefficient mean and standard deviation fitted on the training set only and re-used for validation, test and inference."])
S += [P("5.2 MFCC configuration", H2),
      table([["Parameter", "Value"], ["Sampling rate", f"{M['features']['sr']} Hz"], ["MFCC coefficients", str(M['features']['n_mfcc'])], ["FFT window / hop", f"2048 / {M['features']['hop']} samples"],
             ["Mel bands", "128"], ["Features per frame", "120 (MFCC + delta + delta-delta)"], ["Input tensor per clip", f"{M['features']['shape'][0]} frames x {M['features']['shape'][1]} features"]], [7, 9.4]),
      Spacer(1, .3 * cm), fig("mfcc_example.png", 13, "Figure 3 - MFCC matrix of a test clip (first 40 coefficients shown).")]

# ------------------------------------------------------------------ 6 model
S += [PageBreak(), P("6. Model Implementation", H1),
      P("The model (<font name='Courier'>ser/model.py</font>, Keras 3) processes the MFCC sequence in three stages:"),
      fig("diagram_architecture.png", 16.4, "Figure 4 - CNN-BiLSTM-Attention architecture.")]
S += bullets(["<b>Convolutional front-end</b> - three Conv1D blocks (64, 128, 128 filters) with batch norm, ReLU, max-pooling and dropout reduce 130 frames to 16 while learning local spectral patterns.",
              "<b>Bidirectional LSTM (96 units)</b> - models how the pattern evolves forward and backward in time.",
              "<b>Additive attention pooling</b> - a small tanh network scores each frame, softmax normalises the scores, and the weighted sum becomes the utterance embedding.",
              "<b>Classifier</b> - Dense(128) + dropout 0.4 + softmax over eight emotions."])
S += [P(f"The network has <b>{tr['params']:,}</b> trainable parameters. L2 weight decay (1e-4), dropout, Gaussian input noise and early stopping combat over-fitting on the small corpus."),
      P("6.1 Baseline", H2),
      P(f"An RBF-SVM is trained on utterance-level statistics (mean and standard deviation of every feature over time, 240 dimensions) with standardisation and balanced class weights. "
        f"C and gamma are tuned on the validation actors (selected: C = {svm['params']['C']}, gamma = {svm['params']['gamma']}). It shows how much the sequence model adds."),
      P("6.2 Training setup", H2),
      table([["Setting", "Value"], ["Optimiser", "Adam, lr 1e-3, ReduceLROnPlateau (factor 0.5)"], ["Loss", "Sparse categorical cross-entropy + balanced class weights"],
             ["Batch size", str(tr['batch_size'])], ["Max epochs / run", f"{tr['epochs_max']} / {tr['epochs_run']} (early stopping on val loss)"],
             ["Random seed", str(tr['seed'])], ["Training time", f"{tr['seconds']:.0f} s (CPU)"]], [7, 9.4])]

# ------------------------------------------------------------------ 7 evaluation
S += [PageBreak(), P("7. Evaluation Methodology", H1),
      P("All reported numbers come from the <b>test actors only</b>, which are not used for training, hyper-parameter tuning or early stopping. The primary metric is <b>macro-F1</b>; "
        "weighted-F1, micro-F1 and accuracy are reported for completeness together with per-class precision, recall and F1, and row-normalised confusion matrices."),
      P("8. Results", H1)]
if SYN: S.append(P("<b>The numbers below come from the synthetic smoke-test dataset.</b> They validate the code path only. Replace them by running the real RAVDESS experiment.", WARN))
S += [P("8.1 Overall F1-scores", H2),
      table([["Model", "Macro-F1", "Weighted-F1", "Micro-F1", "Accuracy"],
             ["SVM baseline (MFCC statistics)", f4(svm["macro_f1"]), f4(svm["weighted_f1"]), f4(svm["micro_f1"]), f4(svm["accuracy"])],
             ["CNN-BiLSTM-Attention (proposed)", f4(deep["macro_f1"]), f4(deep["weighted_f1"]), f4(deep["micro_f1"]), f4(deep["accuracy"])]], [6.2, 2.5, 2.7, 2.5, 2.5], hl_rows=(2,)),
      Spacer(1, .3 * cm),
      P(f"The deep model changes macro-F1 by <b>{deep['macro_f1'] - svm['macro_f1']:+.4f}</b> relative to the baseline. Validation macro-F1 of the selected checkpoint was {f4(M['deep_val']['macro_f1'])}."),
      P("8.2 Per-class results (proposed model)", H2)]
rows = [["Emotion", "Precision", "Recall", "F1", "Support"]] + [[k, f4(v["precision"]), f4(v["recall"]), f4(v["f1"]), str(v["support"])] for k, v in deep["per_class"].items()]
S += [table(rows, [4, 3, 3, 3, 3]), Spacer(1, .4 * cm), fig("f1_per_class.png", 14, "Figure 5 - Per-class F1: baseline vs proposed model."),
      fig("training_curves.png", 15, "Figure 6 - Training and validation loss / accuracy."),
      fig("confusion_matrix.png", 12.5, "Figure 7 - Row-normalised confusion matrix of the proposed model on test speakers."),
      fig("confusion_matrix_svm.png", 11, "Figure 8 - Confusion matrix of the SVM baseline.")]
best = max(deep["per_class"].items(), key=lambda kv: kv[1]["f1"]); worst = min(deep["per_class"].items(), key=lambda kv: kv[1]["f1"])
S += [P("9. Discussion", H1),
      P(f"The best-recognised class was <b>{best[0]}</b> (F1 {f4(best[1]['f1'])}) and the weakest was <b>{worst[0]}</b> (F1 {f4(worst[1]['f1'])}). "
        "Inspect the confusion matrix for systematic confusions: on real acted speech, emotions with similar arousal (for example calm and neutral, or fearful and surprised) are commonly confused, because MFCCs mainly encode spectral envelope and arousal rather than valence."),
      P("9.1 Limitations", H2)]
S += bullets(["RAVDESS is <i>acted</i> speech from 24 speakers with two sentences; spontaneous, noisy or multilingual speech will behave differently.",
              "With four test actors the F1 estimate has noticeable variance; repeated runs with different actor splits (leave-speakers-out cross-validation) give a more reliable estimate.",
              "Only MFCC features are used, as required; prosodic features (pitch, energy) or self-supervised embeddings could add valence information.",
              "No demographic fairness analysis was performed."])
S += [P("9.2 Future work", H2)]
S += bullets(["Speaker-grouped k-fold cross-validation and confidence intervals.", "SpecAugment-style masking and mixup; additional corpora (TESS, CREMA-D, EMO-DB) for cross-corpus evaluation.",
              "Transformer / wav2vec-style encoders fine-tuned with the same evaluation protocol.", "Real-time microphone demo and a REST API."])
S += [P("10. Conclusion", H1),
      P("The project delivers every element of the problem statement: data collection from RAVDESS, a documented pre-processing and MFCC feature pipeline, a CNN-BiLSTM-Attention implementation, "
        "and an F1-score-centred evaluation under a speaker-independent protocol, together with a classical baseline, unit tests and a one-command reproducible workflow."),
      P("11. How to Reproduce", H1),
      P("pip install -r requirements.txt<br/>python run_pipeline.py --download --data data/RAVDESS<br/>python build_report.py --name \"Your Name\"<br/>python build_ppt.py --name \"Your Name\"<br/>python predict.py path/to/clip.wav<br/>pytest -q", CODE),
      P("Project structure", H2),
      P("ser/config.py - settings | ser/data.py - download, metadata, split | ser/features.py - preprocessing, MFCC, augmentation | ser/model.py - network | "
        "ser/train.py - SVM and deep training | ser/evaluate.py - F1 metrics and plots | run_pipeline.py - orchestration | predict.py - inference | tests/ - unit tests", BODY),
      P("References", H1)]
refs = ["Livingstone, S. R. and Russo, F. A. (2018). The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS). PLoS ONE 13(5): e0196391.",
        "Davis, S. and Mermelstein, P. (1980). Comparison of parametric representations for monosyllabic word recognition in continuously spoken sentences. IEEE Trans. ASSP 28(4), 357-366.",
        "Hochreiter, S. and Schmidhuber, J. (1997). Long short-term memory. Neural Computation 9(8), 1735-1780.",
        "Kingma, D. P. and Ba, J. (2015). Adam: A method for stochastic optimization. ICLR.",
        "Ioffe, S. and Szegedy, C. (2015). Batch normalization: accelerating deep network training by reducing internal covariate shift. ICML.",
        "Srivastava, N. et al. (2014). Dropout: a simple way to prevent neural networks from overfitting. JMLR 15, 1929-1958.",
        "Bahdanau, D., Cho, K. and Bengio, Y. (2015). Neural machine translation by jointly learning to align and translate. ICLR.",
        "McFee, B. et al. (2015). librosa: Audio and music signal analysis in Python. Proc. SciPy.",
        "Pedregosa, F. et al. (2011). Scikit-learn: machine learning in Python. JMLR 12, 2825-2830."]
S += [Paragraph(f"[{i}] {r}", ParagraphStyle("ref", parent=BODY, fontSize=9, leading=12.5, leftIndent=18, firstLineIndent=-18, alignment=0)) for i, r in enumerate(refs, 1)]

# cover uses template 0; switch to body template after the cover page
from reportlab.platypus import NextPageTemplate
S.insert(0, NextPageTemplate("body"))
Doc(args.out).multiBuild(S)
print("wrote", args.out)
