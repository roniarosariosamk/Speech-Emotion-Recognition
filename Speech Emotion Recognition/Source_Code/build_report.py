"""Generate the project report PDF from the REAL final results (results/final/final_summary.json).

    python summarize_results.py                      # first (verifies metrics, selects model, makes figures)
    python build_report.py --name "Your Name" --roll "ID"
Every number in the report is read from the saved metrics; nothing is hard-coded.
"""
import argparse
import json
from datetime import date
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether, NextPageTemplate, PageBreak, PageTemplate,
                                Paragraph, Spacer, Table, TableStyle)
from reportlab.platypus.tableofcontents import TableOfContents

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results" / "final"
INDIGO, SLATE, LIGHT, AMBER = colors.HexColor("#4F46E5"), colors.HexColor("#1E293B"), colors.HexColor("#EEF2FF"), colors.HexColor("#F59E0B")

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="Ronia"); ap.add_argument("--roll", default="")
ap.add_argument("--out", default=str(ROOT.parent / "Speech_Emotion_Recognition_Report.pdf"))
args = ap.parse_args()
if not (RES / "final_summary.json").is_file():
    raise SystemExit("results/final/final_summary.json not found - run:  python summarize_results.py")
S_ = json.loads((RES / "final_summary.json").read_text())
SYN = False
SEL = S_["selected_final_model"]; T = SEL["test"]; CFGM = SEL["config"]; SVM = S_["svm_baseline"]; DS = S_["dataset"]
EXPS = {r["experiment"]: r for r in S_["experiments"]}
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
    _cs = ParagraphStyle("cell0", parent=BODY, fontSize=8.8, leading=11, alignment=0, spaceAfter=0)
    data = [[Paragraph(c, _cs) if (isinstance(c, str) and len(c) > 42 and r > 0) else c for c in row] for r, row in enumerate(data)]
    t = Table(data, colWidths=[x * cm for x in widths], repeatRows=1)
    st = [("BACKGROUND", (0, 0), (-1, 0), INDIGO), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
          ("FONTSIZE", (0, 0), (-1, -1), 8.8), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]), ("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#CBD5E1")),
          ("ALIGN", (1, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]
    if left: st.append(("ALIGN", (1, 0), (-1, -1), "LEFT"))
    for r in hl_rows: st += [("FONTNAME", (0, r), (-1, r), "Helvetica-Bold"), ("BACKGROUND", (0, r), (-1, r), colors.HexColor("#E0E7FF"))]
    t.setStyle(TableStyle(st)); return t

f4 = lambda x: f"{x:.4f}"
import numpy as np

f4 = lambda x: f"{x:.4f}"
pct = lambda x: f"{x * 100:.2f}%"
pc = T["per_class"]; cmat = np.array(T["confusion_matrix"]); npred = cmat.sum(0)
LAB = list(pc)
best = max(pc.items(), key=lambda kv: kv[1]["f1"]); worst = min(pc.items(), key=lambda kv: kv[1]["f1"])
low_rec = sorted(pc.items(), key=lambda kv: kv[1]["recall"])[:3]
tc = SEL["top_confusions"]
done = [r for r in S_["experiments"] if r["status"] == "completed"]
E4 = EXPS["experiment_04"]["status"] == "completed"
N_EXP = len(done)
sp = DS["split"]
S = []
P = lambda t, s=BODY: Paragraph(t, s)
TOCH = ParagraphStyle("t0", parent=BODY)


def bullets(items): return [Paragraph(i, BUL, bulletText="\u2022") for i in items]


def plain(t): return P("<b>In plain words:</b> " + t, WARN2)


WARN2 = ParagraphStyle("plain", parent=BODY, backColor=colors.HexColor("#ECFEFF"), borderColor=colors.HexColor("#06B6D4"), borderWidth=1,
                       borderPadding=7, spaceBefore=4, spaceAfter=12, alignment=0, textColor=SLATE)

# ------------------------------------------------------------------ cover
S += [Spacer(1, 1.3 * cm), P("<font color='white' size=11>INDIVIDUAL MAJOR PROJECT  |  MACHINE LEARNING INTERNSHIP</font>", ParagraphStyle("c0", parent=BODY, alignment=0)), Spacer(1, .5 * cm),
      P("<font color='white' size=26><b>Machine Learning System for Emotion Recognition in Speech</b></font>", ParagraphStyle("c1", parent=BODY, leading=32, alignment=0)),
      P("<font color='white' size=15>Using MFCC Features and a CNN-BiLSTM-Attention Deep Learning Model</font>", ParagraphStyle("c2", parent=BODY, leading=20, alignment=0)),
      Spacer(1, 6.4 * cm),
      table([["Submitted by", args.name + (f"  ({args.roll})" if args.roll else "")],
             ["Task", "Data collection, preprocessing, model implementation and F1-score evaluation"],
             ["Dataset", f"{DS['name']} - {DS['clips']} clips, {DS['actors']} actors, 8 emotions"],
             ["Final model", f"{SEL['label']}: {SEL['title']} ({SEL['parameters']:,} parameters)"],
             ["Headline result", f"Test Macro F1 {f4(T['macro_f1'])} | accuracy {pct(T['accuracy'])} (SVM baseline Macro F1 {f4(SVM['test_macro_f1'])})"],
             ["Framework", "Python, librosa, TensorFlow / Keras, scikit-learn"], ["Date", date.today().strftime("%d %B %Y")]], [3.6, 12.8], left=True), Spacer(1, .4 * cm),
      NextPageTemplate("body"), PageBreak()]
S.append(P("Table of Contents", H1))
toc = TableOfContents(); toc.levelStyles = [ParagraphStyle("t1", fontName="Helvetica-Bold", fontSize=10, leading=13.5, leftIndent=0), ParagraphStyle("t2", fontName="Helvetica", fontSize=9, leading=11.5, leftIndent=16)]
S += [toc, PageBreak()]

# ------------------------------------------------------------------ abstract
S += [P("Abstract", H1),
      P("Speech carries emotional information in addition to words. This project builds and evaluates a machine learning system that recognises eight emotions (neutral, calm, happy, sad, angry, fearful, disgust, surprised) "
        "from short speech clips of the RAVDESS corpus. Each clip is resampled, trimmed, normalised and converted into <b>Mel-Frequency Cepstral Coefficients (MFCC)</b> with their first- and second-order derivatives "
        "(130 frames x 120 features). A compact <b>CNN-BiLSTM-Attention</b> network (" + f"{SEL['parameters']:,}" + " parameters) learns local spectral patterns, temporal context and the most informative frames. "
        "The evaluation is <b>actor-independent</b>: actors 1-16 are used for training, 17-20 for validation and 21-24 for testing, so no voice appears in more than one partition."),
      P(f"On the held-out test actors the selected model ({SEL['label']}) obtains a <b>Macro F1 of {f4(T['macro_f1'])}</b>, weighted F1 {f4(T['weighted_f1'])} and accuracy {pct(T['accuracy'])}, compared with a Macro F1 of "
        f"{f4(SVM['test_macro_f1'])} for a tuned RBF-SVM baseline ({SEL['gain_over_svm_macro_f1']:+.4f}). Performance is uneven across emotions: recall is high for angry and surprised speech but low for sad and fearful speech. "
        "The results are obtained on acted speech from 24 actors and should not be read as a claim that the system reliably recognises emotion in everyday conversation."),
      P("<b>Keywords:</b> speech emotion recognition, MFCC, CNN, BiLSTM, attention, RAVDESS, Macro F1, actor-independent evaluation.", BODY), PageBreak()]

# ------------------------------------------------------------------ 1 introduction
S += [P("1. Introduction", H1),
      P("Emotion-aware applications such as call-centre analytics, tutoring software, wellbeing assistants and human-robot interaction need to understand <i>how</i> something is said, not only <i>what</i> is said. "
        "Speech Emotion Recognition (SER) maps an audio signal to an emotion category. It is hard because emotion is subjective, every speaker expresses it differently, and public datasets are small, so models easily learn "
        "to recognise <i>voices</i> instead of <i>emotions</i>."),
      P("1.1 Problem statement", H2),
      P("<i>Develop a machine learning system for emotion recognition in speech using MFCC features and a deep learning model. The project should include data collection, preprocessing, model implementation, and evaluation using F1-score.</i>"),
      P("1.2 Objectives and where each is delivered", H2),
      table([["Objective", "Where it is delivered"],
             ["Data collection", "ser/data.py - RAVDESS download, filename-based labelling, metadata table (Section 3)"],
             ["Preprocessing", "ser/features.py - resampling, silence trimming, normalisation, fixed 3 s window, augmentation (Section 4)"],
             ["MFCC features", "40 MFCC + delta + delta-delta = 120 features x 130 frames per clip (Section 5)"],
             ["Deep learning model", "ser/model.py - CNN + BiLSTM + temporal attention (Section 6)"],
             ["Classical baseline", "ser/train.py - RBF-SVM on utterance-level MFCC statistics (Section 7)"],
             ["F1-score evaluation", "ser/evaluate.py, ser/final.py - macro / weighted / per-class F1, confusion matrices (Sections 8-10)"],
             ["Deliverables", "Source code, tests, this report, presentation, README, results summary"]], [4.2, 12.2], left=True),
      Spacer(1, .3 * cm), P("1.3 Contributions", H2)]
S += bullets(["A reproducible, tested pipeline with a fixed random seed, central configuration and automatic verification of every reported metric.",
              "An actor-independent split that prevents speaker leakage, a frequent flaw in SER studies.",
              "A compact CNN-BiLSTM-Attention model compared against a tuned classical baseline over several documented experiments.",
              "An honest error analysis that separates observed errors from plausible, untested explanations."])

# ------------------------------------------------------------------ 2 background
S += [P("2. Background on Speech Emotion Recognition", H1),
      P("2.1 MFCC", H2),
      P("MFCCs (Davis and Mermelstein, 1980) describe the short-time spectral envelope of speech on the perceptual mel scale. The signal is cut into overlapping frames, each frame is converted to a power spectrum, "
        "passed through a mel filter bank, log-compressed and decorrelated with a discrete cosine transform. The low-order coefficients mainly describe the shape of the vocal tract, which changes with arousal and speaking style."),
      plain("MFCCs turn raw sound into a small table of numbers that describes the 'timbre' of the voice at every instant, similar to how the ear perceives it."),
      P("2.2 Delta and delta-delta features", H2),
      P("The delta (first derivative) of each coefficient measures how fast it changes between frames; the delta-delta (second derivative) measures how that change accelerates. Emotion is expressed through dynamics - "
        "an angry voice changes abruptly, a sad voice slowly - so adding both derivatives gives the network information about movement, not only about static timbre."),
      P("2.3 CNN, BiLSTM and attention", H2),
      P("A <b>1-D convolutional layer</b> slides small filters along time and detects short local patterns in the MFCC table. A <b>bidirectional LSTM</b> (Hochreiter and Schmidhuber, 1997) reads the sequence forwards and backwards "
        "and remembers context, for example a rising pitch followed by a pause. <b>Attention pooling</b> (Bahdanau et al., 2015, adapted here to a single utterance-level summary) lets the network learn which frames matter most and take a weighted average of them."),
      plain("The CNN spots short sound patterns, the BiLSTM understands how they follow each other, and attention decides which moments of the clip deserve the most weight in the final decision."),
      P("2.4 Macro F1-score", H2),
      P("For each emotion, precision = TP / (TP + FP) (how often a predicted emotion is right), recall = TP / (TP + FN) (how many real examples are found) and F1 = 2PR / (P + R). "
        "<b>Macro F1</b> averages the F1 of all classes with equal weight, so a model cannot hide poor performance on one emotion behind good performance on others; this is why it is the primary metric here. "
        "<b>Weighted F1</b> weights classes by their number of test clips."),
      plain("Accuracy only counts how many answers are right overall. Macro F1 asks: is the system good at every emotion, not just the easy ones?")]

# ------------------------------------------------------------------ 3 dataset
cc = DS["class_counts"]
S += [P("3. Dataset", H1),
      P("The project uses the <b>Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)</b> (Livingstone and Russo, 2018), available for research under CC BY-NC-SA 4.0. The speech subset has 1,440 recordings: "
        "24 professional actors (12 female, 12 male) speak two matched sentences in eight emotions at two intensities (neutral has a single intensity, hence 96 neutral clips and 192 of every other emotion). "
        "Emotion, intensity, statement, repetition and actor are encoded in the filename, for example <font name='Courier'>03-01-05-02-01-01-07.wav</font> = speech, angry, strong intensity, actor 7. "
        "<b>Collection:</b> <font name='Courier'>python run_pipeline.py --download</font> downloads and unzips the archive; labels are parsed from filenames into a pandas metadata table."),
      table([["Item", "Value"], ["Dataset", DS["name"]], ["Clips / actors / classes", f"{DS['clips']} / {DS['actors']} / 8"],
             ["Training", f"{sp['train_clips']} clips, actors {sp['train_actors'][0]}-{sp['train_actors'][-1]} ({sp['train_after_aug']} samples after augmentation)"],
             ["Validation", f"{sp['val_clips']} clips, actors {sp['val_actors'][0]}-{sp['val_actors'][-1]} (not augmented)"],
             ["Test", f"{sp['test_clips']} clips, actors {sp['test_actors'][0]}-{sp['test_actors'][-1]} (not augmented)"],
             ["Class counts", ", ".join(f"{k} {v}" for k, v in cc.items())], ["Random seed", str(DS["seed"])]], [4.6, 11.8], left=True),
      Spacer(1, .3 * cm), fig("class_distribution.png", 11.5, "Figure 1 - Number of clips per emotion (neutral has half as many clips).")]

# ------------------------------------------------------------------ 4 preprocessing
S += [P("4. Data Preprocessing", H1), fig("diagram_pipeline.png", 16.4, "Figure 2 - End-to-end pipeline."), P("Every clip passes through the same steps:")]
S += bullets(["<b>Resampling</b> to 22,050 Hz mono (RAVDESS is recorded at 48 kHz).",
              "<b>Silence trimming</b> (30 dB threshold) and <b>peak normalisation</b> remove leading/trailing silence and loudness differences.",
              "<b>Fixed 3-second window</b>: centre crop or zero-padding, so every clip yields a tensor of the same shape.",
              f"<b>Augmentation (training clips only)</b>: two extra copies per clip, each with one random transformation - Gaussian noise, pitch shift (+/-2 semitones), time stretch (0.9-1.1x) or time shift - giving {sp['train_after_aug']:,} training samples. Validation and test clips are never augmented.",
              "<b>Standardisation</b>: per-feature mean and standard deviation are computed on the <b>training set only</b>, saved to <font name='Courier'>norm.npz</font>, and re-used unchanged for validation, test and inference.",
              "<b>Input validation</b>: unreadable, empty or silent audio raises a clear error instead of producing a silent wrong prediction."])

# ------------------------------------------------------------------ 5 MFCC
ft = DS["features"]
S += [P("5. MFCC Feature Extraction", H1),
      table([["Parameter", "Value"], ["Sampling rate", f"{ft['sr']} Hz"], ["Clip length", f"{ft['duration']} s"], ["MFCC coefficients", str(ft["n_mfcc"])], ["FFT size / hop length", f"{ft['n_fft']} / {ft['hop']} samples"],
             ["Mel bands", str(ft["n_mels"])], ["Features per frame", "120 = 40 MFCC + 40 delta + 40 delta-delta"], ["Tensor per clip", f"{ft['shape'][0]} frames x {ft['shape'][1]} features"]], [6, 10.4]),
      Spacer(1, .3 * cm), fig("mfcc_example.png", 13, "Figure 3 - MFCC matrix of a test clip (first 40 coefficients shown)."),
      P("The tensor shape (130, 120) was confirmed from the saved experiment metadata and by the unit tests, which extract features from audio and check the shape and finiteness.")]

# ------------------------------------------------------------------ 6 architecture
cf = CFGM
S += [PageBreak(), P("6. Proposed CNN-BiLSTM-Attention Architecture", H1),
      P(f"The final model is the compact network of <b>{SEL['label']}</b> ({SEL['parameters']:,} trainable parameters). It processes the MFCC sequence in four stages:"),
      fig("diagram_architecture.png", 16.4, "Figure 4 - Architecture of the selected model (values read from the trained checkpoint).")]
S += bullets([f"<b>Gaussian input noise</b> (sigma {cf['gaussian_noise']}) acts as a regulariser during training only.",
              f"<b>Two Conv1D blocks</b> ({cf['conv1']['filters']} filters, kernel {cf['conv1']['kernel']}; {cf['conv2']['filters']} filters, kernel {cf['conv2']['kernel']}), each with batch normalisation, ReLU, max-pooling and dropout ({cf['conv1']['dropout']}, {cf['conv2']['dropout']}); 130 frames are reduced to 32.",
              f"<b>Bidirectional LSTM</b> ({cf['lstm']['units']} units per direction, dropout {cf['lstm']['dropout']}) returns one vector per time step.",
              f"<b>Additive attention pooling</b> ({cf['attention_units']} units): a small tanh network scores every frame, a softmax turns the scores into weights, and the weighted sum is the utterance embedding.",
              f"<b>Classifier</b>: Dense({cf['dense']['units']}, ReLU, L2 {cf['l2']}) - Dropout {cf['dense']['dropout']} - Dense(8, softmax) giving eight probabilities that sum to one."])
S += [P("Loss is sparse categorical cross-entropy with balanced class weights; the optimiser is Adam. The attention layer is a registered custom Keras layer so that checkpoints save and load reliably "
        "(a Keras notice about its un-built sub-layers appears when loading; the model nevertheless loads, predicts and reproduces the training-time outputs).")]

# ------------------------------------------------------------------ 7 SVM
S += [P("7. SVM Baseline", H1),
      P(f"A classical baseline shows how much the deep model adds. Every clip is summarised by the mean and standard deviation of each of the 120 features over time (240 numbers). A standardiser (fitted on training data only) and an "
        f"RBF-kernel SVM with balanced class weights are trained on the same actor-independent training set. The hyper-parameters C in {{1, 10, 100}} and gamma in {{scale, 0.001, 0.01}} are chosen by Macro F1 on the "
        f"<b>validation actors only</b> (selected: C = {SVM['params']['C']}, gamma = {SVM['params']['gamma']}); the final baseline is then evaluated once on the test actors. It uses the same class-label order as the deep model.")]

# ------------------------------------------------------------------ 8 methodology
S += [P("8. Experimental Methodology", H1), P("8.1 Actor-independent data split", H2),
      P("Actors 1-16 form the training set, 17-20 the validation set and 21-24 the test set. The code asserts that no actor appears in two partitions. A random clip-level split would put the same voices in training and test "
        "data and overstate performance, because the model could recognise the speaker instead of the emotion."),
      P("8.2 Training configuration (selected model)", H2),
      table([["Setting", "Value"], ["Optimiser / learning rate", "Adam, 1e-3; halved after 4 epochs without validation-loss improvement (minimum 1e-5)"], ["Loss / class weights", "Sparse categorical cross-entropy / balanced"],
             ["Batch size / maximum epochs", f"32 / 60"], ["Checkpoint selection", f"validation Macro F1 (best epoch {SEL['best_epoch']} of {SEL['epochs_run']} run)"],
             ["Early stopping", "12 epochs without validation Macro F1 improvement"], ["Random seed", str(DS["seed"])], ["Training time", f"{SEL['training_seconds']:.0f} s on CPU"],
             ["Software (project environment)", "Python 3.13, TensorFlow 2.21.0, Keras 3.15.1, librosa 1.0.0, scikit-learn"]], [5.4, 11], left=True),
      P("8.3 Evaluation metrics", H2),
      P("On the test actors the report gives accuracy, macro precision, macro recall, Macro F1, weighted F1, per-class precision/recall/F1 and the confusion matrix, always over all eight labels. "
        "Every figure is recomputed from the saved confusion matrix and checked against the stored value by <font name='Courier'>summarize_results.py</font> and by the unit tests."),
      P("8.4 Experiment log", H2)]
chg = {"experiment_01": "First full model (3 Conv1D blocks, BiLSTM 96, Dense 128); checkpoint by validation loss.",
       "experiment_02": "Compact network (2 Conv1D blocks, BiLSTM 48, Dense 64), stronger regularisation; checkpoint by validation loss.",
       "experiment_03": "Same compact network; checkpoint and early stopping on validation Macro F1; up to 60 epochs.",
       "experiment_04": "Compact network with slightly stronger dropout (0.35 / 0.40 / 0.35 / 0.50)."}
rows = [["Experiment", "Change", "Params", "Val Macro F1", "Test Macro F1"]]
for k in ("experiment_01", "experiment_02", "experiment_03", "experiment_04"):
    r = EXPS[k]
    rows.append([r["label"], Paragraph(chg[k], ParagraphStyle("cell", parent=BODY, fontSize=8.3, leading=10.5, alignment=0)),
                 f"{r['params']:,}" if r["status"] == "completed" else "-", f4(r["val_macro_f1"]) if r["status"] == "completed" else "not run", f4(r["test_macro_f1"]) if r["status"] == "completed" else "not run"])
S += [table(rows, [2.5, 7.4, 1.9, 2.3, 2.3], hl_rows=(3,) if SEL["experiment"] == "experiment_03" else ()),
      Spacer(1, .2 * cm),
      P(("<b>Experiment 4 (completed, single run, seed 42):</b> validation Macro F1 " + f4(EXPS["experiment_04"]["val_macro_f1"]) + ", test Macro F1 " + f4(EXPS["experiment_04"]["test_macro_f1"]) +
         " versus " + f4(EXPS["experiment_03"]["val_macro_f1"]) + " / " + f4(EXPS["experiment_03"]["test_macro_f1"]) + " for Experiment 3. Stronger dropout did not improve generalisation in this run, so Experiment 3 remains the final model.")
        if E4 else f"<b>Experiment 4 status:</b> {S_['experiment_4_status']}", WARN)]

# ------------------------------------------------------------------ 9 results
S += [PageBreak(), P("9. Results and Comparison", H1), P("9.1 Overall comparison", H2)]
rows = [["Model", "Val Macro F1", "Test acc.", "Test Macro F1", "Test weighted F1"]]
for r in done: rows.append([f"{r['label']} - deep model", f4(r["val_macro_f1"]), f4(r["test_accuracy"]), f4(r["test_macro_f1"]), f4(r["test_weighted_f1"])])
rows.append(["SVM baseline (MFCC statistics)", "-", f4(SVM["test_accuracy"]), f4(SVM["test_macro_f1"]), f4(SVM["test_weighted_f1"])])
S += [table(rows, [6.4, 2.5, 2.4, 2.6, 2.7], hl_rows=(1 + [r["experiment"] for r in done].index(SEL["experiment"]),)), Spacer(1, .3 * cm), fig("comparison_macro_f1.png", 12.5, "Figure 5 - Validation and test Macro F1 of every completed experiment and the SVM baseline."),
      P(f"<b>Selected final model: {SEL['label']}.</b> It has the highest validation Macro F1 ({f4(SEL['validation_macro_f1'])}) of the completed experiments, so it ranks first under the validation-only selection rule used for this project; it also has the highest test Macro F1. "
        f"Its test Macro F1 is {SEL['gain_over_svm_macro_f1']:+.4f} higher than the SVM baseline (macro precision {f4(T['macro_precision'])}, macro recall {f4(T['macro_recall'])}). "
        "Because the test set has only 240 clips from 4 actors, differences of a few F1 points between experiments are within plausible sampling noise and should not be over-interpreted. "
        "Note also that Experiments 1-3 were compared while test results were visible, so the test figure is a slightly optimistic estimate; the validation ranking points in the same direction.")]
S += [P("9.2 Classification report (test set, selected model)", H2)]
rows = [["Emotion", "Precision", "Recall", "F1", "Support"]] + [[k, f4(v["precision"]), f4(v["recall"]), f4(v["f1"]), str(v["support"])] for k, v in pc.items()]
rows.append(["macro average", f4(T["macro_precision"]), f4(T["macro_recall"]), f4(T["macro_f1"]), str(sum(v["support"] for v in pc.values()))])
S += [table(rows, [4.2, 3, 3, 3, 3], hl_rows=(len(rows) - 1,)), Spacer(1, .3 * cm), fig("per_class_prf.png", 13.5, "Figure 6 - Per-class precision, recall and F1 on the test actors."),
      fig("training_curves.png", 15, "Figure 7 - Training and validation loss / accuracy of the selected run."),
      KeepTogether([P("9.3 Confusion matrix", H2), fig("final_confusion_matrix.png", 12.5, "Figure 8 - Row-normalised confusion matrix of the selected model (rows: true emotion, columns: predicted).")]),
      fig("confusion_matrix_svm.png", 11, "Figure 9 - Confusion matrix of the SVM baseline for comparison.")]

# ------------------------------------------------------------------ 10 discussion
S += [P("10. Discussion", H1), P("10.1 What was observed", H2)]
S += bullets([f"The best recognised emotion was <b>{best[0]}</b> (F1 {f4(best[1]['f1'])}); the weakest was <b>{worst[0]}</b> (F1 {f4(worst[1]['f1'])}, recall {f4(worst[1]['recall'])}).",
              "Lowest recalls: " + ", ".join(f"<b>{k}</b> {f4(v['recall'])}" for k, v in low_rec) + ". Highest recall: <b>angry</b> " + f4(pc['angry']['recall']) + ".",
              f"<b>Neutral is over-predicted</b>: the model assigned {int(npred[0])} test clips to neutral although only {pc['neutral']['support']} are neutral, giving neutral a precision of just {f4(pc['neutral']['precision'])}.",
              "Most frequent confusions (true -> predicted): " + "; ".join(f"{c['true']} -> {c['predicted']} ({c['count']} clips, {c['share_of_true_class']:.0%})" for c in tc[:6]) + ".",
              f"<b>Fearful</b> speech is rarely predicted wrongly (precision {f4(pc['fearful']['precision'])}) but is often missed (recall {f4(pc['fearful']['recall'])}): the model is cautious about this class."])
S += [P("10.2 Possible explanations (not experimentally tested)", H2),
      P("The following are plausible hypotheses, not proven causes. (i) Sad, calm and neutral speech share low energy and slow, flat delivery, so MFCC-based features may overlap; this would fit the confusions of sad with calm and neutral. "
        "(ii) Angry, disgust, fearful and happy speech can share high arousal, which may explain confusions between them. (iii) With only 16 training actors the model may learn actor-specific habits, which would hurt unseen actors. "
        "(iv) Four test actors give a noisy estimate, so a few clips shift a class recall noticeably. Testing these hypotheses would require additional experiments (for example per-actor analysis or cross-validation over actors).")]

# ------------------------------------------------------------------ 11 limitations
S += [P("11. Limitations", H1)]
S += bullets(["<b>Small, acted dataset</b>: 1,440 clips, 24 actors, two sentences; natural, spontaneous or noisy speech was not evaluated, and the model must not be assumed to understand human emotion in general.",
              "<b>Unseen speakers</b>: generalisation is measured on only four test actors; an unlucky or lucky actor split can move the scores.",
              "<b>Class imbalance</b>: neutral has half as many clips as the other classes (handled with class weights, which does not remove the effect completely).",
              f"<b>Test-set exposure</b>: Experiments 1-3 were compared with test results visible, so the final test score is mildly optimistic" + (" (Experiment 4 was selected or rejected by the validation-only rule)." if E4 else "."),
              "<b>Features</b>: only MFCC-based features were used, as required; no pitch/energy features, no noise-robustness or cross-corpus tests.",
              ("<b>Experiment 4 is a single run</b>: the stronger-dropout variant was trained once with one seed, so its difference from Experiment 3 may be within run-to-run variation." if E4 else "<b>Experiment 4 not executed</b>: the stronger-dropout variant is implemented and verified on synthetic audio only because the RAVDESS data was not available in the build environment; no result for it is claimed."),
              "<b>Ethics</b>: emotion recognition can be misused for surveillance or high-stakes decisions; this system is a research prototype and no demographic fairness analysis was carried out."])

# ------------------------------------------------------------------ 12-13
S += [P("12. Conclusion", H1),
      P(f"The project delivers every element of the problem statement: data collection from RAVDESS, a documented preprocessing and MFCC pipeline, a CNN-BiLSTM-Attention implementation and an F1-centred evaluation under an actor-independent protocol. "
        f"The selected model reaches a test Macro F1 of {f4(T['macro_f1'])} (accuracy {pct(T['accuracy'])}), clearly above the SVM baseline ({f4(SVM['test_macro_f1'])}) and the first deep model, but performance is far from perfect and uneven across emotions, "
        "particularly for sad and fearful speech. The results are an honest, reproducible baseline for acted speech rather than evidence of general emotion understanding."),
      P("13. Future Improvements", H1)]
S += bullets([("Repeat the Experiment 3 / 4 comparison with several random seeds to separate real gains from run-to-run variation." if E4 else "Run Experiment 4 on RAVDESS and, if its validation Macro F1 is higher, compare it fairly with Experiment 3 (the runner and model are ready)."),
              "Actor-grouped k-fold cross-validation with confidence intervals for a more reliable estimate.",
              "SpecAugment-style masking, mixup, and additional corpora (TESS, CREMA-D, EMO-DB) for cross-corpus evaluation.",
              "Additional prosodic features (pitch, energy) or pre-trained speech encoders, evaluated with the same protocol.",
              "A real-time microphone demo and an API, with explicit user-consent and privacy safeguards."])

# ------------------------------------------------------------------ reproducibility + references
S += [P("14. Reproducibility", H1),
      P("python -m venv .venv<br/>.\\.venv\\Scripts\\Activate.ps1<br/>pip install -r requirements.txt<br/>python -m pytest -q<br/>python summarize_results.py<br/>python predict.py path\\to\\clip.wav<br/>python build_report.py --name \"Your Name\"<br/>python build_ppt.py --name \"Your Name\"", CODE),
      P("Project structure", H2),
      P("ser/config.py - settings | ser/data.py - download, labels, split | ser/features.py - preprocessing, MFCC, augmentation | ser/model.py - Experiment 1-3 network | ser/model_exp4.py - Experiment 4 network | "
        "ser/train.py, ser/train_exp4.py - training | ser/evaluate.py - metrics and plots | ser/final.py - result verification and model selection | predict.py - inference | tests/ - automated tests", BODY),
      P("References", H1)]
refs = ["Livingstone, S. R. and Russo, F. A. (2018). The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS). PLoS ONE 13(5): e0196391.",
        "Davis, S. and Mermelstein, P. (1980). Comparison of parametric representations for monosyllabic word recognition in continuously spoken sentences. IEEE Trans. ASSP 28(4), 357-366.",
        "Hochreiter, S. and Schmidhuber, J. (1997). Long short-term memory. Neural Computation 9(8), 1735-1780.",
        "Kingma, D. P. and Ba, J. (2015). Adam: A method for stochastic optimization. ICLR.",
        "Ioffe, S. and Szegedy, C. (2015). Batch normalization: accelerating deep network training by reducing internal covariate shift. ICML.",
        "Srivastava, N. et al. (2014). Dropout: a simple way to prevent neural networks from overfitting. JMLR 15, 1929-1958.",
        "Bahdanau, D., Cho, K. and Bengio, Y. (2015). Neural machine translation by jointly learning to align and translate. ICLR.",
        "McFee, B. et al. (2015). librosa: Audio and music signal analysis in Python. Proc. SciPy.",
        "Pedregosa, F. et al. (2011). Scikit-learn: machine learning in Python. JMLR 12, 2825-2830.",
        "Akcay, M. B. and Oguz, K. (2020). Speech emotion recognition: emotional models, databases, features, preprocessing methods, supporting modalities, and classifiers. Speech Communication 116, 56-76."]
S += [Paragraph(f"[{i}] {r}", ParagraphStyle("ref", parent=BODY, fontSize=8.4, leading=10.6, leftIndent=18, firstLineIndent=-18, alignment=0, spaceAfter=2)) for i, r in enumerate(refs, 1)]

Doc(args.out).multiBuild(S)
print("wrote", args.out)
