"""Generate the presentation from results/metrics.json  (python build_ppt.py [--name "Your Name"])."""
import argparse
import json
from pathlib import Path

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_LABEL_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt, Emu

ROOT = Path(__file__).resolve().parent; RES = ROOT / "results"
ap = argparse.ArgumentParser(); ap.add_argument("--name", default="Student Name")
ap.add_argument("--out", default=str(ROOT / "Speech_Emotion_Recognition_Presentation.pptx")); args = ap.parse_args()
M = json.loads((RES / "metrics.json").read_text()); SYN = M["synthetic"]; deep, svm, tr = M["deep"], M["svm"], M["training"]

C = lambda h: RGBColor.from_string(h)
INDIGO, CYAN, AMBER, SLATE, MUTED, LIGHT, WHITE, DARK = map(C, ["4F46E5", "06B6D4", "F59E0B", "1E293B", "64748B", "EEF2FF", "FFFFFF", "0F172A"])
prs = Presentation(); prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]


def rect(s, x, y, w, h, fill, shape=MSO_SHAPE.RECTANGLE, line=None):
    r = s.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h)); r.fill.solid(); r.fill.fore_color.rgb = fill
    if line: r.line.color.rgb = line; r.line.width = Pt(1.25)
    else: r.line.fill.background()
    r.shadow.inherit = False; return r


def text(s, x, y, w, h, t, size=16, color=SLATE, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, font="Calibri"):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, line in enumerate(t if isinstance(t, list) else [t]):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); p.alignment = align
        r = p.add_run(); r.text = line; r.font.size = Pt(size); r.font.bold = bold; r.font.color.rgb = color; r.font.name = font
    return tb


def bullets(s, x, y, w, items, size=17, gap=10):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(0.5)); tf = tb.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = 0
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); p.space_after = Pt(gap)
        r0 = p.add_run(); r0.text = "\u25A0  "; r0.font.size = Pt(size - 6); r0.font.color.rgb = INDIGO
        head, _, rest = it.partition("|")
        if rest:
            r1 = p.add_run(); r1.text = head; r1.font.bold = True; r1.font.size = Pt(size); r1.font.color.rgb = SLATE; r1.font.name = "Calibri"
            r2 = p.add_run(); r2.text = rest; r2.font.size = Pt(size); r2.font.color.rgb = SLATE; r2.font.name = "Calibri"
        else:
            r1 = p.add_run(); r1.text = head; r1.font.size = Pt(size); r1.font.color.rgb = SLATE; r1.font.name = "Calibri"
    return tb


def slide(title, kicker="", notes=""):
    s = prs.slides.add_slide(BLANK); rect(s, 0, 0, 13.333, 0.12, INDIGO)
    if kicker: text(s, 0.7, 0.42, 9, 0.3, kicker.upper(), 12, INDIGO, True)
    text(s, 0.7, 0.72, 11.9, 0.8, title, 32, DARK, True)
    text(s, 0.7, 7.08, 8, 0.25, "Speech Emotion Recognition  |  MFCC + CNN-BiLSTM-Attention", 10, MUTED)
    text(s, 12.0, 7.08, 0.65, 0.25, str(len(prs.slides)), 10, MUTED, align=PP_ALIGN.RIGHT)
    if SYN and kicker.lower().startswith("results"):
        b = rect(s, 7.2, 0.38, 5.45, 0.34, C("FEF3C7"), MSO_SHAPE.ROUNDED_RECTANGLE, AMBER)
        text(s, 7.2, 0.38, 5.45, 0.34, "SYNTHETIC SMOKE-TEST DATA - NOT RAVDESS RESULTS", 11, C("92400E"), True, PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE)
    if notes: s.notes_slide.notes_text_frame.text = notes
    return s


def card(s, x, y, w, h, head, body, accent=INDIGO, bsize=14):
    rect(s, x, y, w, h, LIGHT, MSO_SHAPE.ROUNDED_RECTANGLE); rect(s, x, y + 0.18, 0.08, h - 0.36, accent)
    text(s, x + 0.3, y + 0.2, w - 0.5, 0.4, head, 17, DARK, True); text(s, x + 0.3, y + 0.68, w - 0.5, h - 0.8, body, bsize, SLATE)


def kpi(s, x, y, w, value, label, fill=INDIGO):
    rect(s, x, y, w, 1.55, fill, MSO_SHAPE.ROUNDED_RECTANGLE)
    text(s, x, y + 0.2, w, 0.8, value, 40, WHITE, True, PP_ALIGN.CENTER); text(s, x, y + 1.0, w, 0.4, label, 13, WHITE, False, PP_ALIGN.CENTER)


def pic(s, name, x, y, w=None, h=None):
    return s.shapes.add_picture(str(RES / name), Inches(x), Inches(y), Inches(w) if w else None, Inches(h) if h else None)


# 1 ── title
s = prs.slides.add_slide(BLANK); rect(s, 0, 0, 13.333, 7.5, DARK); rect(s, 0, 0, 0.35, 7.5, INDIGO); rect(s, 0.9, 3.55, 1.6, 0.07, AMBER)
text(s, 0.9, 1.5, 11, 0.4, "INDIVIDUAL MAJOR PROJECT  |  MACHINE LEARNING INTERNSHIP", 14, CYAN, True)
text(s, 0.9, 2.0, 11.5, 1.5, "Speech Emotion Recognition", 54, WHITE, True)
text(s, 0.9, 3.85, 11, 0.9, "MFCC features + CNN-BiLSTM-Attention deep learning model, evaluated with F1-score", 22, C("CBD5E1"))
text(s, 0.9, 5.6, 8, 0.4, args.name, 20, WHITE, True); text(s, 0.9, 6.05, 8, 0.4, f"Dataset: {M['dataset']}", 14, C("94A3B8"))
s.notes_slide.notes_text_frame.text = "Introduce the project: recognise emotion from speech using MFCC features and a deep network, evaluated with F1-score."

# 2 ── problem
s = slide("Why recognise emotion in speech?", "Problem statement", "Emotion matters in call-centre analytics, tutoring, wellbeing apps and HRI. State the assigned problem statement.")
card(s, 0.7, 1.9, 5.9, 2.2, "Assigned task", "Build an ML system that recognises emotion in speech using MFCC features and a deep model, covering data collection, preprocessing, model implementation and F1-score evaluation.")
card(s, 0.7, 4.35, 5.9, 2.3, "Why it is hard", "Emotion is subjective, speakers differ, and public corpora are small - models tend to memorise voices instead of emotion.", AMBER)
rect(s, 7.0, 1.9, 5.65, 4.75, DARK, MSO_SHAPE.ROUNDED_RECTANGLE); text(s, 7.35, 2.15, 5, 0.4, "Requirement checklist", 18, CYAN, True)
for i, (a, b) in enumerate([("Data collection", "RAVDESS download + labels"), ("Preprocessing", "trim, normalise, augment"), ("MFCC features", "40 MFCC + delta + delta-delta"),
                            ("Deep model", "CNN - BiLSTM - Attention"), ("F1-score evaluation", "macro / weighted / per-class")]):
    y = 2.85 + i * 0.75; rect(s, 7.35, y + .04, 0.32, 0.32, CYAN, MSO_SHAPE.OVAL); text(s, 7.35, y + .04, .32, .32, "\u2713", 13, DARK, True, PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE)
    text(s, 7.85, y, 4.6, 0.35, a, 16, WHITE, True); text(s, 7.85, y + 0.32, 4.6, 0.3, b, 12, C("CBD5E1"))

# 3 ── pipeline
s = slide("End-to-end pipeline", "Approach", "Walk through the six stages left to right. Each stage is a separate tested module.")
pic(s, "diagram_pipeline.png", 0.7, 2.0, w=11.9)
bullets(s, 0.9, 4.8, 11.5, ["Reproducible|  - one command, fixed seeds, configuration dataclasses, unit tests",
                           "Leak-free|  - speaker-independent split; normalisation statistics fitted on training data only"], 17)

# 4 ── dataset
s = slide("Data collection: RAVDESS", "Dataset", "RAVDESS speech: 1,440 clips, 24 actors, 8 emotions. Labels are in the filename. Actors are split so no voice is shared across partitions.")
kpi(s, 0.7, 1.85, 2.9, f"{M['n_clips']:,}", "clips used"); kpi(s, 3.8, 1.85, 2.9, str(M["n_actors"]), "actors", CYAN); kpi(s, 6.9, 1.85, 2.9, "8", "emotion classes", AMBER)
sp = M["split"]
card(s, 0.7, 3.65, 9.1, 1.4, "Speaker-independent split", f"Train {sp['train_clips']} clips ({len(sp['train_actors'])} actors)  |  Validation {sp['val_clips']} ({len(sp['val_actors'])})  |  Test {sp['test_clips']} ({len(sp['test_actors'])}). No actor appears in two partitions.", INDIGO, 14)
bullets(s, 0.7, 5.35, 9.1, ["Labels|  parsed from filenames such as 03-01-05-02-01-01-07.wav", "Source|  Zenodo, CC BY-NC-SA 4.0, auto-downloaded by --download"], 15, 6)
pic(s, "class_distribution.png", 10.0, 1.9, w=2.95)
text(s, 10.0, 3.0, 2.95, 0.4, "", 10)

# 5 ── preprocessing + MFCC
s = slide("Pre-processing and MFCC features", "Features", "Explain trimming, fixed window, augmentation (train only), and MFCC + deltas giving a 130 x 120 matrix.")
bullets(s, 0.7, 1.85, 6.2, ["Resample|  to 22.05 kHz mono; trim silence; peak-normalise", "Fixed window|  3 s centre crop / zero-pad", "Augment|  noise, pitch +/-2 st, stretch 0.9-1.1x, shift (train only)",
                            "MFCC|  40 coefficients + delta + delta-delta = 120 features", "Standardise|  train-set mean / std per coefficient"], 16, 9)
rect(s, 0.7, 6.0, 6.2, 0.8, LIGHT, MSO_SHAPE.ROUNDED_RECTANGLE); text(s, 0.95, 6.0, 5.8, 0.8, f"Input tensor per clip:  {M['features']['shape'][0]} frames  x  {M['features']['shape'][1]} features", 16, INDIGO, True, anchor=MSO_ANCHOR.MIDDLE)
pic(s, "mfcc_example.png", 7.2, 2.1, w=5.5)
text(s, 7.2, 4.5, 5.5, 0.6, "MFCCs capture the spectral envelope (vocal-tract shape); deltas capture how it changes - key cues for arousal and emotion.", 13, MUTED)

# 6 ── architecture
s = slide("Model: CNN - BiLSTM - Attention", "Model", f"Three conv blocks reduce 130 frames to 16; BiLSTM adds temporal context; attention pools frames. {tr['params']:,} parameters.")
pic(s, "diagram_architecture.png", 0.5, 1.85, w=12.3)
card(s, 0.7, 4.75, 3.9, 1.9, "Conv1D x 3", "Local spectral patterns; BatchNorm, ReLU, MaxPool, Dropout.", CYAN, 14)
card(s, 4.75, 4.75, 3.9, 1.9, "BiLSTM (96)", "Context in both time directions.", INDIGO, 14)
card(s, 8.8, 4.75, 3.85, 1.9, "Attention pooling", "Learns which frames carry emotion.", AMBER, 14)

# 7 ── training
s = slide("Training and baseline", "Model", "Adam with LR scheduling, early stopping on validation loss, class weights. SVM baseline tuned on validation actors.")
card(s, 0.7, 1.9, 5.9, 4.7, "Deep model", ["Adam, lr 1e-3, ReduceLROnPlateau", "Class-weighted cross-entropy", f"Batch {tr['batch_size']}, up to {tr['epochs_max']} epochs (ran {tr['epochs_run']})", "Early stopping on validation loss", f"{tr['params']:,} parameters", f"Training time {tr['seconds']:.0f} s (CPU)"], INDIGO, 15)
card(s, 6.9, 1.9, 5.75, 2.6, "Classical baseline", ["RBF-SVM on utterance-level MFCC mean + std", f"Tuned on validation actors: C = {svm['params']['C']}, gamma = {svm['params']['gamma']}"], AMBER, 15)
card(s, 6.9, 4.7, 5.75, 1.9, "Regularisation", "Dropout, L2, Gaussian noise, augmentation, speaker-independent early stopping.", CYAN, 15)

# 8 ── metric
s = slide("Evaluation with F1-score", "Evaluation", "Macro-F1 is primary because it weights every emotion equally. Test actors are untouched during training and tuning.")
rect(s, 0.7, 1.9, 12.0, 1.3, DARK, MSO_SHAPE.ROUNDED_RECTANGLE); text(s, 0.7, 1.9, 12.0, 1.3, "F1  =  2 \u00B7 Precision \u00B7 Recall  /  (Precision + Recall)", 28, WHITE, True, PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE)
card(s, 0.7, 3.5, 3.85, 2.9, "Macro-F1", "Mean of per-class F1. Primary metric; penalises ignoring any emotion.", INDIGO, 14)
card(s, 4.75, 3.5, 3.85, 2.9, "Weighted / micro-F1", "Weighted by class support; micro pools all decisions (equals accuracy).", CYAN, 14)
card(s, 8.8, 3.5, 3.9, 2.9, "Per-class + confusion", "Precision, recall, F1 for each emotion and row-normalised confusion matrices.", AMBER, 14)

# 9 ── results overview
s = slide("Results: overall F1", "Results", "Report macro-F1 for deep vs baseline, then discuss the difference. " + ("These numbers are from the synthetic smoke test only." if SYN else ""))
kpi(s, 0.7, 1.9, 2.9, f"{deep['macro_f1']:.3f}", "Macro-F1 (proposed)"); kpi(s, 3.8, 1.9, 2.9, f"{deep['weighted_f1']:.3f}", "Weighted-F1", CYAN); kpi(s, 6.9, 1.9, 2.9, f"{deep['accuracy']:.3f}", "Accuracy", AMBER)
kpi(s, 10.0, 1.9, 2.65, f"{svm['macro_f1']:.3f}", "SVM baseline macro-F1", SLATE)
cd = CategoryChartData(); cd.categories = ["Macro-F1", "Weighted-F1", "Micro-F1", "Accuracy"]
cd.add_series("SVM baseline", [svm[k] for k in ("macro_f1", "weighted_f1", "micro_f1", "accuracy")]); cd.add_series("CNN-BiLSTM-Attention", [deep[k] for k in ("macro_f1", "weighted_f1", "micro_f1", "accuracy")])
ch = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.7), Inches(3.7), Inches(12), Inches(3.2), cd).chart
ch.has_legend = True; ch.legend.position = XL_LEGEND_POSITION.TOP; ch.legend.include_in_layout = False; ch.legend.font.size = Pt(12)
ch.value_axis.maximum_scale = 1.0; ch.value_axis.minimum_scale = 0; ch.value_axis.major_gridlines.format.line.color.rgb = C("E2E8F0"); ch.value_axis.tick_labels.font.size = Pt(11); ch.category_axis.tick_labels.font.size = Pt(12)
for ser, col in zip(ch.plots[0].series, (C("94A3B8"), INDIGO)):
    ser.format.fill.solid(); ser.format.fill.fore_color.rgb = col
pl = ch.plots[0]; pl.gap_width = 70; pl.has_data_labels = True; pl.data_labels.number_format = "0.000"; pl.data_labels.number_format_is_linked = False; pl.data_labels.font.size = Pt(11); pl.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END

# 10 ── per class
s = slide("Results: per-class F1 and errors", "Results", "Point out the best and worst classes and where the confusion matrix shows systematic mistakes.")
labels = list(deep["per_class"]); cd = CategoryChartData(); cd.categories = labels
cd.add_series("SVM baseline", [svm["per_class"][l]["f1"] for l in labels]); cd.add_series("CNN-BiLSTM-Attention", [deep["per_class"][l]["f1"] for l in labels])
ch = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.5), Inches(1.7), Inches(7.1), Inches(5.1), cd).chart
ch.has_legend = True; ch.legend.position = XL_LEGEND_POSITION.TOP; ch.legend.include_in_layout = False; ch.legend.font.size = Pt(11)
ch.value_axis.maximum_scale = 1.0; ch.value_axis.minimum_scale = 0; ch.value_axis.major_gridlines.format.line.color.rgb = C("E2E8F0"); ch.value_axis.tick_labels.font.size = Pt(10); ch.category_axis.tick_labels.font.size = Pt(10)
for ser, col in zip(ch.plots[0].series, (C("94A3B8"), INDIGO)): ser.format.fill.solid(); ser.format.fill.fore_color.rgb = col
ch.plots[0].gap_width = 60
pic(s, "confusion_matrix.png", 7.75, 1.75, h=5.1)

# 11 ── discussion
best = max(deep["per_class"].items(), key=lambda kv: kv[1]["f1"]); worst = min(deep["per_class"].items(), key=lambda kv: kv[1]["f1"])
s = slide("Discussion and limitations", "Analysis", "Be upfront about limitations: acted speech, few test speakers, MFCC-only features.")
card(s, 0.7, 1.9, 5.9, 2.0, "Findings", f"Best class: {best[0]} (F1 {best[1]['f1']:.3f}). Weakest: {worst[0]} (F1 {worst[1]['f1']:.3f}). Check the confusion matrix for arousal-similar pairs.", INDIGO, 14)
card(s, 0.7, 4.15, 5.9, 2.5, "Why speaker-independent matters", "A clip-level random split lets the model recognise the voice, inflating scores. Our actor-level split measures generalisation to new speakers.", CYAN, 14)
card(s, 6.9, 1.9, 5.75, 4.75, "Limitations", ["Acted speech, 24 speakers, two sentences", "Four test actors -> high variance; use speaker-grouped k-fold", "MFCC-only: weak on valence", "No fairness / demographic analysis"], AMBER, 15)

# 12 ── conclusion
s = slide("Conclusion and future work", "Wrap-up", "Summarise delivered work, then future work: cross-corpus evaluation, SpecAugment, pretrained speech encoders, real-time demo.")
card(s, 0.7, 1.9, 5.9, 4.75, "Delivered", ["Data collection, preprocessing, MFCC pipeline", "CNN-BiLSTM-Attention model + SVM baseline", "F1-based evaluation, speaker-independent", "Unit tests, one-command reproducibility", "Source code + PDF report + this deck"], INDIGO, 15)
card(s, 6.9, 1.9, 5.75, 4.75, "Next steps", ["Leave-speakers-out cross-validation with confidence intervals", "SpecAugment / mixup; TESS, CREMA-D cross-corpus tests", "wav2vec-style encoders under the same protocol", "Real-time microphone demo + REST API"], CYAN, 15)

# 13 ── thanks
s = prs.slides.add_slide(BLANK); rect(s, 0, 0, 13.333, 7.5, DARK); rect(s, 0, 0, 0.35, 7.5, INDIGO)
text(s, 0.9, 2.6, 11, 1, "Thank you", 54, WHITE, True); text(s, 0.9, 3.9, 11, 0.6, "Questions?", 26, CYAN, True)
text(s, 0.9, 5.4, 11, 0.4, "Code: run_pipeline.py  |  Report: Speech_Emotion_Recognition_Report.pdf", 14, C("94A3B8"))
prs.save(args.out); print("wrote", args.out)
