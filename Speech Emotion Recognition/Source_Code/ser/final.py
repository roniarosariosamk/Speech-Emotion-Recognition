"""Collect experiment results, verify them, select the final model and build the final summary.

Model-selection rule (defensible and test-set-free): among completed experiments trained on REAL RAVDESS data,
choose the one with the highest VALIDATION Macro F1. Test scores are reported next to it but never used to select.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import LABELS, ROOT

RES, MDL = ROOT / "results", ROOT / "models"

_COMPACT = dict(gaussian_noise=0.08, conv1=dict(filters=32, kernel=5, dropout=0.30), conv2=dict(filters=64, kernel=3, dropout=0.35),
                lstm=dict(units=48, dropout=0.30), attention_units=32, dense=dict(units=64, dropout=0.50), l2=2e-4)
_WIDE = dict(gaussian_noise=0.05, conv_blocks="Conv1D 64(k5,drop .20) -> 128(k5,drop .25) -> 128(k3,drop .30)",
             lstm=dict(units=96, dropout=0.20), attention_units=64, dense=dict(units=128, dropout=0.40), l2=1e-4)

EXPERIMENTS = {
    "experiment_01": dict(label="Experiment 1", title="Wide CNN-BiLSTM-Attention (first full model)", metrics=RES / "metrics.json",
                          model=MDL / "ser_model_best.keras", norm=MDL / "norm.npz", config=_WIDE,
                          selection="validation loss (early stopping, max 40 epochs)"),
    "experiment_02": dict(label="Experiment 2", title="Compact CNN-BiLSTM-Attention", metrics=RES / "experiment_02" / "metrics.json",
                          model=MDL / "experiment_02" / "ser_model_best.keras", norm=MDL / "experiment_02" / "norm.npz", config=_COMPACT,
                          selection="validation loss (early stopping, max 40 epochs)"),
    "experiment_03": dict(label="Experiment 3", title="Compact CNN-BiLSTM-Attention, checkpoint by validation Macro F1",
                          metrics=RES / "experiment_03" / "metrics.json", model=MDL / "experiment_03" / "ser_model_best.keras",
                          norm=MDL / "experiment_03" / "norm.npz", config=_COMPACT,
                          selection="validation Macro F1 (early stopping patience 12, max 60 epochs)"),
    "experiment_04": dict(label="Experiment 4", title="Compact CNN-BiLSTM-Attention + stronger dropout",
                          metrics=RES / "experiment_04" / "metrics.json", model=MDL / "experiment_04" / "ser_model_best.keras",
                          norm=MDL / "experiment_04" / "norm.npz", config=None,
                          selection="validation Macro F1 (early stopping patience 12, max 60 epochs)"),
}


def derive(cm) -> dict:
    """Recompute every headline metric from a confusion matrix (independent check of the stored numbers)."""
    cm = np.asarray(cm, dtype=float)
    tp = np.diag(cm); sup = cm.sum(1); pred = cm.sum(0)
    prec = np.divide(tp, pred, out=np.zeros_like(tp), where=pred > 0)
    rec = np.divide(tp, sup, out=np.zeros_like(tp), where=sup > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return dict(accuracy=float(tp.sum() / cm.sum()), macro_precision=float(prec.mean()), macro_recall=float(rec.mean()),
                macro_f1=float(f1.mean()), weighted_f1=float((f1 * sup).sum() / sup.sum()),
                per_class={l: dict(precision=float(prec[i]), recall=float(rec[i]), f1=float(f1[i]), support=int(sup[i])) for i, l in enumerate(LABELS)})


def top_confusions(cm, k: int = 8) -> list:
    """Largest off-diagonal cells: (true, predicted, count, share of the true class)."""
    cm = np.asarray(cm); out = []
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            if i != j and cm[i, j] > 0:
                out.append(dict(true=LABELS[i], predicted=LABELS[j], count=int(cm[i, j]), share_of_true_class=float(cm[i, j] / cm[i].sum())))
    return sorted(out, key=lambda d: -d["count"])[:k]


def load(key: str):
    info = EXPERIMENTS[key]
    if not info["metrics"].is_file():
        return None
    m = json.loads(info["metrics"].read_text())
    if m.get("synthetic"):
        return None                                   # synthetic smoke tests are never results
    d = derive(m["deep"]["confusion_matrix"]); s = derive(m["svm"]["confusion_matrix"])
    checks = {"deep_accuracy": abs(d["accuracy"] - m["deep"]["accuracy"]) < 1e-9, "deep_macro_f1": abs(d["macro_f1"] - m["deep"]["macro_f1"]) < 1e-9,
              "deep_weighted_f1": abs(d["weighted_f1"] - m["deep"]["weighted_f1"]) < 1e-9, "svm_macro_f1": abs(s["macro_f1"] - m["svm"]["macro_f1"]) < 1e-9}
    cfg = info["config"] or m.get("config")
    hist = m.get("history", {})
    if m["training"].get("best_epoch"): best_epoch = m["training"]["best_epoch"]
    elif "val_macro_f1" in hist: best_epoch = int(np.argmax(hist["val_macro_f1"])) + 1
    else: best_epoch = int(np.argmin(hist["val_loss"])) + 1          # Experiments 1-2 checkpointed on validation loss
    return dict(key=key, label=info["label"], title=info["title"], selection=info["selection"], config=cfg, raw=m, derived=d, svm_derived=s,
                integrity=checks, best_epoch=best_epoch,
                artefacts=dict(model=str(info["model"].relative_to(ROOT)), model_exists=info["model"].is_file(),
                               norm=str(info["norm"].relative_to(ROOT)), norm_exists=info["norm"].is_file()))


def select(exps: dict, force: str | None = None) -> str:
    ok = {k: v for k, v in exps.items() if v and v["artefacts"]["model_exists"] and v["artefacts"]["norm_exists"] and all(v["integrity"].values())}
    if force:
        if force not in ok: raise ValueError(f"{force} is not a valid completed experiment")
        return force
    return max(ok, key=lambda k: (ok[k]["raw"]["deep_val"]["macro_f1"], k))   # tie -> later experiment


def build_summary(force: str | None = None) -> dict:
    exps = {k: load(k) for k in EXPERIMENTS}
    sel = select(exps, force)
    S = exps[sel]; m = S["raw"]; d = S["derived"]
    rows = []
    for k, v in exps.items():
        if v is None:
            rows.append(dict(experiment=k, label=EXPERIMENTS[k]["label"], status="not run / no real-data results"))
            continue
        r = v["raw"]
        rows.append(dict(experiment=k, label=v["label"], status="completed", title=v["title"], params=r["training"]["params"],
                         epochs_run=r["training"]["epochs_run"], best_epoch=v["best_epoch"], training_seconds=r["training"]["seconds"],
                         val_macro_f1=r["deep_val"]["macro_f1"], val_accuracy=r["deep_val"]["accuracy"],
                         test_accuracy=r["deep"]["accuracy"], test_macro_precision=v["derived"]["macro_precision"],
                         test_macro_recall=v["derived"]["macro_recall"], test_macro_f1=r["deep"]["macro_f1"],
                         test_weighted_f1=r["deep"]["weighted_f1"], integrity_ok=all(v["integrity"].values())))
    svm = S["raw"]["svm"]
    return dict(
        project="Speech Emotion Recognition - MFCC + CNN-BiLSTM-Attention (RAVDESS)",
        dataset=dict(name=m["dataset"], clips=m["n_clips"], actors=m["n_actors"], class_counts=m["class_counts"], split=m["split"],
                     labels=LABELS, seed=m["training"]["seed"],
                     features=dict(shape=m["features"]["shape"], sr=m["features"]["sr"], n_mfcc=m["features"]["n_mfcc"], hop=m["features"]["hop"],
                                   duration=m["features"]["duration"], n_fft=2048, n_mels=128)),
        experiments=rows,
        svm_baseline=dict(params=svm["params"], test_accuracy=svm["accuracy"], test_macro_f1=svm["macro_f1"], test_weighted_f1=svm["weighted_f1"],
                          test_macro_precision=S["svm_derived"]["macro_precision"], test_macro_recall=S["svm_derived"]["macro_recall"],
                          note="identical across experiments (same split, features and tuning on validation actors only)"),
        selected_final_model=dict(
            experiment=sel, label=S["label"], title=S["title"], checkpoint=S["artefacts"]["model"], normaliser=S["artefacts"]["norm"],
            selection_rule="highest validation Macro F1 among completed real-data experiments (test set not used for selection)",
            config=S["config"], parameters=m["training"]["params"], best_epoch=S["best_epoch"], epochs_run=m["training"]["epochs_run"],
            training_seconds=m["training"]["seconds"], validation_macro_f1=m["deep_val"]["macro_f1"],
            test=dict(accuracy=d["accuracy"], macro_precision=d["macro_precision"], macro_recall=d["macro_recall"], macro_f1=d["macro_f1"],
                      weighted_f1=d["weighted_f1"], per_class=d["per_class"], confusion_matrix=m["deep"]["confusion_matrix"]),
            top_confusions=top_confusions(m["deep"]["confusion_matrix"]),
            gain_over_svm_macro_f1=d["macro_f1"] - svm["macro_f1"]),
        experiment_4_status=("completed" if exps["experiment_04"] else
                             "NOT run on real data (RAVDESS not available in the build environment). Code, model and runner are implemented and "
                             "verified on synthetic audio only; no Experiment 4 result is claimed."),
        limitations=[
            "RAVDESS is acted speech from 24 actors and two sentences; real-world, noisy or spontaneous speech will behave differently.",
            "The test set contains 240 clips from 4 actors, so every metric has substantial sampling variance; differences of a few F1 points between experiments are not statistically established.",
            "Experiment 3 was selected using validation Macro F1. However, test results were also examined when comparing Experiments 1-3, so repeated model development informed by the test results may make the reported test performance somewhat optimistic.",
            "Sad and fearful speech have low recall; neutral is over-predicted (low precision). Explanations (acoustic similarity, small data) are plausible but were not tested experimentally.",
            "Only MFCC-based features were used; no cross-corpus or noise-robustness evaluation was performed.",
        ])


def markdown(S: dict) -> str:
    f = lambda x: f"{x:.4f}"
    sel = S["selected_final_model"]; t = sel["test"]
    L = [f"# Final results summary\n", f"**Project:** {S['project']}\n",
         f"**Dataset:** {S['dataset']['name']} - {S['dataset']['clips']} clips, {S['dataset']['actors']} actors, actor-independent split "
         f"(train {S['dataset']['split']['train_actors'][0]}-{S['dataset']['split']['train_actors'][-1]}, val {S['dataset']['split']['val_actors'][0]}-{S['dataset']['split']['val_actors'][-1]}, "
         f"test {S['dataset']['split']['test_actors'][0]}-{S['dataset']['split']['test_actors'][-1]}), seed {S['dataset']['seed']}\n",
         "## Experiment comparison\n", "| Experiment | Parameters | Epochs run (best) | Train time (s) | Val Macro F1 | Test accuracy | Test Macro F1 | Test weighted F1 |", "|---|---|---|---|---|---|---|---|"]
    for r in S["experiments"]:
        if r["status"] != "completed":
            L.append(f"| {r['label']} | - | - | - | - | - | - | not run (see status below) |"); continue
        L.append(f"| {r['label']}{' **(selected)**' if r['experiment'] == sel['experiment'] else ''} | {r['params']:,} | {r['epochs_run']} ({r['best_epoch']}) | {r['training_seconds']:.0f} | "
                 f"{f(r['val_macro_f1'])} | {f(r['test_accuracy'])} | {f(r['test_macro_f1'])} | {f(r['test_weighted_f1'])} |")
    b = S["svm_baseline"]
    L += [f"| SVM baseline (RBF, C={b['params']['C']}, gamma={b['params']['gamma']}) | - | - | - | - | {f(b['test_accuracy'])} | {f(b['test_macro_f1'])} | {f(b['test_weighted_f1'])} |\n",
          f"**Experiment 4 status:** {S['experiment_4_status']}\n", f"## Selected final model: {sel['label']}\n",
          f"- Checkpoint: `{sel['checkpoint']}`  |  Normaliser: `{sel['normaliser']}`", f"- Selection rule: {sel['selection_rule']}",
          f"- Validation Macro F1 {f(sel['validation_macro_f1'])} | Test accuracy {f(t['accuracy'])} | Macro precision {f(t['macro_precision'])} | Macro recall {f(t['macro_recall'])} | "
          f"Macro F1 {f(t['macro_f1'])} | Weighted F1 {f(t['weighted_f1'])}", f"- Macro F1 gain over SVM baseline: {sel['gain_over_svm_macro_f1']:+.4f}\n",
          "### Per-class test results\n", "| Emotion | Precision | Recall | F1 | Support |", "|---|---|---|---|---|"]
    L += [f"| {k} | {f(v['precision'])} | {f(v['recall'])} | {f(v['f1'])} | {v['support']} |" for k, v in t["per_class"].items()]
    L += ["\n### Most frequent test confusions (observed)\n"] + [f"- {c['true']} -> {c['predicted']}: {c['count']} clips ({c['share_of_true_class']:.0%} of {c['true']})" for c in sel["top_confusions"][:6]]
    L += ["\n## Limitations\n"] + [f"- {x}" for x in S["limitations"]]
    return "\n".join(L) + "\n"

