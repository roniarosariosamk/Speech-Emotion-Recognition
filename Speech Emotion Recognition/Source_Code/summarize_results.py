"""Verify all experiment results, select the final model and write results/final/ (JSON + Markdown + figures).

    python summarize_results.py                       # select by validation Macro F1
    python summarize_results.py --force experiment_03 # pin a specific experiment (must be a valid completed run)
Existing experiment folders are only READ; everything is written to results/final/.
"""
import argparse
import json
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ser import diagrams, final
from ser.config import LABELS, ROOT
from ser.evaluate import PALETTE, plot_confusion


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--force"); a = ap.parse_args()
    S = final.build_summary(a.force)
    out = ROOT / "results" / "final"; out.mkdir(parents=True, exist_ok=True)
    sel = S["selected_final_model"]; key = sel["experiment"]
    (out / "final_summary.json").write_text(json.dumps(S, indent=2)); (out / "final_summary.md").write_text(final.markdown(S))

    # figures -----------------------------------------------------------------------------------
    done = [r for r in S["experiments"] if r["status"] == "completed"]
    names = [r["label"].replace("Experiment ", "Exp ") for r in done] + ["SVM"]
    fig, ax = plt.subplots(figsize=(8.5, 4)); x = np.arange(len(names)); w = .38
    ax.bar(x[:-1] - w / 2, [r["val_macro_f1"] for r in done], w, color=PALETTE["warm"], label="validation Macro F1")
    test = [r["test_macro_f1"] for r in done] + [S["svm_baseline"]["test_macro_f1"]]
    ax.bar(x + w / 2, test, w, color=[PALETTE["primary"]] * len(done) + [PALETTE["grey"]], label="test Macro F1")
    for i, v in enumerate(test): ax.text(i + w / 2, v + .008, f"{v:.3f}", ha="center", fontsize=9)
    for i, r in enumerate(done): ax.text(i - w / 2, r["val_macro_f1"] + .008, f"{r['val_macro_f1']:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylim(0, .7); ax.set_ylabel("Macro F1"); ax.legend(frameon=False)
    ax.set_title("Experiment comparison (Macro F1)", fontweight="bold"); ax.grid(axis="y", alpha=.25); ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); fig.savefig(out / "comparison_macro_f1.png", dpi=200); plt.close(fig)

    pc = sel["test"]["per_class"]; order = sorted(LABELS, key=lambda l: pc[l]["f1"])
    fig, ax = plt.subplots(figsize=(8.5, 4)); y = np.arange(len(order)); h = .27
    for k, (m, c) in enumerate([("precision", PALETTE["accent"]), ("recall", PALETTE["warm"]), ("f1", PALETTE["primary"])]):
        ax.barh(y + (k - 1) * h, [pc[l][m] for l in order], h, color=c, label=m)
    ax.set_yticks(y); ax.set_yticklabels(order); ax.set_xlim(0, 1); ax.legend(frameon=False, ncol=3, loc="lower right")
    ax.set_title(f"{sel['label']}: per-class test precision / recall / F1", fontweight="bold"); ax.grid(axis="x", alpha=.25); ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); fig.savefig(out / "per_class_prf.png", dpi=200); plt.close(fig)

    plot_confusion(sel["test"]["confusion_matrix"], LABELS, out / "final_confusion_matrix.png", f"{sel['label']}: test confusion matrix (row-normalised)")
    diagrams.architecture(out / "diagram_architecture.png", cfg=sel["config"] if "conv1" in (sel["config"] or {}) else None)
    diagrams.pipeline(out / "diagram_pipeline.png")
    src = ROOT / "results" / key
    for f in ("training_curves.png", "mfcc_example.png", "class_distribution.png", "confusion_matrix_svm.png", "f1_per_class.png",
              "classification_report_deep.txt", "classification_report_svm.txt"):
        if (src / f).is_file(): shutil.copy2(src / f, out / f)
    print(f"[final] selected {key} ({sel['label']}); val macro-F1 {sel['validation_macro_f1']:.4f}, test macro-F1 {sel['test']['macro_f1']:.4f}")
    print(f"[final] wrote {out}")


if __name__ == "__main__":
    main()
