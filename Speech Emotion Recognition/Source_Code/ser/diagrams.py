"""Matplotlib block diagrams used in the report and slides."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

INDIGO, CYAN, AMBER, SLATE, LIGHT = "#4F46E5", "#06B6D4", "#F59E0B", "#334155", "#EEF2FF"


def _box(ax, x, y, w, h, text, fc=LIGHT, ec=INDIGO, tc=SLATE, fs=9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec=ec, lw=1.6))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc,
            fontweight="bold" if bold else "normal", linespacing=1.3)


def _arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color=SLATE, lw=1.6))


def pipeline(path):
    steps = ["Audio\n(RAVDESS .wav)", "Pre-processing\nresample 22.05 kHz\ntrim · normalise\n3 s window", "Augmentation\nnoise · pitch\nstretch · shift\n(train only)",
             "MFCC features\n40 MFCC + Δ + ΔΔ\n130 × 120", "Deep model\nCNN → BiLSTM\n→ Attention", "Evaluation\nmacro / weighted F1\nconfusion matrix"]
    fig, ax = plt.subplots(figsize=(12, 2.6)); ax.set_xlim(0, 12.6); ax.set_ylim(0, 2.6); ax.axis("off")
    for i, s in enumerate(steps):
        x = 0.1 + i * 2.1
        _box(ax, x, 0.35, 1.8, 1.9, s, fc=INDIGO if i in (3, 4) else LIGHT, tc="white" if i in (3, 4) else SLATE, fs=8.6, bold=i in (3, 4))
        if i < len(steps) - 1: _arrow(ax, x + 1.82, 1.3, x + 2.08, 1.3)
    plt.tight_layout(pad=0.2); fig.savefig(path, dpi=200, transparent=False, facecolor="white"); plt.close(fig)


def architecture(path, n_frames=130, n_feat=120, n_classes=8):
    L = [("Input\n(130 × 120)\nMFCC+Δ+ΔΔ", LIGHT, SLATE), ("Gaussian\nnoise 0.05", LIGHT, SLATE),
         ("Conv1D 64 · k5\nBN · ReLU\nMaxPool · Drop .2", CYAN, "white"), ("Conv1D 128 · k5\nBN · ReLU\nMaxPool · Drop .25", CYAN, "white"),
         ("Conv1D 128 · k3\nBN · ReLU\nMaxPool · Drop .3", CYAN, "white"), ("BiLSTM 96\n(return seq.)", INDIGO, "white"),
         ("Attention\npooling", AMBER, "white"), ("Dense 128\nReLU · Drop .4", LIGHT, SLATE), (f"Softmax\n{n_classes} emotions", SLATE, "white")]
    fig, ax = plt.subplots(figsize=(12, 2.8)); ax.set_xlim(0, 12.8); ax.set_ylim(0, 2.8); ax.axis("off")
    w = 1.25
    for i, (t, fc, tc) in enumerate(L):
        x = 0.1 + i * 1.42
        _box(ax, x, 0.7, w, 1.5, t, fc=fc, ec=fc if fc != LIGHT else INDIGO, tc=tc, fs=7.6, bold=True)
        if i < len(L) - 1: _arrow(ax, x + w + .01, 1.45, x + 1.41, 1.45)
    ax.text(3.3, 0.3, "local spectral-pattern extraction", ha="center", fontsize=8, color=SLATE, style="italic")
    ax.text(7.3, 0.3, "temporal context + focus on salient frames", ha="center", fontsize=8, color=SLATE, style="italic")
    plt.tight_layout(pad=0.2); fig.savefig(path, dpi=200, facecolor="white"); plt.close(fig)
