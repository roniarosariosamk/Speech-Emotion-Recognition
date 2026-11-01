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


def architecture(path, n_classes=8, cfg=None):
    """Architecture block diagram. `cfg` describes the network (defaults to the Experiment-3 model)."""
    c = cfg or dict(gaussian_noise=0.08, conv1=dict(filters=32, kernel=5, dropout=0.30), conv2=dict(filters=64, kernel=3, dropout=0.35),
                    lstm=dict(units=48, dropout=0.30), attention_units=32, dense=dict(units=64, dropout=0.50))
    L = [("Input\n130 x 120\nMFCC+d+dd", LIGHT, SLATE), (f"Gaussian\nnoise {c['gaussian_noise']}", LIGHT, SLATE),
         (f"Conv1D {c['conv1']['filters']} - k{c['conv1']['kernel']}\nBN - ReLU\nMaxPool - Drop {c['conv1']['dropout']}", CYAN, "white"),
         (f"Conv1D {c['conv2']['filters']} - k{c['conv2']['kernel']}\nBN - ReLU\nMaxPool - Drop {c['conv2']['dropout']}", CYAN, "white"),
         (f"BiLSTM {c['lstm']['units']}\n(return seq.)\nDrop {c['lstm']['dropout']}", INDIGO, "white"),
         (f"Attention\npooling ({c['attention_units']})", AMBER, "white"),
         (f"Dense {c['dense']['units']}\nReLU - Drop {c['dense']['dropout']}", LIGHT, SLATE), (f"Softmax\n{n_classes} emotions", SLATE, "white")]
    fig, ax = plt.subplots(figsize=(12, 2.8)); ax.set_xlim(0, 12.8); ax.set_ylim(0, 2.8); ax.axis("off")
    w = 1.3
    for i, (t, fc, tc) in enumerate(L):
        x = 0.1 + i * 1.58
        _box(ax, x, 0.7, w, 1.5, t, fc=fc, ec=fc if fc != LIGHT else INDIGO, tc=tc, fs=7.8, bold=True)
        if i < len(L) - 1: _arrow(ax, x + w + .01, 1.45, x + 1.57, 1.45)
    ax.text(4.2, 0.3, "local spectral-pattern extraction (CNN)", ha="center", fontsize=8, color=SLATE, style="italic")
    ax.text(9.0, 0.3, "temporal context + focus on salient frames", ha="center", fontsize=8, color=SLATE, style="italic")
    plt.tight_layout(pad=0.2); fig.savefig(path, dpi=200, facecolor="white"); plt.close(fig)
