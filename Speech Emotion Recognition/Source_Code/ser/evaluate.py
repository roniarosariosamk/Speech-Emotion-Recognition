"""Evaluation with F1-score (macro / weighted / micro / per-class) and publication-quality plots."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_recall_fscore_support)

PALETTE = {"primary": "#4F46E5", "accent": "#06B6D4", "warm": "#F59E0B", "grey": "#94A3B8"}


def compute_metrics(y_true, y_pred, labels):
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=range(len(labels)), zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "per_class": {l: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
                      for i, l in enumerate(labels)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=range(len(labels))).tolist(),
        "report_text": classification_report(y_true, y_pred, labels=range(len(labels)),
                                             target_names=labels, zero_division=0, digits=3),
    }


def plot_confusion(cm, labels, path, title):
    cm = np.array(cm, dtype=float)
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7.2, 6))
    sns.heatmap(norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=labels, yticklabels=labels,
                cbar_kws={"label": "row-normalised"}, linewidths=.5, ax=ax, vmin=0, vmax=1)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title, fontweight="bold")
    plt.xticks(rotation=35, ha="right"); plt.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def plot_history(h, path):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ep = range(1, len(h["loss"]) + 1)
    ax[0].plot(ep, h["loss"], color=PALETTE["primary"], label="train"); ax[0].plot(ep, h["val_loss"], color=PALETTE["warm"], label="validation")
    ax[0].set_title("Loss", fontweight="bold"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(ep, h["accuracy"], color=PALETTE["primary"], label="train"); ax[1].plot(ep, h["val_accuracy"], color=PALETTE["warm"], label="validation")
    ax[1].set_title("Accuracy", fontweight="bold"); ax[1].set_xlabel("epoch"); ax[1].legend()
    for a in ax: a.grid(alpha=.25); a.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def plot_f1_per_class(deep, base, labels, path):
    x = np.arange(len(labels)); w = .38
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - w / 2, [base["per_class"][l]["f1"] for l in labels], w, color=PALETTE["grey"], label="SVM baseline")
    ax.bar(x + w / 2, [deep["per_class"][l]["f1"] for l in labels], w, color=PALETTE["primary"], label="CNN-BiLSTM-Attention")
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0, 1.05); ax.set_ylabel("F1-score")
    ax.set_title("Per-class F1 on held-out test speakers", fontweight="bold")
    ax.legend(frameon=False); ax.grid(axis="y", alpha=.25); ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def plot_class_distribution(counts: dict, path):
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.bar(list(counts), list(counts.values()), color=PALETTE["accent"])
    ax.set_title("Class distribution (all clips)", fontweight="bold"); ax.set_ylabel("clips")
    ax.grid(axis="y", alpha=.25); ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def plot_mfcc_example(X, label, path):
    fig, ax = plt.subplots(figsize=(8, 3.2))
    im = ax.imshow(X[:, :40].T, origin="lower", aspect="auto", cmap="magma")
    ax.set_title(f"MFCC example ({label})", fontweight="bold"); ax.set_xlabel("frame"); ax.set_ylabel("MFCC coefficient")
    fig.colorbar(im, ax=ax); plt.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)
