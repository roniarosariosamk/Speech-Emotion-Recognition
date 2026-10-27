r"""Predict the emotion of a speech clip with a trained model.

Usage (PowerShell, from Source_Code):
    python predict.py path\to\clip.wav                      # uses the selected final model
    python predict.py path\to\clip.wav --experiment experiment_03
    python predict.py path\to\clip.wav --all --json         # all 8 probabilities as JSON

The model, the matching normalisation file (training mean / std) and the audio / MFCC settings are the same
ones used for training: no new normaliser is ever fitted on the input clip.
"""
from __future__ import annotations

import argparse
import json
import os
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

try:
    import keras
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("TensorFlow/Keras is required. Install it with:  pip install -r requirements.txt") from exc

from ser import model as _m  # noqa: F401  (registers the custom AttentionPool layer)
from ser.config import LABELS, ROOT, AudioCfg
from ser.features import apply_normaliser, extract_mfcc, load_audio

SUPPORTED_EXT = {".wav", ".flac", ".ogg", ".mp3"}
FALLBACK_EXPERIMENT = "experiment_03"          # best validated model (Macro F1 selection)

#            experiment key -> (model checkpoint, matching normalisation file)
ARTEFACTS = {
    "experiment_01": (ROOT / "models" / "ser_model_best.keras", ROOT / "models" / "norm.npz"),
    "experiment_02": (ROOT / "models" / "experiment_02" / "ser_model_best.keras", ROOT / "models" / "experiment_02" / "norm.npz"),
    "experiment_03": (ROOT / "models" / "experiment_03" / "ser_model_best.keras", ROOT / "models" / "experiment_03" / "norm.npz"),
    "experiment_04": (ROOT / "models" / "experiment_04" / "ser_model_best.keras", ROOT / "models" / "experiment_04" / "norm.npz"),
}


def default_experiment() -> str:
    """The final model chosen by summarize_results.py (falls back to Experiment 3)."""
    f = ROOT / "results" / "final" / "final_summary.json"
    try:
        sel = json.loads(f.read_text())["selected_final_model"]["experiment"]
        return sel if sel in ARTEFACTS else FALLBACK_EXPERIMENT
    except (OSError, KeyError, ValueError):
        return FALLBACK_EXPERIMENT


@lru_cache(maxsize=4)
def load_artefacts(experiment: str):
    if experiment not in ARTEFACTS:
        raise ValueError(f"Unknown experiment '{experiment}'. Choose from: {', '.join(ARTEFACTS)}")
    model_path, norm_path = ARTEFACTS[experiment]
    if not model_path.is_file():
        raise FileNotFoundError(f"Trained model not found: {model_path}")
    if not norm_path.is_file():
        raise FileNotFoundError(f"Normalisation file not found: {norm_path}")
    model = keras.saving.load_model(model_path, compile=False)
    with np.load(norm_path) as z:
        mu, sd = z["mu"], z["sd"]
    cfg = AudioCfg()
    if tuple(model.input_shape[1:]) != (cfg.n_frames, cfg.n_features) or mu.shape[-1] != cfg.n_features:
        raise RuntimeError(f"Model/normaliser do not match the feature shape ({cfg.n_frames}, {cfg.n_features}).")
    return model, mu, sd


def predict_proba(path: str | Path, experiment: str | None = None) -> np.ndarray:
    """Return a (8,) probability vector in LABELS order."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Not a file: {path}")
    if path.suffix.lower() not in SUPPORTED_EXT:
        raise ValueError(f"Unsupported audio type '{path.suffix}'. Supported: {', '.join(sorted(SUPPORTED_EXT))}")
    model, mu, sd = load_artefacts(experiment or default_experiment())
    cfg = AudioCfg()
    x = apply_normaliser(extract_mfcc(load_audio(str(path), cfg), cfg)[None, ...], mu, sd)   # training statistics
    p = model.predict(x, verbose=0)[0]
    if p.shape != (len(LABELS),) or not np.all(np.isfinite(p)) or abs(float(p.sum()) - 1.0) > 1e-3:
        raise RuntimeError("Model produced invalid probabilities.")
    return p.astype(float)


def predict(path: str | Path, topk: int = 3, experiment: str | None = None):
    """Return the top-k (emotion, probability) pairs, most likely first."""
    if not 1 <= topk <= len(LABELS):
        raise ValueError(f"topk must be between 1 and {len(LABELS)}.")
    p = predict_proba(path, experiment)
    return [(LABELS[i], float(p[i])) for i in np.argsort(p)[::-1][:topk]]


def main():
    ap = argparse.ArgumentParser(description="Predict emotion from a speech audio file.")
    ap.add_argument("audio", help="path to a .wav / .flac / .ogg / .mp3 file")
    ap.add_argument("--experiment", choices=list(ARTEFACTS), help="model to use (default: selected final model)")
    ap.add_argument("--topk", type=int, default=3, help="number of top emotions to show (default 3)")
    ap.add_argument("--all", action="store_true", help="show all 8 emotion probabilities")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    a = ap.parse_args()
    exp = a.experiment or default_experiment()
    try:
        p = predict_proba(a.audio, exp)
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as e:
        ap.exit(1, f"Error: {e}\n")
    order = np.argsort(p)[::-1]
    shown = order if a.all else order[: max(1, min(a.topk, len(LABELS)))]
    if a.json:
        print(json.dumps({"file": str(a.audio), "experiment": exp, "predicted": LABELS[order[0]],
                          "probabilities": {LABELS[i]: float(p[i]) for i in order}}, indent=2))
        return
    print(f"\nSpeech Emotion Recognition  [{exp}]\n" + "-" * 44)
    print(f"Predicted emotion : {LABELS[order[0]].upper()}  ({p[order[0]]:.1%})\n")
    for i in shown:
        print(f"  {LABELS[i]:10s} {p[i]:7.2%}  {'#' * int(round(p[i] * 30))}")


if __name__ == "__main__":
    main()
