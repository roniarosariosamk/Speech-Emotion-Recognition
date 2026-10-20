"""Pre-processing, MFCC feature extraction and waveform augmentation."""
from __future__ import annotations

import numpy as np
import librosa
import pandas as pd

from .config import LABEL2ID, AudioCfg, Cfg


# --------------------------------------------------------------------------- audio I/O
def fix_length(y: np.ndarray, n: int) -> np.ndarray:
    """Centre-crop or zero-pad to exactly n samples."""
    if len(y) >= n:
        s = (len(y) - n) // 2
        return y[s:s + n]
    return np.pad(y, (0, n - len(y)))


def load_audio(path: str, a: AudioCfg) -> np.ndarray:
    y, _ = librosa.load(path, sr=a.sr, mono=True)
    y, _ = librosa.effects.trim(y, top_db=a.top_db)       # remove leading / trailing silence
    y = y / (np.abs(y).max() + 1e-9)                       # peak normalisation
    return fix_length(y, a.n_samples).astype(np.float32)


# --------------------------------------------------------------------------- features
def extract_mfcc(y: np.ndarray, a: AudioCfg) -> np.ndarray:
    """Return (frames, 3*n_mfcc): MFCC, delta and delta-delta stacked along the feature axis."""
    m = librosa.feature.mfcc(y=y, sr=a.sr, n_mfcc=a.n_mfcc, n_fft=a.n_fft,
                             hop_length=a.hop_length, n_mels=a.n_mels)
    d1 = librosa.feature.delta(m, order=1)
    d2 = librosa.feature.delta(m, order=2)
    return np.concatenate([m, d1, d2], axis=0).T.astype(np.float32)


# --------------------------------------------------------------------------- augmentation
def augment(y: np.ndarray, a: AudioCfg, rng: np.random.Generator) -> np.ndarray:
    """Apply one randomly chosen waveform augmentation (training data only)."""
    kind = rng.choice(["noise", "pitch", "stretch", "shift"])
    if kind == "noise":
        y = y + rng.normal(0, rng.uniform(0.002, 0.01), len(y))
    elif kind == "pitch":
        y = librosa.effects.pitch_shift(y, sr=a.sr, n_steps=float(rng.uniform(-2, 2)))
    elif kind == "stretch":
        y = librosa.effects.time_stretch(y, rate=float(rng.uniform(0.9, 1.1)))
    else:
        y = np.roll(y, int(rng.uniform(-0.1, 0.1) * len(y)))
    return fix_length(y, a.n_samples).astype(np.float32)


# --------------------------------------------------------------------------- dataset assembly
def build_features(df: pd.DataFrame, cfg: Cfg, augment_copies: int = 0, seed: int = 0, desc: str = ""):
    """Extract (X, y, actors) for all clips; optionally append augmented copies."""
    a = cfg.audio
    rng = np.random.default_rng(seed)
    X, y, actors = [], [], []
    for i, r in enumerate(df.itertuples(index=False)):
        wav = load_audio(r.path, a)
        X.append(extract_mfcc(wav, a)); y.append(LABEL2ID[r.emotion]); actors.append(r.actor)
        for _ in range(augment_copies):
            X.append(extract_mfcc(augment(wav, a, rng), a)); y.append(LABEL2ID[r.emotion]); actors.append(r.actor)
        if (i + 1) % 100 == 0:
            print(f"[features] {desc} {i + 1}/{len(df)}", flush=True)
    return np.stack(X), np.array(y), np.array(actors)


def fit_normaliser(X: np.ndarray):
    """Per-coefficient mean/std computed on TRAIN only (prevents data leakage)."""
    return X.mean(axis=(0, 1), keepdims=True), X.std(axis=(0, 1), keepdims=True) + 1e-8


def apply_normaliser(X, mu, sd):
    return ((X - mu) / sd).astype(np.float32)


def summary_stats(X: np.ndarray) -> np.ndarray:
    """Utterance-level mean + std over time -> input for the classical SVM baseline."""
    return np.concatenate([X.mean(axis=1), X.std(axis=1)], axis=1)
