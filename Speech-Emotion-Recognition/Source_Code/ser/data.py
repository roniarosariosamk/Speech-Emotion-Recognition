"""Data collection: download RAVDESS, parse metadata, create speaker-independent splits.

A synthetic generator is included ONLY for smoke-testing the pipeline offline / in CI.
Results obtained on synthetic audio must never be reported as real-world performance.
"""
from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from .config import EMOTIONS, LABELS, RAVDESS_URL, Cfg


# --------------------------------------------------------------------------- download
def download_ravdess(dest: str | Path, url: str = RAVDESS_URL) -> Path:
    """Download + unzip the RAVDESS speech-audio archive (~215 MB, CC BY-NC-SA 4.0)."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    if any(dest.rglob("*.wav")):
        print(f"[data] RAVDESS already present in {dest}")
        return dest
    zpath = dest / "Audio_Speech_Actors_01-24.zip"
    if not zpath.exists():
        print(f"[data] downloading {url}")

        def hook(blocks, bs, total):
            if total > 0 and blocks % 200 == 0:
                print(f"\r[data] {min(100, blocks * bs * 100 // total)}%", end="", flush=True)

        urllib.request.urlretrieve(url, zpath, hook)
        print()
    with zipfile.ZipFile(zpath) as zf:
        zf.extractall(dest)
    print(f"[data] extracted to {dest}")
    return dest


# --------------------------------------------------------------------------- metadata
def parse_ravdess_name(path: Path) -> dict:
    """modality-channel-emotion-intensity-statement-repetition-actor.wav"""
    m, ch, emo, inten, stmt, rep, actor = path.stem.split("-")
    return dict(path=str(path), emotion=EMOTIONS[emo], intensity=int(inten),
                statement=int(stmt), repetition=int(rep), actor=int(actor),
                gender="female" if int(actor) % 2 == 0 else "male")


def build_metadata(root: str | Path) -> pd.DataFrame:
    files = sorted(Path(root).rglob("*.wav"))
    if not files:
        raise FileNotFoundError(f"No .wav files under {root}. Run with --download or --synthetic.")
    df = pd.DataFrame([parse_ravdess_name(f) for f in files])
    return df[df.emotion.isin(LABELS)].reset_index(drop=True)


def split_by_actor(df: pd.DataFrame, cfg: Cfg):
    s = cfg.split
    tr = df[df.actor.isin(s.train_actors)]
    va = df[df.actor.isin(s.val_actors)]
    te = df[df.actor.isin(s.test_actors)]
    assert not (set(tr.actor) & set(te.actor)) and not (set(tr.actor) & set(va.actor)), "speaker leakage!"
    return tr.reset_index(drop=True), va.reset_index(drop=True), te.reset_index(drop=True)


# --------------------------------------------------------------------------- synthetic (smoke test only)
_SYN = {  # f0 (Hz), f0 jitter, syllable rate (Hz), loudness, brightness (harmonic decay), breathiness
    "neutral":   (120, 0.03, 3.5, 0.50, 1.6, 0.02),
    "calm":      (110, 0.02, 2.6, 0.40, 1.9, 0.02),
    "happy":     (190, 0.12, 4.6, 0.70, 1.2, 0.03),
    "sad":       (95,  0.02, 2.2, 0.30, 2.0, 0.05),
    "angry":     (160, 0.10, 5.0, 0.95, 0.8, 0.06),
    "fearful":   (210, 0.15, 5.4, 0.55, 1.1, 0.07),
    "disgust":   (105, 0.05, 3.0, 0.65, 1.4, 0.08),
    "surprised": (240, 0.18, 4.2, 0.75, 1.0, 0.03),
}


def make_synthetic_dataset(root: str | Path, n_actors: int = 24, per_emotion: int = 4,
                           sr: int = 22_050, seed: int = 0) -> Path:
    """Write RAVDESS-style .wav files with emotion-dependent prosody (f0, tempo, energy, timbre)."""
    rng = np.random.default_rng(seed)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    codes = {v: k for k, v in EMOTIONS.items()}
    for actor in range(1, n_actors + 1):
        base_shift = (1.0 if actor % 2 else 1.7) * rng.uniform(0.85, 1.15)   # male / female pitch
        (root / f"Actor_{actor:02d}").mkdir(exist_ok=True)
        for emo, (f0, jit, rate, loud, decay, breath) in _SYN.items():
            for k in range(per_emotion):
                dur = rng.uniform(3.0, 3.8)
                t = np.arange(int(dur * sr)) / sr
                f = f0 * base_shift * rng.uniform(0.8, 1.2)
                contour = 1 + jit * np.sin(2 * np.pi * rng.uniform(0.5, 2.5) * t + rng.uniform(0, 6.28))
                contour += rng.normal(0, jit / 4, size=len(t)).cumsum() / sr * 50
                phase = 2 * np.pi * np.cumsum(f * contour) / sr
                sig = sum(np.sin(h * phase) / (h ** (decay * rng.uniform(0.8, 1.25))) for h in range(1, 12))
                env = 0.5 * (1 + np.sin(2 * np.pi * rate * rng.uniform(0.85, 1.15) * t)) ** 1.5
                sig = sig * env * loud * rng.uniform(0.7, 1.2) + rng.normal(0, breath + 0.01, len(t))
                sig = np.concatenate([np.zeros(int(sr * 0.2)), sig, np.zeros(int(sr * 0.2))])
                sig = (0.8 * sig / (np.abs(sig).max() + 1e-9)).astype(np.float32)
                name = f"03-01-{codes[emo]}-01-0{1 + k % 2}-0{1 + k // 2 % 2}-{actor:02d}.wav"
                sf.write(root / f"Actor_{actor:02d}" / name, sig, sr)
    return root
