"""Predict the emotion of one audio file:  python predict.py path/to/clip.wav"""
import sys

import numpy as np
import keras

from ser import model as _m  # noqa: F401  (registers AttentionPool)
from ser.config import LABELS, ROOT, AudioCfg
from ser.features import apply_normaliser, extract_mfcc, load_audio


def predict(path: str, topk: int = 3):
    a = AudioCfg()
    model = keras.saving.load_model(ROOT / "models" / "ser_model.keras")
    n = np.load(ROOT / "models" / "norm.npz")
    x = apply_normaliser(extract_mfcc(load_audio(path, a), a)[None], n["mu"], n["sd"])
    p = model.predict(x, verbose=0)[0]
    return [(LABELS[i], float(p[i])) for i in np.argsort(p)[::-1][:topk]]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for lab, pr in predict(sys.argv[1]):
        print(f"{lab:10s} {pr:6.1%}")
