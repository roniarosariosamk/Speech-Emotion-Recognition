"""Predict speech emotion from an audio file.

Usage:
    python predict.py path/to/audio.wav
    python predict.py path/to/audio.wav --topk 5
    python predict.py --help
"""

import argparse

import numpy as np
import keras

from ser import model as _m  # Registers the custom AttentionPool layer
from ser.config import LABELS, ROOT, AudioCfg
from ser.features import apply_normaliser, extract_mfcc, load_audio


def predict(path: str, topk: int = 3):
    """Return the top predicted emotions and their probabilities."""

    if not 1 <= topk <= len(LABELS):
        raise ValueError(f"topk must be between 1 and {len(LABELS)}.")

    # Load the best trained model.
    model_path = ROOT / "models" / "ser_model_best.keras"
    norm_path = ROOT / "models" / "norm.npz"

    if not model_path.is_file():
        raise FileNotFoundError(f"Trained model not found: {model_path}")

    if not norm_path.is_file():
        raise FileNotFoundError(f"Normalization file not found: {norm_path}")

    audio_config = AudioCfg()

    model = keras.saving.load_model(model_path, compile=False)

    with np.load(norm_path) as normaliser:
        mu = normaliser["mu"]
        sd = normaliser["sd"]

    audio = load_audio(path, audio_config)
    features = extract_mfcc(audio, audio_config)

    features = apply_normaliser(
        features[None, ...],
        mu,
        sd,
    )

    probabilities = model.predict(features, verbose=0)[0]

    # Return the highest-probability emotions first.
    top_indices = np.argsort(probabilities)[::-1][:topk]

    return [
        (LABELS[i], float(probabilities[i]))
        for i in top_indices
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict emotion from a speech audio file."
    )

    parser.add_argument(
        "audio",
        help="Path to an audio file, such as a .wav file.",
    )

    parser.add_argument(
        "--topk",
        type=int,
        default=3,
        help="Number of top emotion predictions to display (default: 3).",
    )

    args = parser.parse_args()

    try:
        predictions = predict(args.audio, args.topk)

        print("\nSpeech Emotion Recognition Results")
        print("-" * 38)

        for emotion, probability in predictions:
            print(f"{emotion:12s} {probability:7.2%}")

    except (FileNotFoundError, ValueError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")
