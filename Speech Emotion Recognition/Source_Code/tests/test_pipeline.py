"""Fast unit tests (run:  pytest -q).  They use a tiny synthetic dataset, never real data."""
import numpy as np
import pytest

pytest.importorskip("tensorflow", reason="TensorFlow not installed in this Python (use Python 3.10-3.13)")

from ser import data as D, features as F
from ser.config import LABELS, Cfg
from ser.model import build_model


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    root = D.make_synthetic_dataset(tmp_path_factory.mktemp("syn"), n_actors=4, per_emotion=1)
    return D.build_metadata(root), Cfg()


def test_metadata_parsing(tiny):
    df, _ = tiny
    assert len(df) == 4 * 8 and set(df.emotion) == set(LABELS)


def test_mfcc_shape(tiny):
    df, cfg = tiny
    X = F.extract_mfcc(F.load_audio(df.path[0], cfg.audio), cfg.audio)
    assert X.shape == (cfg.audio.n_frames, cfg.audio.n_features)


def test_no_speaker_leakage(tiny):
    df, cfg = tiny
    cfg.split.train_actors, cfg.split.val_actors, cfg.split.test_actors = (1, 2), (3,), (4,)
    tr, va, te = D.split_by_actor(df, cfg)
    assert not (set(tr.actor) & set(te.actor)) and len(tr) + len(va) + len(te) == len(df)


def test_normaliser_uses_train_stats_only():
    rng = np.random.default_rng(0)
    X = rng.normal(5, 2, (10, 20, 4)).astype("float32")
    mu, sd = F.fit_normaliser(X)
    Z = F.apply_normaliser(X, mu, sd)
    assert abs(Z.mean()) < 1e-3 and abs(Z.std() - 1) < 1e-2


def test_model_forward_and_serialisation(tmp_path):
    import keras
    m = build_model((130, 120), 8)
    out = m.predict(np.zeros((2, 130, 120), "float32"), verbose=0)
    assert out.shape == (2, 8) and np.allclose(out.sum(1), 1, atol=1e-4)
    m.save(tmp_path / "m.keras")
    assert keras.saving.load_model(tmp_path / "m.keras").count_params() == m.count_params()
