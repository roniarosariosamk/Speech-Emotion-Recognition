"""Project-level validation: config, labels, features, models (Exp 3 + Exp 4), artefacts, inference, metrics, outputs.

Run from Source_Code:   python -m pytest -q
Tests that need trained artefacts / real results skip themselves if those files are absent.
Synthetic audio is used ONLY to exercise code paths; it never produces reportable results.
"""
import ast
import json
from pathlib import Path

import numpy as np
import pytest

from ser import data as D, evaluate as E, features as F, final
from ser.config import EMOTIONS, LABELS, LABEL2ID, ROOT, AudioCfg, Cfg

N_FRAMES, N_FEAT = 130, 120


@pytest.fixture(scope="module")
def syn(tmp_path_factory):
    root = D.make_synthetic_dataset(tmp_path_factory.mktemp("syn"), n_actors=4, per_emotion=1)
    return root, D.build_metadata(root)


# ---------------------------------------------------------------- config / labels / data
def test_config_values():
    c = Cfg()
    assert (c.audio.sr, c.audio.duration, c.audio.n_mfcc, c.audio.n_fft, c.audio.hop_length, c.audio.n_mels) == (22050, 3.0, 40, 2048, 512, 128)
    assert (c.audio.n_frames, c.audio.n_features) == (N_FRAMES, N_FEAT)
    assert c.split.train_actors == tuple(range(1, 17)) and c.split.val_actors == tuple(range(17, 21)) and c.split.test_actors == tuple(range(21, 25))
    assert c.train.seed == 42


def test_label_mapping_is_fixed_and_consistent():
    assert LABELS == ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
    assert [LABEL2ID[l] for l in LABELS] == list(range(8))
    assert EMOTIONS["03"] == "happy" and EMOTIONS["08"] == "surprised"


def test_dataset_discovery_and_filename_parsing(syn):
    root, df = syn
    assert len(df) == 32 and set(df.emotion) == set(LABELS) and set(df.actor) == {1, 2, 3, 4}
    row = D.parse_ravdess_name(Path("03-01-05-02-01-01-07.wav"))
    assert row["emotion"] == "angry" and row["actor"] == 7 and row["gender"] == "male"
    with pytest.raises(FileNotFoundError):
        D.build_metadata(root / "does_not_exist")


def test_real_split_sizes_from_saved_metrics():
    f = ROOT / "results" / "experiment_03" / "metrics.json"
    if not f.is_file(): pytest.skip("experiment_03 metrics not present")
    m = json.loads(f.read_text()); s = m["split"]
    assert (m["n_clips"], m["n_actors"]) == (1440, 24) and (s["train_clips"], s["val_clips"], s["test_clips"]) == (960, 240, 240)
    assert not (set(s["train_actors"]) & set(s["val_actors"])) and not (set(s["train_actors"]) & set(s["test_actors"])) and not (set(s["val_actors"]) & set(s["test_actors"]))
    assert s["train_after_aug"] == 2880 and m["features"]["shape"] == [N_FRAMES, N_FEAT]


# ---------------------------------------------------------------- features
def test_feature_shape_and_finite(syn):
    _, df = syn; cfg = Cfg()
    X = F.extract_mfcc(F.load_audio(df.path[0], cfg.audio), cfg.audio)
    assert X.shape == (N_FRAMES, N_FEAT) and X.dtype == np.float32 and np.isfinite(X).all()


def test_augmentation_only_when_requested(syn):
    _, df = syn; cfg = Cfg(); sub = df.head(4)
    X0, y0, _ = F.build_features(sub, cfg, 0)
    X2, y2, _ = F.build_features(sub, cfg, 2, seed=1)
    assert len(X0) == 4 and len(X2) == 12 and set(y2) <= set(y0)


def test_normaliser_fit_on_train_only_and_reused():
    rng = np.random.default_rng(0)
    tr, te = rng.normal(5, 2, (20, 10, 4)).astype("float32"), rng.normal(9, 2, (5, 10, 4)).astype("float32")
    mu, sd = F.fit_normaliser(tr)
    assert np.allclose(mu.ravel(), tr.mean((0, 1)), atol=1e-5)
    assert F.apply_normaliser(te, mu, sd).mean() > 1.0       # test data is NOT re-centred -> no leakage


def test_unreadable_and_silent_audio_raise_clear_errors(tmp_path):
    import soundfile as sf
    bad = tmp_path / "bad.wav"; bad.write_bytes(b"not audio")
    with pytest.raises(ValueError, match="Could not decode"): F.load_audio(str(bad), AudioCfg())
    sil = tmp_path / "silent.wav"; sf.write(sil, np.zeros(22050, dtype="float32"), 22050)
    with pytest.raises(ValueError, match="silent"): F.load_audio(str(sil), AudioCfg())


# ---------------------------------------------------------------- models
def _models():
    from ser.model import build_model as b3
    from ser.model_exp4 import build_model as b4
    return {"exp3": b3, "exp4": b4}


@pytest.mark.parametrize("which", ["exp3", "exp4"])
def test_model_outputs_eight_finite_probabilities(which):
    m = _models()[which]((N_FRAMES, N_FEAT), 8)
    p = m.predict(np.random.default_rng(0).normal(size=(3, N_FRAMES, N_FEAT)).astype("float32"), verbose=0)
    assert p.shape == (3, 8) and np.isfinite(p).all() and np.allclose(p.sum(1), 1, atol=1e-4)
    assert m.count_params() == 79080                          # same size: only dropout differs


@pytest.mark.parametrize("which", ["exp3", "exp4"])
def test_model_save_load_roundtrip(which, tmp_path):
    import keras
    from ser import model as _  # noqa: F401  registers AttentionPool
    m = _models()[which]((N_FRAMES, N_FEAT), 8); x = np.random.default_rng(1).normal(size=(2, N_FRAMES, N_FEAT)).astype("float32")
    m.save(tmp_path / "m.keras"); m2 = keras.saving.load_model(tmp_path / "m.keras", compile=False)
    assert np.allclose(m.predict(x, verbose=0), m2.predict(x, verbose=0), atol=1e-5)


def test_experiment4_dropout_differs_from_experiment3_and_is_isolated():
    import keras
    def rates(m): return [round(l.rate, 2) for l in m.layers if isinstance(l, keras.layers.Dropout)] + [m.layers[[type(l).__name__ for l in m.layers].index("Bidirectional")].forward_layer.dropout]
    b = _models()
    assert rates(b["exp3"]((N_FRAMES, N_FEAT), 8)) == [0.30, 0.35, 0.50, 0.30]
    assert rates(b["exp4"]((N_FRAMES, N_FEAT), 8)) == [0.35, 0.40, 0.50, 0.35]
    src = (ROOT / "ser" / "train_exp4.py").read_text()
    imports = [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.ImportFrom) and n.module]
    assert any(n.module == "model_exp4" and n.level == 1 for n in imports), "train_exp4 must import ser.model_exp4"
    assert not any(n.module == "model" and n.level == 1 for n in imports), "train_exp4 must NOT import ser.model"
    run = (ROOT / "run_experiment_04.py").read_text()
    literals = {n.value for n in ast.walk(ast.parse(run)) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert {"experiment_01", "experiment_02", "experiment_03"}.isdisjoint(literals), "Experiment 4 runner must not reference other experiments' folders"
    assert "experiment_04" in literals


# ---------------------------------------------------------------- saved artefacts + inference
ART = ROOT / "models" / "experiment_03"


@pytest.mark.skipif(not (ART / "ser_model_best.keras").is_file(), reason="trained model not present")
def test_final_model_and_normaliser_match_features():
    import predict
    model, mu, sd = predict.load_artefacts("experiment_03")
    assert model.input_shape == (None, N_FRAMES, N_FEAT) and model.output_shape == (None, 8) and model.count_params() == 79080
    assert mu.shape == sd.shape == (1, 1, N_FEAT) and (sd > 0).all()


@pytest.mark.skipif(not (ART / "ser_model_best.keras").is_file(), reason="trained model not present")
def test_prediction_technical_smoke_on_synthetic_clip(syn):
    """TECHNICAL SMOKE TEST ONLY: synthetic audio, not an evaluation of accuracy."""
    import predict
    _, df = syn
    p = predict.predict_proba(df.path[0], "experiment_03")
    assert p.shape == (8,) and np.isfinite(p).all() and abs(p.sum() - 1) < 1e-3
    top = predict.predict(df.path[0], 3, "experiment_03")
    assert len(top) == 3 and top[0][0] in LABELS and top[0][1] >= top[1][1] >= top[2][1]


def test_prediction_errors_are_clear(tmp_path):
    import predict
    with pytest.raises(FileNotFoundError): predict.predict_proba(tmp_path / "missing.wav", "experiment_03")
    t = tmp_path / "a.txt"; t.write_text("x")
    with pytest.raises(ValueError, match="Unsupported"): predict.predict_proba(t, "experiment_03")
    with pytest.raises(ValueError, match="Unknown experiment"): predict.load_artefacts("nope")
    with pytest.raises(ValueError): predict.predict(t, topk=99)


# ---------------------------------------------------------------- metrics
def test_metrics_match_hand_computation():
    y = np.array([0, 0, 1, 1, 2, 2, 3, 3]); p = np.array([0, 1, 1, 1, 2, 0, 3, 3])
    m = E.compute_metrics(y, p, LABELS)          # all 8 labels explicitly; classes 4-7 have zero support
    assert m["accuracy"] == pytest.approx(6 / 8)
    d = final.derive(m["confusion_matrix"])
    assert d["macro_f1"] == pytest.approx(m["macro_f1"]) and d["weighted_f1"] == pytest.approx(m["weighted_f1"])
    assert len(m["per_class"]) == 8 and np.array(m["confusion_matrix"]).sum() == 8


def test_saved_metrics_are_internally_consistent_and_model_selection_is_valid():
    S = final.build_summary()
    assert all(r.get("integrity_ok", True) for r in S["experiments"])
    sel = S["selected_final_model"]; done = [r for r in S["experiments"] if r["status"] == "completed"]
    assert sel["validation_macro_f1"] == max(r["val_macro_f1"] for r in done)
    assert sel["test"]["macro_f1"] > S["svm_baseline"]["test_macro_f1"]


# ---------------------------------------------------------------- outputs / isolation
def test_output_directories_are_created(tmp_path):
    d = tmp_path / "results" / "experiment_x"; d.mkdir(parents=True, exist_ok=True)
    E.plot_confusion(np.eye(8, dtype=int) * 3, LABELS, d / "cm.png", "t")
    assert (d / "cm.png").stat().st_size > 1000


def test_experiment_artefacts_are_kept_separate():
    for exp in ("experiment_02", "experiment_03"):
        assert (ROOT / "models" / exp / "ser_model_best.keras").is_file() and (ROOT / "results" / exp / "metrics.json").is_file()
    assert not (ROOT / "results" / "experiment_04" / "metrics.json").exists() or json.loads((ROOT / "results" / "experiment_04" / "metrics.json").read_text())["experiment"] == "experiment_04"
