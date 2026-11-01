"""Experiment 4: Experiment-3 pipeline with slightly stronger dropout (see ser/model_exp4.py).

Everything except the model is identical to run_experiment_03.py: same RAVDESS data, preprocessing, MFCC
features, actor-independent split, seed, augmentation, class weights and evaluation. All outputs go ONLY to
    models/experiment_04/   and   results/experiment_04/
so Experiment 1-3 artefacts are never touched. Existing Experiment 4 outputs are not overwritten unless
--overwrite is passed.

    python run_experiment_04.py --data data\\RAVDESS --epochs 60
    python run_experiment_04.py --synthetic --epochs 2 --actors 24 --per-emotion 1 --aug 1   # code-path smoke test
(synthetic runs write to results\\_smoke\\experiment_04 and models\\_smoke\\experiment_04 and are NOT results).
"""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from ser import data as D, features as F, evaluate as E, train_exp4 as T
from ser.config import LABELS, ROOT, Cfg
from ser.model_exp4 import EXP4_CONFIG

EXPERIMENT = "experiment_04"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "RAVDESS"))
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--synthetic", action="store_true", help="smoke test on generated audio (NOT real data)")
    ap.add_argument("--overwrite", action="store_true", help="allow overwriting existing experiment_04 outputs")
    ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--aug", type=int); ap.add_argument("--seed", type=int)
    ap.add_argument("--actors", type=int, default=24); ap.add_argument("--per-emotion", type=int, default=4)
    args = ap.parse_args()

    cfg = Cfg()
    cfg.train.epochs = args.epochs                       # Experiment 3 used a maximum of 60 epochs
    if args.aug is not None: cfg.train.aug_copies = args.aug
    if args.seed: cfg.train.seed = args.seed

    res = (ROOT / "results" / "_smoke" / EXPERIMENT) if args.synthetic else (ROOT / "results" / EXPERIMENT)
    mdl = (ROOT / "models" / "_smoke" / EXPERIMENT) if args.synthetic else (ROOT / "models" / EXPERIMENT)
    if (res / "metrics.json").exists() and not args.overwrite:
        raise SystemExit(f"{res / 'metrics.json'} already exists. Re-run with --overwrite to replace Experiment 4 outputs.")
    res.mkdir(parents=True, exist_ok=True); mdl.mkdir(parents=True, exist_ok=True)

    # 1 -- data
    if args.synthetic:
        data_dir = ROOT / "data" / "synthetic"
        D.make_synthetic_dataset(data_dir, n_actors=args.actors, per_emotion=args.per_emotion, sr=cfg.audio.sr)
        dataset_name, is_synth = "Synthetic smoke-test audio", True
    else:
        data_dir = D.download_ravdess(args.data) if args.download else Path(args.data)
        dataset_name, is_synth = "RAVDESS (speech)", False
    df = D.build_metadata(data_dir)
    if args.synthetic:
        a = sorted(df.actor.unique()); n = len(a)
        cfg.split.train_actors = tuple(a[: int(n * .66)]); cfg.split.val_actors = tuple(a[int(n * .66): int(n * .83)])
        cfg.split.test_actors = tuple(a[int(n * .83):])
    tr, va, te = D.split_by_actor(df, cfg)
    assert not (set(tr.actor) & set(va.actor)) and not (set(va.actor) & set(te.actor)), "actor overlap between splits"
    print(f"[data] {len(df)} clips | train {len(tr)} | val {len(va)} | test {len(te)}")

    # 2 -- features (augmentation: training only; normaliser: training statistics only)
    t0 = time.time()
    Xtr, ytr, _ = F.build_features(tr, cfg, cfg.train.aug_copies, cfg.train.seed, "train")
    Xva, yva, _ = F.build_features(va, cfg, 0, desc="val")
    Xte, yte, _ = F.build_features(te, cfg, 0, desc="test")
    feat_time = time.time() - t0
    mu, sd = F.fit_normaliser(Xtr)
    np.savez(mdl / "norm.npz", mu=mu, sd=sd)
    Xtr, Xva, Xte = (F.apply_normaliser(x, mu, sd) for x in (Xtr, Xva, Xte))
    print(f"[features] train {Xtr.shape} | val {Xva.shape} | test {Xte.shape}")

    # 3 -- SVM baseline (identical to previous experiments)
    svm, best = T.train_svm_baseline(Xtr, ytr, Xva, yva)
    base = E.compute_metrics(yte, svm.predict(F.summary_stats(Xte)), LABELS)
    print(f"[baseline] SVM {best}  macro-F1 = {base['macro_f1']:.4f}")

    # 4 -- deep model (Experiment 4 architecture)
    model, hist, fit_time = T.train_deep(Xtr, ytr, Xva, yva, cfg, str(mdl / "ser_model.keras"))
    deep = E.compute_metrics(yte, model.predict(Xte, verbose=0).argmax(1), LABELS)
    deep_val = E.compute_metrics(yva, model.predict(Xva, verbose=0).argmax(1), LABELS)
    best_epoch = int(np.argmax(hist["val_macro_f1"])) + 1
    print(f"[deep] val macro-F1 = {deep_val['macro_f1']:.4f} | test macro-F1 = {deep['macro_f1']:.4f} | acc = {deep['accuracy']:.4f}")

    # 5 -- reports & figures (experiment_04 only)
    E.plot_confusion(deep["confusion_matrix"], LABELS, res / "confusion_matrix.png", "Experiment 4: test confusion matrix")
    E.plot_confusion(base["confusion_matrix"], LABELS, res / "confusion_matrix_svm.png", "SVM baseline: test confusion matrix")
    E.plot_history(hist, res / "training_curves.png")
    E.plot_f1_per_class(deep, base, LABELS, res / "f1_per_class.png")
    E.plot_class_distribution({l: int((df.emotion == l).sum()) for l in LABELS}, res / "class_distribution.png")
    E.plot_mfcc_example(Xte[0], LABELS[yte[0]], res / "mfcc_example.png")
    (res / "classification_report_deep.txt").write_text(deep["report_text"])
    (res / "classification_report_svm.txt").write_text(base["report_text"])
    (res / "training_history.json").write_text(json.dumps({k: [float(x) for x in v] for k, v in hist.items()}, indent=2))

    out = dict(
        experiment=EXPERIMENT, dataset=dataset_name, synthetic=is_synth, n_clips=len(df),
        class_counts={l: int((df.emotion == l).sum()) for l in LABELS}, n_actors=int(df.actor.nunique()),
        config=dict(model="compact_cnn_bilstm_attention_exp4", **EXP4_CONFIG, lr=cfg.train.lr, batch_size=cfg.train.batch_size,
                    patience=cfg.train.patience, aug_copies=cfg.train.aug_copies, seed=cfg.train.seed),
        split=dict(train_clips=len(tr), val_clips=len(va), test_clips=len(te), train_actors=list(map(int, cfg.split.train_actors)),
                   val_actors=list(map(int, cfg.split.val_actors)), test_actors=list(map(int, cfg.split.test_actors)),
                   train_after_aug=int(len(Xtr))),
        features=dict(shape=list(Xtr.shape[1:]), sr=cfg.audio.sr, n_mfcc=cfg.audio.n_mfcc, hop=cfg.audio.hop_length,
                      duration=cfg.audio.duration, extraction_seconds=round(feat_time, 1)),
        training=dict(epochs_run=len(hist["loss"]), epochs_max=cfg.train.epochs, batch_size=cfg.train.batch_size,
                      seconds=round(fit_time, 1), params=int(model.count_params()), seed=cfg.train.seed,
                      best_epoch=best_epoch, best_val_macro_f1=float(max(hist["val_macro_f1"])),
                      best_val_loss=float(min(hist["val_loss"]))),
        svm=dict(params=best, **{k: v for k, v in base.items() if k != "report_text"}),
        deep=dict(**{k: v for k, v in deep.items() if k != "report_text"}),
        deep_val=dict(macro_f1=deep_val["macro_f1"], accuracy=deep_val["accuracy"]),
        history={k: [float(x) for x in v] for k, v in hist.items() if k in ("loss", "val_loss", "accuracy", "val_accuracy", "val_macro_f1")},
        environment=dict(python=platform.python_version(), platform=platform.platform()),
    )
    (res / "metrics.json").write_text(json.dumps(out, indent=2))
    (res / "experiment_summary.md").write_text(
        f"# Experiment 4 summary{' (SYNTHETIC SMOKE TEST - NOT A RESULT)' if is_synth else ''}\n\n"
        f"- Dataset: {dataset_name}\n- Parameters: {out['training']['params']:,}\n- Best epoch (val Macro F1): {best_epoch}\n"
        f"- Validation Macro F1: {deep_val['macro_f1']:.4f}\n- Test accuracy: {deep['accuracy']:.4f}\n"
        f"- Test Macro F1: {deep['macro_f1']:.4f}\n- Test weighted F1: {deep['weighted_f1']:.4f}\n"
        f"- Training time: {fit_time:.1f} s\n- Config: `{json.dumps(out['config'])}`\n")
    print("\n" + deep["report_text"]); print(f"[done] artefacts in {res} and {mdl}")


if __name__ == "__main__":
    main()
