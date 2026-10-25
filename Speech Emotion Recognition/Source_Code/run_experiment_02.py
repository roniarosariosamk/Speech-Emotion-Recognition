"""End-to-end pipeline:  data -> features -> SVM baseline + deep model -> F1 evaluation -> results/.

Examples
    python run_pipeline.py --download --data data/RAVDESS          # real run (recommended)
    python run_pipeline.py --data data/RAVDESS --epochs 80
    python run_pipeline.py --synthetic                              # offline smoke test only
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from ser import data as D, features as F, evaluate as E, train as T
from ser.config import LABELS, ROOT, Cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "RAVDESS"))
    ap.add_argument("--download", action="store_true", help="download RAVDESS from Zenodo")
    ap.add_argument("--synthetic", action="store_true", help="smoke test on generated audio (NOT real data)")
    ap.add_argument("--epochs", type=int); ap.add_argument("--aug", type=int); ap.add_argument("--seed", type=int)
    ap.add_argument("--actors", type=int, default=24, help="synthetic only")
    ap.add_argument("--per-emotion", type=int, default=4, help="synthetic only")
    args = ap.parse_args()

    cfg = Cfg()
    if args.epochs: cfg.train.epochs = args.epochs
    if args.aug is not None: cfg.train.aug_copies = args.aug
    if args.seed: cfg.train.seed = args.seed
    res = ROOT / "results" / "experiment_02"
    mdl = ROOT / "models" / "experiment_02"

    res.mkdir(parents=True, exist_ok=True)
    mdl.mkdir(parents=True, exist_ok=True)

    # 1 ── data collection
    if args.synthetic:
        data_dir = ROOT / "data" / "synthetic"
        D.make_synthetic_dataset(data_dir, n_actors=args.actors, per_emotion=args.per_emotion, sr=cfg.audio.sr)
        dataset_name, is_synth = "Synthetic smoke-test audio", True
    else:
        data_dir = D.download_ravdess(args.data) if args.download else Path(args.data)
        dataset_name, is_synth = "RAVDESS (speech)", False
    df = D.build_metadata(data_dir)
    if args.synthetic:   # the synthetic set may have fewer actors -> rescale split proportionally
        a = sorted(df.actor.unique()); n = len(a)
        cfg.split.train_actors = tuple(a[: int(n * .66)]); cfg.split.val_actors = tuple(a[int(n * .66): int(n * .83)])
        cfg.split.test_actors = tuple(a[int(n * .83):])
    tr, va, te = D.split_by_actor(df, cfg)
    print(f"[data] {len(df)} clips | train {len(tr)} | val {len(va)} | test {len(te)}")

    # 2 ── pre-processing + MFCC
    t0 = time.time()
    Xtr, ytr, _ = F.build_features(tr, cfg, cfg.train.aug_copies, cfg.train.seed, "train")
    Xva, yva, _ = F.build_features(va, cfg, 0, desc="val")
    Xte, yte, _ = F.build_features(te, cfg, 0, desc="test")
    feat_time = time.time() - t0
    mu, sd = F.fit_normaliser(Xtr)
    np.savez(mdl / "norm.npz", mu=mu, sd=sd)
    Xtr, Xva, Xte = (F.apply_normaliser(x, mu, sd) for x in (Xtr, Xva, Xte))
    print(f"[features] train {Xtr.shape} (incl. {cfg.train.aug_copies} aug copies) | val {Xva.shape} | test {Xte.shape}")

    # 3 ── baseline
    svm, best = T.train_svm_baseline(Xtr, ytr, Xva, yva)
    base_pred = svm.predict(F.summary_stats(Xte))
    base = E.compute_metrics(yte, base_pred, LABELS)
    print(f"[baseline] SVM {best}  macro-F1 = {base['macro_f1']:.4f}")

    # 4 ── deep model
    model, hist, fit_time = T.train_deep(Xtr, ytr, Xva, yva, cfg, str(mdl / "ser_model.keras"))
    deep_pred = model.predict(Xte, verbose=0).argmax(1)
    deep = E.compute_metrics(yte, deep_pred, LABELS)
    deep_val = E.compute_metrics(yva, model.predict(Xva, verbose=0).argmax(1), LABELS)
    print(f"[deep] test macro-F1 = {deep['macro_f1']:.4f} | acc = {deep['accuracy']:.4f}")

    # 5 ── reports & figures
    E.plot_confusion(deep["confusion_matrix"], LABELS, res / "confusion_matrix.png", "CNN-BiLSTM-Attention: test confusion matrix")
    E.plot_confusion(base["confusion_matrix"], LABELS, res / "confusion_matrix_svm.png", "SVM baseline: test confusion matrix")
    E.plot_history(hist, res / "training_curves.png")
    E.plot_f1_per_class(deep, base, LABELS, res / "f1_per_class.png")
    counts = {l: int((df.emotion == l).sum()) for l in LABELS}
    E.plot_class_distribution(counts, res / "class_distribution.png")
    E.plot_mfcc_example(Xte[0], LABELS[yte[0]], res / "mfcc_example.png")
    (res / "classification_report_deep.txt").write_text(deep["report_text"])
    (res / "classification_report_svm.txt").write_text(base["report_text"])

    out = dict(
        dataset=dataset_name, synthetic=is_synth, n_clips=len(df), class_counts=counts,
        n_actors=int(df.actor.nunique()),
        split=dict(train_clips=len(tr), val_clips=len(va), test_clips=len(te),
                   train_actors=list(map(int, cfg.split.train_actors)), val_actors=list(map(int, cfg.split.val_actors)),
                   test_actors=list(map(int, cfg.split.test_actors)), train_after_aug=int(len(Xtr))),
        features=dict(shape=list(Xtr.shape[1:]), sr=cfg.audio.sr, n_mfcc=cfg.audio.n_mfcc, hop=cfg.audio.hop_length,
                      duration=cfg.audio.duration, extraction_seconds=round(feat_time, 1)),
        training=dict(epochs_run=len(hist["loss"]), epochs_max=cfg.train.epochs, batch_size=cfg.train.batch_size,
                      seconds=round(fit_time, 1), params=int(model.count_params()), seed=cfg.train.seed,
                      best_val_loss=float(min(hist["val_loss"]))),
        svm=dict(params=best, **{k: v for k, v in base.items() if k != "report_text"}),
        deep=dict(**{k: v for k, v in deep.items() if k != "report_text"}),
        deep_val=dict(macro_f1=deep_val["macro_f1"], accuracy=deep_val["accuracy"]),
        history={k: [float(x) for x in v] for k, v in hist.items() if k in ("loss", "val_loss", "accuracy", "val_accuracy")},
    )
    (res / "metrics.json").write_text(json.dumps(out, indent=2))
    print("\n" + deep["report_text"])
    print(f"[done] artefacts in {res}")


if __name__ == "__main__":
    main()
