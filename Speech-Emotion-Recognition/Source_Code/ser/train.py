"""Training: classical SVM baseline + deep CNN-BiLSTM-Attention model."""
from __future__ import annotations

import time

import numpy as np

import keras

from sklearn.model_selection import GridSearchCV, PredefinedSplit  # pyright: ignore[reportMissingModuleSource]
from sklearn.pipeline import make_pipeline  # pyright: ignore[reportMissingModuleSource]
from sklearn.preprocessing import StandardScaler  # pyright: ignore[reportMissingModuleSource]
from sklearn.svm import SVC  # pyright: ignore[reportMissingModuleSource]
from sklearn.utils.class_weight import compute_class_weight  # pyright: ignore[reportMissingModuleSource]

from .features import summary_stats
from .model import build_model


def set_seed(seed: int):
    keras.utils.set_random_seed(seed)


def train_svm_baseline(Xtr, ytr, Xva, yva):
    """RBF-SVM on utterance-level MFCC statistics; C/gamma tuned on the validation actors."""
    Xs = np.concatenate([summary_stats(Xtr), summary_stats(Xva)])
    ys = np.concatenate([ytr, yva])
    fold = np.r_[-np.ones(len(ytr)), np.zeros(len(yva))]           # -1 = always train, 0 = validation
    pipe = make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced"))
    grid = {"svc__C": [1, 10, 100], "svc__gamma": ["scale", 1e-3, 1e-2]}
    gs = GridSearchCV(pipe, grid, cv=PredefinedSplit(fold), scoring="f1_macro", refit=False)
    gs.fit(Xs, ys)
    best = make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced",
                                               C=gs.best_params_["svc__C"], gamma=gs.best_params_["svc__gamma"]))
    best.fit(summary_stats(Xtr), ytr)
    return best, {k.split("__")[1]: v for k, v in gs.best_params_.items()}

def train_deep(Xtr, ytr, Xva, yva, cfg, model_path: str):
    """Train the deep model and select its checkpoint using validation Macro F1."""
    import os
    from sklearn.metrics import f1_score  # pyright: ignore[reportMissingModuleSource]

    t = cfg.train
    set_seed(t.seed)

    n_classes = int(max(ytr.max(), yva.max())) + 1

    model = build_model(
        Xtr.shape[1:],
        n_classes,
        lr=t.lr,
    )

    cw = compute_class_weight(
        "balanced",
        classes=np.unique(ytr),
        y=ytr,
    )

    class_weights = {
        int(cls): float(weight)
        for cls, weight in zip(np.unique(ytr), cw)
    }

    best_checkpoint = model_path.replace(
        "ser_model.keras",
        "ser_model_best.keras",
    )

    best_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    patience = t.patience
    lr_patience = max(3, patience // 3)
    lr_wait = 0
    best_val_loss = float("inf")

    history = {
        "loss": [],
        "accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_macro_f1": [],
    }

    t0 = time.time()

    for epoch in range(t.epochs):
        print(f"\nEpoch {epoch + 1}/{t.epochs}")

        epoch_history = model.fit(
            Xtr,
            ytr,
            validation_data=(Xva, yva),
            initial_epoch=epoch,
            epochs=epoch + 1,
            batch_size=t.batch_size,
            class_weight=class_weights,
            verbose=2,
        )

        for key in ("loss", "accuracy", "val_loss", "val_accuracy"):
            history[key].append(epoch_history.history[key][0])

        val_probabilities = model.predict(Xva, verbose=0)
        val_predictions = np.argmax(val_probabilities, axis=1)

        val_macro_f1 = f1_score(
            yva,
            val_predictions,
            labels=np.arange(n_classes),
            average="macro",
            zero_division=0,
        )

        history["val_macro_f1"].append(float(val_macro_f1))

        print(f"Validation Macro F1: {val_macro_f1:.4f}")

        if val_macro_f1 > best_f1:
            best_f1 = float(val_macro_f1)
            best_epoch = epoch + 1
            epochs_without_improvement = 0

            model.save(best_checkpoint)

            print(
                f"Saved best checkpoint: epoch {best_epoch}, "
                f"validation Macro F1={best_f1:.4f}"
            )
        else:
            epochs_without_improvement += 1

        current_val_loss = history["val_loss"][-1]

        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            lr_wait = 0
        else:
            lr_wait += 1

        if lr_wait >= lr_patience:
            old_lr = float(
                keras.ops.convert_to_numpy(model.optimizer.learning_rate
                )
            )
            new_lr = max(old_lr * 0.5, 1e-5)

            if new_lr < old_lr:
                model.optimizer.learning_rate.assign(new_lr)


                print(
                    f"Learning rate reduced: "
                    f"{old_lr:.7f} -> {new_lr:.7f}"
                )

            lr_wait = 0

        if epochs_without_improvement >= patience:
            print(
                f"Early stopping at epoch {epoch + 1}. "
                f"Best validation Macro F1={best_f1:.4f} "
                f"at epoch {best_epoch}."
            )
            break

    best_model = keras.models.load_model(best_checkpoint)

    return best_model, history, time.time() - t0
