"""Training: classical SVM baseline + deep CNN-BiLSTM-Attention model."""
from __future__ import annotations

import time

import numpy as np
import keras
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight

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
    t = cfg.train
    set_seed(t.seed)
    model = build_model(Xtr.shape[1:], int(max(ytr.max(), yva.max())) + 1, lr=t.lr)
    cw = compute_class_weight("balanced", classes=np.unique(ytr), y=ytr)
    cbs = [
        keras.callbacks.ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=t.patience, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(3, t.patience // 3), min_lr=1e-5),
    ]
    t0 = time.time()
    hist = model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=t.epochs, batch_size=t.batch_size,
                     class_weight=dict(enumerate(cw)), callbacks=cbs, verbose=2)
    return model, hist.history, time.time() - t0
