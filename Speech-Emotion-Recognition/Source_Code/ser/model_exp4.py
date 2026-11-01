"""Experiment 4 model: Experiment-3 compact CNN-BiLSTM-Attention with slightly stronger dropout.

Only the dropout rates differ from Experiment 3 (see EXP4_CONFIG vs. ser/model.py):

    first CNN dropout   0.30 -> 0.35
    second CNN dropout  0.35 -> 0.40
    BiLSTM dropout      0.30 -> 0.35
    dense dropout       0.50 (unchanged)
    Gaussian noise 0.08, L2 2e-4, filters/units (unchanged)

The custom AttentionPool layer is re-used unchanged from ser.model so that it is registered exactly
once (as "ser>AttentionPool") and checkpoints from every experiment load with the same class.
"""
from __future__ import annotations

import keras
from keras import layers

from .model import AttentionPool  # unchanged layer

EXP4_CONFIG = dict(
    gaussian_noise=0.08,
    conv1=dict(filters=32, kernel=5, dropout=0.35),
    conv2=dict(filters=64, kernel=3, dropout=0.40),
    lstm=dict(units=48, dropout=0.35),
    attention_units=32,
    dense=dict(units=64, dropout=0.50),
    l2=2e-4,
)


def conv_block(x, filters: int, kernel: int, drop: float, l2: float):
    x = layers.Conv1D(filters, kernel, padding="same", kernel_regularizer=keras.regularizers.l2(l2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(2)(x)
    return layers.Dropout(drop)(x)


def build_model(input_shape: tuple, n_classes: int, lr: float = 5e-4):
    """Compact CNN-BiLSTM-Attention network (Experiment 4 regularisation)."""
    c = EXP4_CONFIG
    inputs = keras.Input(shape=input_shape, name="mfcc")
    x = layers.GaussianNoise(c["gaussian_noise"])(inputs)
    x = conv_block(x, c["conv1"]["filters"], c["conv1"]["kernel"], c["conv1"]["dropout"], c["l2"])
    x = conv_block(x, c["conv2"]["filters"], c["conv2"]["kernel"], c["conv2"]["dropout"], c["l2"])
    x = layers.Bidirectional(layers.LSTM(c["lstm"]["units"], return_sequences=True,
                                         dropout=c["lstm"]["dropout"], recurrent_dropout=0.0))(x)
    x = AttentionPool(c["attention_units"], name="attention")(x)
    x = layers.Dense(c["dense"]["units"], activation="relu", kernel_regularizer=keras.regularizers.l2(c["l2"]))(x)
    x = layers.Dropout(c["dense"]["dropout"])(x)
    outputs = layers.Dense(n_classes, activation="softmax", name="emotion")(x)
    model = keras.Model(inputs, outputs, name="compact_cnn_bilstm_attention_exp4")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss=keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])
    return model
