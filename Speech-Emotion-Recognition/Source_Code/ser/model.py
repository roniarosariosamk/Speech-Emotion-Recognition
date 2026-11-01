
"""Smaller CNN-BiLSTM-Attention model for speech emotion recognition."""

from __future__ import annotations

import keras

from keras import layers


@keras.saving.register_keras_serializable(package="ser")
class AttentionPool(layers.Layer):
    """Learn which time steps are most relevant for emotion recognition."""

    def __init__(self, units: int = 32, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.proj = layers.Dense(units, activation="tanh")
        self.score = layers.Dense(1, use_bias=False)

    def call(self, x):
        weights = keras.ops.softmax(
            self.score(self.proj(x)),
            axis=1,
        )
        return keras.ops.sum(weights * x, axis=1)

    def get_config(self):
        return {
            **super().get_config(),
            "units": self.units,
        }


def conv_block(x, filters: int, kernel: int, drop: float):
    """Convolutional block with regularization."""

    x = layers.Conv1D(
        filters,
        kernel,
        padding="same",
        kernel_regularizer=keras.regularizers.l2(2e-4),
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(drop)(x)

    return x


def build_model(input_shape: tuple, n_classes: int, lr: float = 5e-4):
    """Build a compact CNN-BiLSTM-Attention network."""

    inputs = keras.Input(shape=input_shape, name="mfcc")

    x = layers.GaussianNoise(0.08)(inputs)

    # Smaller CNN feature extractor.
    x = conv_block(x, filters=32, kernel=5, drop=0.30)
    x = conv_block(x, filters=64, kernel=3, drop=0.35)

    # Smaller recurrent layer to reduce model complexity.
    x = layers.Bidirectional(
        layers.LSTM(
            48,
            return_sequences=True,
            dropout=0.30,
            recurrent_dropout=0.0,
        )
    )(x)

    # Temporal attention.
    x = AttentionPool(32, name="attention")(x)

    # Smaller classification head.
    x = layers.Dense(
        64,
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(2e-4),
    )(x)

    x = layers.Dropout(0.50)(x)

    outputs = layers.Dense(
        n_classes,
        activation="softmax",
        name="emotion",
    )(x)

    model = keras.Model(
        inputs,
        outputs,
        name="compact_cnn_bilstm_attention",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    return model

