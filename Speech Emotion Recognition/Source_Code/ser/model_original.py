"""CNN -> BiLSTM -> temporal-attention network for MFCC sequences."""
from __future__ import annotations

import keras
from keras import layers



@keras.saving.register_keras_serializable(package="ser")
class AttentionPool(layers.Layer):
    """Additive attention pooling over time."""

    def __init__(self, units: int = 64, **kw):
        super().__init__(**kw)
        self.units = units
        self.proj = layers.Dense(units, activation="tanh")
        self.score = layers.Dense(1, use_bias=False)

    def build(self, input_shape):
        self.proj.build(input_shape)
        projected_shape = (*input_shape[:-1], self.units)
        self.score.build(projected_shape)
        super().build(input_shape)

    def call(self, x):
        w = keras.ops.softmax(
            self.score(self.proj(x)),
            axis=1
        )
        return keras.ops.sum(w * x, axis=1)

    def get_config(self):
        return {
            **super().get_config(),
            "units": self.units
        }


def conv_block(x, filters: int, kernel: int, drop: float):
    x = layers.Conv1D(filters, kernel, padding="same", kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(2)(x)
    return layers.Dropout(drop)(x)


def build_model(input_shape: tuple, n_classes: int, lr: float = 1e-3):
    inp = keras.Input(shape=input_shape, name="mfcc")
    x = layers.GaussianNoise(0.05)(inp)                          # light regularisation on features
    x = conv_block(x, 64, 5, 0.2)
    x = conv_block(x, 128, 5, 0.25)
    x = conv_block(x, 128, 3, 0.3)
    x = layers.Bidirectional(layers.LSTM(96, return_sequences=True, dropout=0.2))(x)
    x = AttentionPool(64, name="attention")(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(n_classes, activation="softmax", name="emotion")(x)
    model = keras.Model(inp, out, name="cnn_bilstm_attention")
    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss=keras.losses.SparseCategoricalCrossentropy(),
                  metrics=["accuracy"])
    return model
