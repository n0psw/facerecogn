"""Model builders for age-gender and emotion tasks."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers


def _conv_block(x: tf.Tensor, filters: int, dropout_rate: float = 0.0) -> tf.Tensor:
    x = layers.Conv2D(filters, kernel_size=3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, kernel_size=3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate)(x)
    return x


def build_age_gender_model(input_shape: tuple[int, int, int] = (128, 128, 3), num_age_classes: int = 8) -> tf.keras.Model:
    """Create a shared CNN backbone with age and gender heads."""

    inputs = layers.Input(shape=input_shape, name="image_input")
    x = _conv_block(inputs, 32, dropout_rate=0.1)
    x = _conv_block(x, 64, dropout_rate=0.15)
    x = _conv_block(x, 128, dropout_rate=0.2)
    x = _conv_block(x, 256, dropout_rate=0.25)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.35)(x)

    age_head = layers.Dense(128, activation="relu")(x)
    age_head = layers.Dropout(0.2)(age_head)
    age_output = layers.Dense(num_age_classes, activation="softmax", name="age_output")(age_head)

    gender_head = layers.Dense(64, activation="relu")(x)
    gender_head = layers.Dropout(0.2)(gender_head)
    gender_output = layers.Dense(1, activation="sigmoid", name="gender_output")(gender_head)

    return tf.keras.Model(inputs=inputs, outputs={"age_output": age_output, "gender_output": gender_output}, name="age_gender_model")


def build_emotion_model(input_shape: tuple[int, int, int] = (64, 64, 1), num_emotions: int = 7) -> tf.keras.Model:
    """Create an emotion classification CNN from scratch."""

    inputs = layers.Input(shape=input_shape, name="emotion_input")
    x = _conv_block(inputs, 32, dropout_rate=0.1)
    x = _conv_block(x, 64, dropout_rate=0.15)
    x = _conv_block(x, 128, dropout_rate=0.2)

    x = layers.Conv2D(256, kernel_size=3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_emotions, activation="softmax", name="emotion_output")(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="emotion_model")

