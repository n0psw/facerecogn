"""Training utilities for age-gender and emotion models."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from src.config import TrainConfig


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _ensure_dir(path: Path | str) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def _base_callbacks(out_dir: Path, task_name: str, monitor: str = "val_loss", mode: str = "min", train_cfg: TrainConfig | None = None):
    cfg = train_cfg or TrainConfig()
    task_out = _ensure_dir(out_dir / task_name)
    ckpt_path = task_out / f"{task_name}_best.keras"
    tb_log_dir = _ensure_dir(task_out / "tensorboard" / _timestamp())
    csv_log_path = task_out / f"{task_name}_history.csv"

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            mode=mode,
            patience=cfg.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            mode=mode,
            factor=cfg.reduce_lr_factor,
            patience=cfg.reduce_lr_patience,
            min_lr=cfg.min_lr,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_path),
            monitor=monitor,
            mode=mode,
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(log_dir=str(tb_log_dir), histogram_freq=1),
        tf.keras.callbacks.CSVLogger(str(csv_log_path), append=False),
    ]
    return callbacks


def train_age_gender(model, train_ds, val_ds, out_dir: str) -> tf.keras.callbacks.History:
    """Compile and train the age+gender multi-output model."""

    cfg = TrainConfig()
    out_path = _ensure_dir(out_dir)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.initial_lr),
        loss={
            "age_output": tf.keras.losses.SparseCategoricalCrossentropy(),
            "gender_output": tf.keras.losses.BinaryCrossentropy(),
        },
        loss_weights={"age_output": 1.0, "gender_output": 1.0},
        metrics={
            "age_output": [tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
            "gender_output": [tf.keras.metrics.BinaryAccuracy(name="accuracy")],
        },
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.age_gender_epochs,
        callbacks=_base_callbacks(out_path, task_name="age_gender", train_cfg=cfg),
        verbose=1,
    )
    return history


def _emotion_class_weights_from_dataset(train_ds, num_classes: int = 7) -> dict[int, float]:
    labels: list[int] = []
    for _, y_batch in train_ds:
        labels.extend(np.asarray(y_batch).astype(np.int32).tolist())
    if not labels:
        return {i: 1.0 for i in range(num_classes)}

    labels_np = np.asarray(labels, dtype=np.int32)
    present_classes = np.unique(labels_np)
    weights_np = compute_class_weight(class_weight="balanced", classes=present_classes, y=labels_np)
    weight_map = {int(c): float(w) for c, w in zip(present_classes, weights_np)}
    for cls in range(num_classes):
        weight_map.setdefault(cls, 1.0)
    return weight_map


def train_emotion(model, train_ds, val_ds, out_dir: str) -> tf.keras.callbacks.History:
    """Compile and train the FER emotion model."""

    cfg = TrainConfig()
    out_path = _ensure_dir(out_dir)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.initial_lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.emotion_epochs,
        callbacks=_base_callbacks(out_path, task_name="emotion", monitor="val_accuracy", mode="max", train_cfg=cfg),
        class_weight=_emotion_class_weights_from_dataset(train_ds, num_classes=7),
        verbose=1,
    )
    return history

