"""Evaluation helpers for classification metrics and confusion matrices."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.config import EMOTION_LABELS, GENDER_LABELS


def evaluate_age_gender(model, test_ds, age_group_labels: list[str]) -> dict:
    """Evaluate age-group and gender outputs and return report dictionary."""

    y_age_true: list[int] = []
    y_gender_true: list[int] = []
    for _, labels in test_ds:
        y_age_true.extend(np.asarray(labels["age_output"]).astype(np.int32).tolist())
        y_gender_true.extend(np.asarray(labels["gender_output"]).astype(np.int32).tolist())

    y_pred = model.predict(test_ds, verbose=0)
    if isinstance(y_pred, dict):
        age_probs = np.asarray(y_pred["age_output"])
        gender_probs = np.asarray(y_pred["gender_output"]).reshape(-1)
    else:
        age_probs = np.asarray(y_pred[0])
        gender_probs = np.asarray(y_pred[1]).reshape(-1)

    y_age_pred = np.argmax(age_probs, axis=1).astype(np.int32)
    y_gender_pred = (gender_probs >= 0.5).astype(np.int32)

    age_acc = accuracy_score(y_age_true, y_age_pred)
    gender_acc = accuracy_score(y_gender_true, y_gender_pred)

    age_cm = confusion_matrix(y_age_true, y_age_pred, labels=list(range(len(age_group_labels))))
    gender_cm = confusion_matrix(y_gender_true, y_gender_pred, labels=[0, 1])

    age_report = classification_report(
        y_age_true,
        y_age_pred,
        labels=list(range(len(age_group_labels))),
        target_names=age_group_labels,
        output_dict=True,
        zero_division=0,
    )
    gender_report = classification_report(
        y_gender_true,
        y_gender_pred,
        labels=[0, 1],
        target_names=GENDER_LABELS,
        output_dict=True,
        zero_division=0,
    )

    return {
        "age_accuracy": float(age_acc),
        "gender_accuracy": float(gender_acc),
        "age_confusion_matrix": age_cm,
        "gender_confusion_matrix": gender_cm,
        "age_report": age_report,
        "gender_report": gender_report,
        "y_age_true": np.asarray(y_age_true),
        "y_age_pred": y_age_pred,
        "y_gender_true": np.asarray(y_gender_true),
        "y_gender_pred": y_gender_pred,
    }


def evaluate_emotion(model, test_ds, emotion_labels: list[str] | None = None) -> dict:
    """Evaluate emotion classifier and return report dictionary."""

    labels = emotion_labels or EMOTION_LABELS
    y_true: list[int] = []
    for _, batch_labels in test_ds:
        y_true.extend(np.asarray(batch_labels).astype(np.int32).tolist())

    probs = np.asarray(model.predict(test_ds, verbose=0))
    y_pred = np.argmax(probs, axis=1).astype(np.int32)

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(labels))),
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    return {
        "emotion_accuracy": float(acc),
        "emotion_confusion_matrix": cm,
        "emotion_report": report,
        "y_true": np.asarray(y_true),
        "y_pred": y_pred,
    }

