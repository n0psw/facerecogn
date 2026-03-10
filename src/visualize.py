"""Visualization utilities used by the Colab notebook."""

from __future__ import annotations

import math
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_label_distribution(df: pd.DataFrame, column: str, title: str, normalize: bool = False) -> None:
    """Plot class distribution for a DataFrame column."""

    if column not in df.columns:
        raise KeyError(f"Column '{column}' is missing in DataFrame.")

    counts = df[column].value_counts(normalize=normalize).sort_index()
    plt.figure(figsize=(10, 4))
    sns.barplot(x=counts.index.astype(str), y=counts.values, color="#4472C4")
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel("ratio" if normalize else "count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def show_image_grid(images: np.ndarray, titles: Iterable[str] | None = None, ncols: int = 6, figsize: tuple[int, int] = (14, 8)) -> None:
    """Show a grid of images with optional titles."""

    if images.size == 0:
        raise ValueError("No images to display.")

    titles_list = list(titles) if titles is not None else []
    total = len(images)
    ncols = max(1, min(ncols, total))
    nrows = math.ceil(total / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.array(axes).reshape(-1)
    for idx, ax in enumerate(axes):
        ax.axis("off")
        if idx >= total:
            continue
        image = images[idx]
        if image.ndim == 2 or (image.ndim == 3 and image.shape[-1] == 1):
            ax.imshow(np.squeeze(image), cmap="gray")
        else:
            ax.imshow(image)
        if idx < len(titles_list):
            ax.set_title(titles_list[idx], fontsize=9)
    plt.tight_layout()
    plt.show()


def plot_training_history(history, title: str) -> None:
    """Plot Keras History object in a compact figure."""

    history_dict = history.history if hasattr(history, "history") else dict(history)
    keys = list(history_dict.keys())

    loss_keys = [k for k in keys if "loss" in k and not k.startswith("val_")]
    metric_keys = [k for k in keys if "accuracy" in k and not k.startswith("val_")]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    if loss_keys:
        for key in loss_keys:
            axes[0].plot(history_dict[key], label=key)
            val_key = f"val_{key}"
            if val_key in history_dict:
                axes[0].plot(history_dict[val_key], label=val_key)
        axes[0].set_title(f"{title} - Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].legend()
    else:
        axes[0].axis("off")

    if metric_keys:
        for key in metric_keys:
            axes[1].plot(history_dict[key], label=key)
            val_key = f"val_{key}"
            if val_key in history_dict:
                axes[1].plot(history_dict[val_key], label=val_key)
        axes[1].set_title(f"{title} - Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()
    else:
        axes[1].axis("off")

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], title: str = "Confusion Matrix") -> None:
    """Render a confusion matrix as a heatmap."""

    cm = np.asarray(cm)
    if cm.ndim != 2:
        raise ValueError("Confusion matrix must be 2D.")

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

