"""Data loading and tf.data pipeline utilities."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split

from src.config import AGE_BINS, SEED


AUTOTUNE = tf.data.AUTOTUNE
_VALID_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
_UTKFACE_NAME_RE = re.compile(r"^(\d{1,3})_(\d)_(\d)_.+")


def _age_to_bin(age: int, age_bins: list[tuple[int, int]]) -> int:
    for idx, (lower, upper) in enumerate(age_bins):
        if lower <= age <= upper:
            return idx
    return -1


def _safe_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (UnidentifiedImageError, OSError, ValueError):
        return False


def _bin_label(age_bins: list[tuple[int, int]], idx: int) -> str:
    lower, upper = age_bins[idx]
    return f"{lower}+" if idx == len(age_bins) - 1 else f"{lower}-{upper}"


def load_utkface_dataframe(root_dir: str, age_bins: list[tuple[int, int]] = AGE_BINS) -> pd.DataFrame:
    """Load UTKFace file metadata and create age/gender labels.

    Returns:
        DataFrame with columns:
        - path
        - age
        - gender_idx (0 male, 1 female)
        - gender_label
        - age_group_idx
        - age_group_label
    """

    root_path = Path(root_dir)
    if not root_path.exists():
        raise FileNotFoundError(f"UTKFace directory does not exist: {root_path}")

    rows = []
    image_paths = [p for p in root_path.rglob("*") if p.suffix.lower() in _VALID_IMAGE_SUFFIXES]
    for img_path in image_paths:
        match = _UTKFACE_NAME_RE.match(img_path.name)
        if match is None:
            continue

        age = int(match.group(1))
        gender_idx = int(match.group(2))
        if gender_idx not in (0, 1):
            continue
        if age < 0 or age > 200:
            continue
        if not _safe_image(img_path):
            continue

        age_group_idx = _age_to_bin(age, age_bins)
        if age_group_idx < 0:
            continue

        rows.append(
            {
                "path": str(img_path),
                "age": age,
                "gender_idx": gender_idx,
                "gender_label": "male" if gender_idx == 0 else "female",
                "age_group_idx": age_group_idx,
                "age_group_label": _bin_label(age_bins, age_group_idx),
            }
        )

    if not rows:
        raise ValueError("No valid UTKFace samples found after parsing.")

    return pd.DataFrame(rows).reset_index(drop=True)


def _split_dataframe(
    df: pd.DataFrame,
    stratify_cols: Iterable[str],
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if df.empty:
        raise ValueError("Cannot split an empty DataFrame.")

    if test_size <= 0 or val_size <= 0 or test_size + val_size >= 1:
        raise ValueError("Split ratios must satisfy 0 < test,val and test+val < 1.")

    train_size = 1.0 - test_size - val_size
    work_df = df.copy()
    work_df["__strat__"] = work_df[list(stratify_cols)].astype(str).agg("_".join, axis=1)

    try:
        train_df, remaining_df = train_test_split(
            work_df,
            train_size=train_size,
            random_state=seed,
            stratify=work_df["__strat__"],
        )
        val_ratio_in_remaining = val_size / (test_size + val_size)
        val_df, test_df = train_test_split(
            remaining_df,
            train_size=val_ratio_in_remaining,
            random_state=seed,
            stratify=remaining_df["__strat__"],
        )
    except ValueError:
        train_df, remaining_df = train_test_split(
            work_df,
            train_size=train_size,
            random_state=seed,
            shuffle=True,
        )
        val_ratio_in_remaining = val_size / (test_size + val_size)
        val_df, test_df = train_test_split(
            remaining_df,
            train_size=val_ratio_in_remaining,
            random_state=seed,
            shuffle=True,
        )

    drop_col = ["__strat__"]
    return (
        train_df.drop(columns=drop_col).reset_index(drop=True),
        val_df.drop(columns=drop_col).reset_index(drop=True),
        test_df.drop(columns=drop_col).reset_index(drop=True),
    )


def _decode_resize_rgb(path: tf.Tensor, image_size: tuple[int, int]) -> tf.Tensor:
    image = tf.io.read_file(path)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, image_size, method="bilinear")
    image = tf.cast(image, tf.float32) / 255.0
    return image


def _augment_rgb(image: tf.Tensor) -> tf.Tensor:
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.08)
    image = tf.image.random_contrast(image, lower=0.85, upper=1.15)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image


def _build_age_gender_dataset(df: pd.DataFrame, batch_size: int, training: bool) -> tf.data.Dataset:
    if df.empty:
        raise ValueError("Age/Gender dataset split is empty.")

    ds = tf.data.Dataset.from_tensor_slices(
        (
            df["path"].astype(str).to_numpy(),
            df["age_group_idx"].astype(np.int32).to_numpy(),
            df["gender_idx"].astype(np.float32).to_numpy(),
        )
    )

    if training:
        ds = ds.shuffle(len(df), seed=SEED, reshuffle_each_iteration=True)

    def map_fn(path: tf.Tensor, age_idx: tf.Tensor, gender_idx: tf.Tensor):
        image = _decode_resize_rgb(path, image_size=(128, 128))
        if training:
            image = _augment_rgb(image)
        labels = {
            "age_output": tf.cast(age_idx, tf.int32),
            "gender_output": tf.cast(gender_idx, tf.float32),
        }
        return image, labels

    ds = ds.map(map_fn, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
    return ds


def build_age_gender_datasets(df: pd.DataFrame, batch_size: int) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Build train/val/test datasets for the age+gender model."""

    train_df, val_df, test_df = _split_dataframe(
        df=df,
        stratify_cols=("age_group_idx", "gender_idx"),
        test_size=0.15,
        val_size=0.15,
        seed=SEED,
    )
    return (
        _build_age_gender_dataset(train_df, batch_size=batch_size, training=True),
        _build_age_gender_dataset(val_df, batch_size=batch_size, training=False),
        _build_age_gender_dataset(test_df, batch_size=batch_size, training=False),
    )


def _split_fer_usage(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    usage = df["Usage"].astype(str).str.lower().str.strip()
    train_df = df[usage == "training"].copy()
    val_df = df[usage == "publictest"].copy()
    test_df = df[usage == "privatetest"].copy()

    if train_df.empty or val_df.empty or test_df.empty:
        train_df, temp_df = train_test_split(
            df,
            train_size=0.7,
            random_state=SEED,
            stratify=df["emotion"],
        )
        val_df, test_df = train_test_split(
            temp_df,
            train_size=0.5,
            random_state=SEED,
            stratify=temp_df["emotion"],
        )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def load_fer2013_dataframe(csv_path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load FER2013 CSV and split into train/val/test frames."""

    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"FER2013 CSV not found: {csv_file}")

    df = pd.read_csv(csv_file)
    required_cols = {"emotion", "pixels"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"FER2013 CSV must include columns: {required_cols}")
    if "Usage" not in df.columns:
        df["Usage"] = "Training"

    df["emotion"] = df["emotion"].astype(np.int32)
    df = df[df["emotion"].between(0, 6)].reset_index(drop=True)
    return _split_fer_usage(df)


def _parse_pixels_to_uint8(pixel_str: str) -> np.ndarray:
    arr = np.fromstring(pixel_str, sep=" ", dtype=np.uint8)
    if arr.size != 48 * 48:
        raise ValueError(f"Invalid FER pixel string length: {arr.size}")
    return arr.reshape(48, 48, 1)


def _fer_frame_to_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    images = np.zeros((len(df), 48, 48, 1), dtype=np.uint8)
    labels = df["emotion"].astype(np.int32).to_numpy()
    valid_idx = []

    for row_idx, pixel_str in enumerate(df["pixels"].astype(str)):
        try:
            images[row_idx] = _parse_pixels_to_uint8(pixel_str)
            valid_idx.append(row_idx)
        except ValueError:
            continue

    if not valid_idx:
        raise ValueError("No valid FER images found in frame split.")

    valid_idx_np = np.array(valid_idx, dtype=np.int32)
    return images[valid_idx_np], labels[valid_idx_np]


def _augment_grayscale(image: tf.Tensor) -> tf.Tensor:
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.08)
    image = tf.image.random_contrast(image, lower=0.85, upper=1.15)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image


def _build_emotion_dataset(images: np.ndarray, labels: np.ndarray, batch_size: int, training: bool) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    if training:
        ds = ds.shuffle(len(images), seed=SEED, reshuffle_each_iteration=True)

    def map_fn(image: tf.Tensor, label: tf.Tensor):
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.image.resize(image, (64, 64), method="bilinear")
        if training:
            image = _augment_grayscale(image)
        return image, tf.cast(label, tf.int32)

    ds = ds.map(map_fn, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
    return ds


def build_emotion_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    batch_size: int,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Build train/val/test datasets for FER emotion classification."""

    x_train, y_train = _fer_frame_to_arrays(train_df)
    x_val, y_val = _fer_frame_to_arrays(val_df)
    x_test, y_test = _fer_frame_to_arrays(test_df)

    return (
        _build_emotion_dataset(x_train, y_train, batch_size=batch_size, training=True),
        _build_emotion_dataset(x_val, y_val, batch_size=batch_size, training=False),
        _build_emotion_dataset(x_test, y_test, batch_size=batch_size, training=False),
    )

