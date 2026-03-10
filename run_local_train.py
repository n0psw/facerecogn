"""Train and evaluate the face analysis pipeline locally (CPU by default)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local training for age-gender and emotion models.")
    parser.add_argument("--utkface-root", type=Path, required=True, help="Path to UTKFace folder.")
    parser.add_argument("--fer-csv", type=Path, required=True, help="Path to fer2013.csv.")
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts"), help="Output directory for checkpoints and metrics.")

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--age-gender-epochs", type=int, default=5)
    parser.add_argument("--emotion-epochs", type=int, default=8)
    parser.add_argument("--initial-lr", type=float, default=1e-3)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--reduce-lr-patience", type=int, default=1)
    parser.add_argument("--reduce-lr-factor", type=float, default=0.5)
    parser.add_argument("--min-lr", type=float, default=1e-6)

    parser.add_argument("--max-utkface-samples", type=int, default=0, help="Optional cap for UTKFace samples. 0 disables cap.")
    parser.add_argument("--max-fer-train", type=int, default=0, help="Optional cap for FER train samples. 0 disables cap.")
    parser.add_argument("--max-fer-val", type=int, default=0, help="Optional cap for FER validation samples. 0 disables cap.")
    parser.add_argument("--max-fer-test", type=int, default=0, help="Optional cap for FER test samples. 0 disables cap.")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu", help="Use CPU by default for local compatibility.")
    return parser.parse_args()


def _configure_device(device: str, tf: Any) -> None:
    if device == "cpu":
        # Disable all TensorFlow GPU devices explicitly.
        tf.config.set_visible_devices([], "GPU")
    print("TensorFlow:", tf.__version__)
    print("Visible GPUs:", tf.config.list_physical_devices("GPU"))


def _sample_df(df: "pd.DataFrame", max_rows: int, seed: int) -> "pd.DataFrame":
    if max_rows <= 0 or len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=seed).reset_index(drop=True)


def _build_train_config(args: argparse.Namespace, TrainConfig: Any) -> Any:
    return TrainConfig(
        batch_size=args.batch_size,
        age_gender_epochs=args.age_gender_epochs,
        emotion_epochs=args.emotion_epochs,
        initial_lr=args.initial_lr,
        early_stopping_patience=args.early_stopping_patience,
        reduce_lr_patience=args.reduce_lr_patience,
        reduce_lr_factor=args.reduce_lr_factor,
        min_lr=args.min_lr,
    )


def main() -> None:
    args = parse_args()
    import tensorflow as tf

    from src.config import AGE_GROUP_LABELS, EMOTION_LABELS, TrainConfig
    from src.data_pipeline import (
        build_age_gender_datasets,
        build_emotion_datasets,
        load_fer2013_dataframe,
        load_utkface_dataframe,
    )
    from src.evaluate import evaluate_age_gender, evaluate_emotion
    from src.models import build_age_gender_model, build_emotion_model
    from src.train import train_age_gender, train_emotion

    _configure_device(args.device, tf=tf)

    if not args.utkface_root.exists():
        raise FileNotFoundError(f"UTKFace directory not found: {args.utkface_root}")
    if not args.fer_csv.exists():
        raise FileNotFoundError(f"FER CSV not found: {args.fer_csv}")

    args.artifacts_root.mkdir(parents=True, exist_ok=True)
    checkpoints_root = args.artifacts_root / "checkpoints"
    checkpoints_root.mkdir(parents=True, exist_ok=True)

    train_cfg = _build_train_config(args, TrainConfig=TrainConfig)
    print("TrainConfig:", train_cfg)

    print("Loading UTKFace...")
    utk_df = load_utkface_dataframe(str(args.utkface_root))
    utk_df = _sample_df(utk_df, max_rows=args.max_utkface_samples, seed=args.seed)
    print("UTKFace samples:", len(utk_df))

    print("Loading FER2013 CSV...")
    fer_train_df, fer_val_df, fer_test_df = load_fer2013_dataframe(str(args.fer_csv))
    fer_train_df = _sample_df(fer_train_df, max_rows=args.max_fer_train, seed=args.seed)
    fer_val_df = _sample_df(fer_val_df, max_rows=args.max_fer_val, seed=args.seed)
    fer_test_df = _sample_df(fer_test_df, max_rows=args.max_fer_test, seed=args.seed)
    print("FER split sizes:", {"train": len(fer_train_df), "val": len(fer_val_df), "test": len(fer_test_df)})

    print("Building datasets...")
    ag_train_ds, ag_val_ds, ag_test_ds = build_age_gender_datasets(utk_df, batch_size=train_cfg.batch_size)
    emo_train_ds, emo_val_ds, emo_test_ds = build_emotion_datasets(
        train_df=fer_train_df,
        val_df=fer_val_df,
        test_df=fer_test_df,
        batch_size=train_cfg.batch_size,
    )

    print("Building models...")
    ag_model = build_age_gender_model(num_age_classes=len(AGE_GROUP_LABELS))
    emo_model = build_emotion_model(num_emotions=len(EMOTION_LABELS))

    print("Training age-gender model...")
    train_age_gender(
        model=ag_model,
        train_ds=ag_train_ds,
        val_ds=ag_val_ds,
        out_dir=str(checkpoints_root),
        train_cfg=train_cfg,
    )

    print("Training emotion model...")
    train_emotion(
        model=emo_model,
        train_ds=emo_train_ds,
        val_ds=emo_val_ds,
        out_dir=str(checkpoints_root),
        train_cfg=train_cfg,
    )

    ag_ckpt = checkpoints_root / "age_gender" / "age_gender_best.keras"
    emo_ckpt = checkpoints_root / "emotion" / "emotion_best.keras"
    ag_eval_model = tf.keras.models.load_model(ag_ckpt) if ag_ckpt.exists() else ag_model
    emo_eval_model = tf.keras.models.load_model(emo_ckpt) if emo_ckpt.exists() else emo_model

    print("Evaluating models...")
    ag_metrics = evaluate_age_gender(ag_eval_model, ag_test_ds, age_group_labels=AGE_GROUP_LABELS)
    emo_metrics = evaluate_emotion(emo_eval_model, emo_test_ds, emotion_labels=EMOTION_LABELS)

    summary = {
        "config": {
            "utkface_root": str(args.utkface_root),
            "fer_csv": str(args.fer_csv),
            "artifacts_root": str(args.artifacts_root),
            "train_cfg": train_cfg.__dict__,
        },
        "age_gender": {
            "accuracy_age": ag_metrics["age_accuracy"],
            "accuracy_gender": ag_metrics["gender_accuracy"],
            "confusion_matrix_age": ag_metrics["age_confusion_matrix"].tolist(),
            "confusion_matrix_gender": ag_metrics["gender_confusion_matrix"].tolist(),
            "report_age": ag_metrics["age_report"],
            "report_gender": ag_metrics["gender_report"],
        },
        "emotion": {
            "accuracy": emo_metrics["emotion_accuracy"],
            "confusion_matrix": emo_metrics["emotion_confusion_matrix"].tolist(),
            "report": emo_metrics["emotion_report"],
        },
        "checkpoints": {
            "age_gender": str(ag_ckpt),
            "emotion": str(emo_ckpt),
        },
    }

    metrics_path = args.artifacts_root / "metrics_summary.json"
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Done. Metrics saved:", metrics_path)
    print("Age accuracy:", f"{ag_metrics['age_accuracy']:.4f}")
    print("Gender accuracy:", f"{ag_metrics['gender_accuracy']:.4f}")
    print("Emotion accuracy:", f"{emo_metrics['emotion_accuracy']:.4f}")


if __name__ == "__main__":
    main()
