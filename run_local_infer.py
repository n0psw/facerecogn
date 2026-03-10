"""Run local face inference for an image or video using trained checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local inference for the face analysis project.")
    parser.add_argument("--image", type=Path, help="Path to input image.")
    parser.add_argument("--video", type=Path, help="Path to input video.")
    parser.add_argument("--out", type=Path, help="Optional output path for annotated media.")
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--age-gender-model", type=Path, default=Path("artifacts/checkpoints/age_gender/age_gender_best.keras"))
    parser.add_argument("--emotion-model", type=Path, default=Path("artifacts/checkpoints/emotion/emotion_best.keras"))
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    args = parser.parse_args()

    if bool(args.image) == bool(args.video):
        raise ValueError("Use exactly one source: --image or --video.")
    return args


def _configure_device(device: str, tf: Any) -> None:
    if device == "cpu":
        tf.config.set_visible_devices([], "GPU")
    print("TensorFlow:", tf.__version__)
    print("Visible GPUs:", tf.config.list_physical_devices("GPU"))


def _default_output_path(args: argparse.Namespace) -> Path:
    in_path = args.image if args.image else args.video
    suffix = ".jpg" if args.image else ".mp4"
    return args.artifacts_root / "inference" / f"{in_path.stem}_annotated{suffix}"


def _assert_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def _run_image_inference(
    image_path: Path,
    out_path: Path,
    cv2: Any,
    annotate_image: Any,
    detect_faces_bgr: Any,
    predict_face: Any,
    detector,
    ag_model: Any,
    emo_model: Any,
) -> list[dict]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    face_boxes = detect_faces_bgr(image, detector)
    detections: list[tuple] = []
    records: list[dict] = []
    for box in face_boxes:
        crop = image[box.y1 : box.y2, box.x1 : box.x2]
        if crop.size == 0:
            continue

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pred = predict_face(crop_rgb, ag_model=ag_model, emo_model=emo_model)
        detections.append((box, pred))
        records.append(
            {
                "box": {"x1": box.x1, "y1": box.y1, "x2": box.x2, "y2": box.y2, "score": box.score},
                "prediction": {
                    "age_group": pred.age_group,
                    "gender": pred.gender,
                    "emotion": pred.emotion,
                    "probs": pred.probs,
                },
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    annotated = annotate_image(image, detections)
    ok = cv2.imwrite(str(out_path), annotated)
    if not ok:
        raise RuntimeError(f"Failed to write output image: {out_path}")
    return records


def main() -> None:
    args = parse_args()
    import cv2
    import tensorflow as tf

    from src.infer import annotate_image, detect_faces_bgr, load_detector, predict_face, run_video_inference

    _configure_device(args.device, tf=tf)

    _assert_exists(args.age_gender_model, "Age-gender checkpoint")
    _assert_exists(args.emotion_model, "Emotion checkpoint")
    if args.image:
        _assert_exists(args.image, "Input image")
    if args.video:
        _assert_exists(args.video, "Input video")

    args.artifacts_root.mkdir(parents=True, exist_ok=True)
    out_path = args.out or _default_output_path(args)

    print("Loading checkpoints...")
    ag_model = tf.keras.models.load_model(args.age_gender_model)
    emo_model = tf.keras.models.load_model(args.emotion_model)
    detector = load_detector()

    if args.image:
        print("Running image inference...")
        records = _run_image_inference(
            image_path=args.image,
            out_path=out_path,
            cv2=cv2,
            annotate_image=annotate_image,
            detect_faces_bgr=detect_faces_bgr,
            predict_face=predict_face,
            detector=detector,
            ag_model=ag_model,
            emo_model=emo_model,
        )
        summary_path = out_path.with_suffix(".json")
        summary_path.write_text(json.dumps({"image": str(args.image), "faces": records}, indent=2), encoding="utf-8")
        print("Annotated image:", out_path)
        print("Predictions:", summary_path)
        print("Faces detected:", len(records))
    else:
        print("Running video inference...")
        saved_video = run_video_inference(
            video_path=str(args.video),
            out_path=str(out_path),
            detector=detector,
            ag_model=ag_model,
            emo_model=emo_model,
        )
        print("Annotated video:", saved_video)


if __name__ == "__main__":
    main()
