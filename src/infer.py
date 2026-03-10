"""Inference utilities for image and video face analysis."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.config import AGE_GROUP_LABELS, EMOTION_LABELS, GENDER_LABELS, TrainConfig
from src.types import FaceBox, FacePrediction


def load_detector() -> object:
    """Create and return MediaPipe face detector."""

    try:
        import mediapipe as mp
    except ImportError as exc:
        raise ImportError("mediapipe is not installed. Run: pip install mediapipe") from exc

    cfg = TrainConfig()
    return mp.solutions.face_detection.FaceDetection(
        model_selection=0,
        min_detection_confidence=cfg.detector_confidence,
    )


def detect_faces_bgr(image: np.ndarray, detector) -> list[FaceBox]:
    """Detect faces in BGR image and return absolute pixel boxes."""

    if image is None or image.size == 0:
        return []

    h, w = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = detector.process(rgb)
    detections = []
    if not results.detections:
        return detections

    for det in results.detections:
        bbox = det.location_data.relative_bounding_box
        x1 = int(max(0, bbox.xmin * w))
        y1 = int(max(0, bbox.ymin * h))
        x2 = int(min(w, (bbox.xmin + bbox.width) * w))
        y2 = int(min(h, (bbox.ymin + bbox.height) * h))
        if x2 <= x1 or y2 <= y1:
            continue
        detections.append(FaceBox(x1=x1, y1=y1, x2=x2, y2=y2, score=float(det.score[0])))
    return detections


def _predict_age_gender(face_rgb: np.ndarray, ag_model):
    face_ag = cv2.resize(face_rgb, (128, 128), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    face_ag = np.expand_dims(face_ag, axis=0)
    pred = ag_model.predict(face_ag, verbose=0)
    if isinstance(pred, dict):
        age_probs = np.asarray(pred["age_output"])[0]
        female_prob = float(np.asarray(pred["gender_output"]).reshape(-1)[0])
    else:
        age_probs = np.asarray(pred[0])[0]
        female_prob = float(np.asarray(pred[1]).reshape(-1)[0])
    return age_probs, female_prob


def _predict_emotion(face_rgb: np.ndarray, emo_model):
    gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    gray = np.expand_dims(gray, axis=(0, -1))
    probs = np.asarray(emo_model.predict(gray, verbose=0))[0]
    return probs


def predict_face(face_rgb: np.ndarray, ag_model, emo_model) -> FacePrediction:
    """Run age-group, gender, and emotion prediction for a cropped face."""

    if face_rgb is None or face_rgb.size == 0:
        raise ValueError("Face crop is empty.")

    age_probs, female_prob = _predict_age_gender(face_rgb, ag_model)
    emotion_probs = _predict_emotion(face_rgb, emo_model)

    age_idx = int(np.argmax(age_probs))
    gender_idx = 1 if female_prob >= 0.5 else 0
    emotion_idx = int(np.argmax(emotion_probs))

    probs = {
        "age": {AGE_GROUP_LABELS[i]: float(age_probs[i]) for i in range(len(AGE_GROUP_LABELS))},
        "gender": {"male": float(1.0 - female_prob), "female": float(female_prob)},
        "emotion": {EMOTION_LABELS[i]: float(emotion_probs[i]) for i in range(len(EMOTION_LABELS))},
    }
    return FacePrediction(
        age_group=AGE_GROUP_LABELS[age_idx],
        gender=GENDER_LABELS[gender_idx],
        emotion=EMOTION_LABELS[emotion_idx],
        probs=probs,
    )


def annotate_image(image: np.ndarray, detections: list[tuple[FaceBox, FacePrediction]]) -> np.ndarray:
    """Draw boxes and labels on image."""

    out = image.copy()
    for box, pred in detections:
        cv2.rectangle(out, (box.x1, box.y1), (box.x2, box.y2), (0, 255, 0), 2)
        label_1 = f"{pred.age_group} | {pred.gender} | {pred.emotion}"
        label_2 = f"det:{box.score:.2f} emo:{max(pred.probs['emotion'].values()):.2f}"
        y_anchor = max(20, box.y1 - 10)
        cv2.putText(out, label_1, (box.x1, y_anchor), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(out, label_2, (box.x1, y_anchor + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def run_video_inference(video_path: str, out_path: str, detector, ag_model, emo_model) -> str:
    """Process a video and save annotated output."""

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        face_boxes = detect_faces_bgr(frame, detector)
        frame_preds: list[tuple[FaceBox, FacePrediction]] = []
        for box in face_boxes:
            crop = frame[box.y1 : box.y2, box.x1 : box.x2]
            if crop.size == 0:
                continue
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            prediction = predict_face(crop_rgb, ag_model=ag_model, emo_model=emo_model)
            frame_preds.append((box, prediction))

        frame_out = annotate_image(frame, frame_preds)
        writer.write(frame_out)

    capture.release()
    writer.release()
    return str(output_path)

