"""Inference utilities for image and video face analysis."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.config import AGE_GROUP_LABELS, EMOTION_LABELS, GENDER_LABELS, TrainConfig
from src.types import FaceBox, FacePrediction


@dataclass(frozen=True)
class RealtimeConfig:
    """Runtime knobs for webcam-friendly inference."""

    inference_stride: int = 2
    smoothing_window: int = 5
    max_track_lost_frames: int = 12
    min_match_iou: float = 0.2
    max_center_distance_ratio: float = 0.35


@dataclass
class _TrackedFace:
    """Internal state for a single tracked face."""

    track_id: int
    last_box: FaceBox
    last_seen_frame_idx: int
    age_probs_history: deque[np.ndarray]
    female_prob_history: deque[float]
    emotion_probs_history: deque[np.ndarray]


@dataclass
class RealtimeState:
    """Mutable state shared across sequential realtime frames."""

    frame_idx: int = 0
    next_track_id: int = 1
    tracks: dict[int, _TrackedFace] = field(default_factory=dict)
    latest_detections: list[tuple[FaceBox, FacePrediction]] = field(default_factory=list)
    latest_results: list[dict[str, Any]] = field(default_factory=list)
    last_error: str | None = None


def _clamp_realtime_config(config: RealtimeConfig | None) -> RealtimeConfig:
    if config is None:
        return RealtimeConfig()
    return RealtimeConfig(
        inference_stride=max(1, int(config.inference_stride)),
        smoothing_window=max(1, int(config.smoothing_window)),
        max_track_lost_frames=max(1, int(config.max_track_lost_frames)),
        min_match_iou=float(np.clip(config.min_match_iou, 0.0, 1.0)),
        max_center_distance_ratio=max(0.01, float(config.max_center_distance_ratio)),
    )


def _box_area(box: FaceBox) -> int:
    return max(0, box.x2 - box.x1) * max(0, box.y2 - box.y1)


def _box_iou(a: FaceBox, b: FaceBox) -> float:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter <= 0:
        return 0.0
    union = _box_area(a) + _box_area(b) - inter
    return float(inter / union) if union > 0 else 0.0


def _box_center_distance_ratio(a: FaceBox, b: FaceBox) -> float:
    ax = (a.x1 + a.x2) * 0.5
    ay = (a.y1 + a.y2) * 0.5
    bx = (b.x1 + b.x2) * 0.5
    by = (b.y1 + b.y2) * 0.5
    dist = float(np.hypot(ax - bx, ay - by))
    scale = float(max(a.x2 - a.x1, a.y2 - a.y1, b.x2 - b.x1, b.y2 - b.y1, 1))
    return dist / scale


def _prediction_from_components(age_probs: np.ndarray, female_prob: float, emotion_probs: np.ndarray) -> FacePrediction:
    age_probs = np.asarray(age_probs, dtype=np.float32)
    emotion_probs = np.asarray(emotion_probs, dtype=np.float32)
    female_prob = float(female_prob)

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


def _new_track(track_id: int, box: FaceBox, frame_idx: int, smoothing_window: int) -> _TrackedFace:
    return _TrackedFace(
        track_id=track_id,
        last_box=box,
        last_seen_frame_idx=frame_idx,
        age_probs_history=deque(maxlen=smoothing_window),
        female_prob_history=deque(maxlen=smoothing_window),
        emotion_probs_history=deque(maxlen=smoothing_window),
    )


def _append_prediction(track: _TrackedFace, prediction: FacePrediction) -> None:
    track.age_probs_history.append(
        np.asarray([prediction.probs["age"][label] for label in AGE_GROUP_LABELS], dtype=np.float32)
    )
    track.female_prob_history.append(float(prediction.probs["gender"]["female"]))
    track.emotion_probs_history.append(
        np.asarray([prediction.probs["emotion"][label] for label in EMOTION_LABELS], dtype=np.float32)
    )


def _smoothed_prediction(track: _TrackedFace) -> FacePrediction:
    if not track.age_probs_history or not track.emotion_probs_history or not track.female_prob_history:
        raise ValueError("Track history is empty.")

    age_probs = np.mean(np.stack(list(track.age_probs_history), axis=0), axis=0)
    female_prob = float(np.mean(np.asarray(track.female_prob_history, dtype=np.float32)))
    emotion_probs = np.mean(np.stack(list(track.emotion_probs_history), axis=0), axis=0)
    return _prediction_from_components(age_probs, female_prob, emotion_probs)


def _track_result(track_id: int, box: FaceBox, prediction: FacePrediction) -> dict[str, Any]:
    return {
        "track_id": track_id,
        "box": {"x1": box.x1, "y1": box.y1, "x2": box.x2, "y2": box.y2, "score": box.score},
        "prediction": {
            "age_group": prediction.age_group,
            "gender": prediction.gender,
            "emotion": prediction.emotion,
            "probs": prediction.probs,
        },
    }


def _load_mediapipe_detector(cfg: TrainConfig) -> dict[str, Any]:
    import mediapipe as mp

    if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_detection"):
        detector = mp.solutions.face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=cfg.detector_confidence,
        )
        return {"backend": "mediapipe", "detector": detector}

    try:
        from mediapipe.python.solutions.face_detection import FaceDetection

        detector = FaceDetection(
            model_selection=0,
            min_detection_confidence=cfg.detector_confidence,
        )
        return {"backend": "mediapipe", "detector": detector}
    except Exception as exc:
        raise ImportError("Mediapipe face detection API is unavailable.") from exc


def _load_opencv_haar_detector() -> dict[str, Any]:
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        raise ImportError(f"Failed to load OpenCV Haar cascade: {cascade_path}")
    return {"backend": "opencv_haar", "detector": detector}


def load_detector() -> object:
    """Create and return a face detector.

    Priority:
    1) MediaPipe (if available and compatible)
    2) OpenCV Haar cascade fallback
    """

    cfg = TrainConfig()

    try:
        return _load_mediapipe_detector(cfg)
    except Exception:
        return _load_opencv_haar_detector()


def _detect_faces_mediapipe(image: np.ndarray, detector_obj) -> list[FaceBox]:
    h, w = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = detector_obj.process(rgb)
    detections: list[FaceBox] = []
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


def _detect_faces_opencv_haar(image: np.ndarray, detector_obj) -> list[FaceBox]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    boxes = detector_obj.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    detections: list[FaceBox] = []
    for (x, y, w, h) in boxes:
        x1 = int(max(0, x))
        y1 = int(max(0, y))
        x2 = int(max(0, x + w))
        y2 = int(max(0, y + h))
        if x2 <= x1 or y2 <= y1:
            continue
        detections.append(FaceBox(x1=x1, y1=y1, x2=x2, y2=y2, score=1.0))
    return detections


def detect_faces_bgr(image: np.ndarray, detector) -> list[FaceBox]:
    """Detect faces in BGR image and return absolute pixel boxes."""

    if image is None or image.size == 0:
        return []

    if isinstance(detector, dict):
        backend = detector.get("backend")
        detector_obj = detector.get("detector")
        if backend == "mediapipe":
            return _detect_faces_mediapipe(image, detector_obj)
        if backend == "opencv_haar":
            return _detect_faces_opencv_haar(image, detector_obj)

    # Backward compatibility with old direct MediaPipe detector object.
    return _detect_faces_mediapipe(image, detector)


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
    return _prediction_from_components(age_probs, female_prob, emotion_probs)


def _match_track(box: FaceBox, tracks: dict[int, _TrackedFace], cfg: RealtimeConfig, used_track_ids: set[int]) -> int | None:
    best_track_id: int | None = None
    best_score = float("-inf")

    for track_id, track in tracks.items():
        if track_id in used_track_ids:
            continue

        iou = _box_iou(box, track.last_box)
        center_ratio = _box_center_distance_ratio(box, track.last_box)
        if iou < cfg.min_match_iou and center_ratio > cfg.max_center_distance_ratio:
            continue

        score = iou - 0.1 * center_ratio
        if score > best_score:
            best_score = score
            best_track_id = track_id

    return best_track_id


def _drop_stale_tracks(state: RealtimeState, cfg: RealtimeConfig) -> None:
    stale_ids = [
        track_id
        for track_id, track in state.tracks.items()
        if state.frame_idx - track.last_seen_frame_idx > cfg.max_track_lost_frames
    ]
    for track_id in stale_ids:
        state.tracks.pop(track_id, None)


def process_realtime_frame(
    frame_bgr: np.ndarray,
    detector,
    ag_model,
    emo_model,
    state: RealtimeState | None = None,
    config: RealtimeConfig | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]], RealtimeState]:
    """Realtime frame inference with lightweight tracking and smoothing."""

    if frame_bgr is None or frame_bgr.size == 0:
        raise ValueError("Input frame is empty.")

    runtime_state = state if state is not None else RealtimeState()
    cfg = _clamp_realtime_config(config)
    runtime_state.frame_idx += 1
    _drop_stale_tracks(runtime_state, cfg)

    should_infer = (
        runtime_state.frame_idx == 1
        or runtime_state.frame_idx % cfg.inference_stride == 0
        or not runtime_state.latest_detections
    )

    if not should_infer:
        annotated = annotate_image(frame_bgr, runtime_state.latest_detections)
        return annotated, list(runtime_state.latest_results), runtime_state

    face_boxes = detect_faces_bgr(frame_bgr, detector)
    used_track_ids: set[int] = set()
    current_detections: list[tuple[FaceBox, FacePrediction]] = []
    current_results: list[dict[str, Any]] = []

    for box in face_boxes:
        crop = frame_bgr[box.y1 : box.y2, box.x1 : box.x2]
        if crop.size == 0:
            continue

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        raw_prediction = predict_face(crop_rgb, ag_model=ag_model, emo_model=emo_model)

        matched_track_id = _match_track(box, runtime_state.tracks, cfg, used_track_ids)
        if matched_track_id is None:
            matched_track_id = runtime_state.next_track_id
            runtime_state.next_track_id += 1
            runtime_state.tracks[matched_track_id] = _new_track(
                track_id=matched_track_id,
                box=box,
                frame_idx=runtime_state.frame_idx,
                smoothing_window=cfg.smoothing_window,
            )

        used_track_ids.add(matched_track_id)
        track = runtime_state.tracks[matched_track_id]
        track.last_box = box
        track.last_seen_frame_idx = runtime_state.frame_idx
        _append_prediction(track, raw_prediction)
        smooth_prediction = _smoothed_prediction(track)

        current_detections.append((box, smooth_prediction))
        current_results.append(_track_result(matched_track_id, box, smooth_prediction))

    runtime_state.latest_detections = current_detections
    runtime_state.latest_results = current_results

    annotated = annotate_image(frame_bgr, current_detections)
    return annotated, list(current_results), runtime_state


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

    runtime_state = RealtimeState()
    runtime_config = RealtimeConfig(inference_stride=1, smoothing_window=1)

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        frame_out, _, runtime_state = process_realtime_frame(
            frame_bgr=frame,
            detector=detector,
            ag_model=ag_model,
            emo_model=emo_model,
            state=runtime_state,
            config=runtime_config,
        )
        writer.write(frame_out)

    capture.release()
    writer.release()
    return str(output_path)
