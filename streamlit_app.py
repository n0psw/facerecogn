"""Streamlit UI for realtime face analysis (age, gender, emotion)."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer

from src.infer import RealtimeConfig, RealtimeState, load_detector, process_realtime_frame


@st.cache_resource(show_spinner=False)
def load_runtime_models(age_gender_model_path: str, emotion_model_path: str):
    """Load models + detector once per distinct path pair."""

    import tensorflow as tf

    ag_model = tf.keras.models.load_model(age_gender_model_path)
    emo_model = tf.keras.models.load_model(emotion_model_path)
    detector = load_detector()
    return ag_model, emo_model, detector


def _valid_model_path(path_text: str) -> tuple[bool, str]:
    candidate = Path(path_text).expanduser()
    if not candidate.exists():
        return False, f"File not found: {candidate}"
    if candidate.suffix.lower() != ".keras":
        return False, "Expected a .keras checkpoint file."
    return True, str(candidate)


def _render_face_card(container, face_result: dict[str, Any]) -> None:
    pred = face_result["prediction"]
    box = face_result["box"]

    title = (
        f"Face #{face_result['track_id']} - "
        f"{pred['age_group']} | {pred['gender']} | {pred['emotion']}"
    )
    container.markdown(f"**{title}**")

    col1, col2 = container.columns(2)
    col1.write(f"Detection: {box['score']:.2f}")
    col2.write(
        "BBox: "
        f"({box['x1']}, {box['y1']}) - ({box['x2']}, {box['y2']})"
    )

    emotion_probs = pred["probs"]["emotion"]
    gender_probs = pred["probs"]["gender"]

    container.write("Emotion probabilities")
    container.json({k: round(float(v), 3) for k, v in emotion_probs.items()})

    container.write("Gender probabilities")
    container.json({k: round(float(v), 3) for k, v in gender_probs.items()})


st.set_page_config(
    page_title="Face Analysis Realtime",
    layout="wide",
)

st.title("Face Analysis System - Realtime Webcam")
st.caption("Live prediction of age group, gender, and emotion from webcam stream.")

with st.sidebar:
    st.header("Runtime Settings")
    age_gender_path = st.text_input(
        "Age/Gender model (.keras)",
        value="artifacts/checkpoints/age_gender/age_gender_best.keras",
    )
    emotion_path = st.text_input(
        "Emotion model (.keras)",
        value="artifacts/checkpoints/emotion/emotion_best.keras",
    )

    inference_stride = st.slider(
        "Inference stride (every N-th frame)",
        min_value=1,
        max_value=6,
        value=2,
        step=1,
    )
    smoothing_window = st.slider(
        "Smoothing window (frames)",
        min_value=1,
        max_value=12,
        value=5,
        step=1,
    )

    st.markdown("---")
    st.write("Tip: lower stride means more accuracy, higher means higher FPS.")

age_ok, age_resolved = _valid_model_path(age_gender_path)
emo_ok, emo_resolved = _valid_model_path(emotion_path)

if not age_ok:
    st.error(age_resolved)
if not emo_ok:
    st.error(emo_resolved)

if not (age_ok and emo_ok):
    st.info("Provide valid model paths to enable webcam inference.")
    st.stop()

try:
    ag_model, emo_model, detector = load_runtime_models(age_resolved, emo_resolved)
except Exception as exc:
    st.error(f"Failed to load models: {exc}")
    st.stop()

left_col, right_col = st.columns([2, 1], gap="large")

stats_placeholder = right_col.empty()
faces_metric_placeholder = right_col.empty()
latency_metric_placeholder = right_col.empty()
status_placeholder = right_col.empty()
faces_placeholder = right_col.empty()

shared = {
    "lock": threading.Lock(),
    "state": RealtimeState(),
    "latest_results": [],
    "processed_frames": 0,
    "last_infer_ms": 0.0,
    "last_error": None,
    "started_at": time.perf_counter(),
}

rt_config = RealtimeConfig(
    inference_stride=inference_stride,
    smoothing_window=smoothing_window,
)


def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    frame_bgr = frame.to_ndarray(format="bgr24")
    start_ts = time.perf_counter()

    with shared["lock"]:
        runtime_state = shared["state"]
        try:
            annotated, results, runtime_state = process_realtime_frame(
                frame_bgr=frame_bgr,
                detector=detector,
                ag_model=ag_model,
                emo_model=emo_model,
                state=runtime_state,
                config=rt_config,
            )
            error_text = None
        except Exception as exc:
            annotated = frame_bgr
            results = []
            error_text = str(exc)

        infer_ms = (time.perf_counter() - start_ts) * 1000.0
        shared["state"] = runtime_state
        shared["latest_results"] = results
        shared["processed_frames"] += 1
        shared["last_infer_ms"] = infer_ms
        shared["last_error"] = error_text

    return av.VideoFrame.from_ndarray(annotated, format="bgr24")


with left_col:
    webrtc_ctx = webrtc_streamer(
        key="face-analysis-realtime",
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with right_col:
    st.subheader("Live Results")

if webrtc_ctx.state.playing:
    status_placeholder.success("Webcam stream is running.")
else:
    status_placeholder.info("Click START on the webcam component to begin realtime analysis.")

if webrtc_ctx.state.playing:
    # Poll callback state while stream is active.
    while webrtc_ctx.state.playing:
        with shared["lock"]:
            results = list(shared["latest_results"])
            processed_frames = int(shared["processed_frames"])
            infer_ms = float(shared["last_infer_ms"])
            last_error = shared["last_error"]
            started_at = float(shared["started_at"])

        elapsed = max(time.perf_counter() - started_at, 1e-6)
        approx_fps = processed_frames / elapsed

        stats_placeholder.metric("Approx FPS", f"{approx_fps:.1f}")
        faces_metric_placeholder.metric("Faces detected", len(results))
        latency_metric_placeholder.metric("Last frame latency", f"{infer_ms:.1f} ms")

        if last_error:
            status_placeholder.error(f"Runtime error: {last_error}")
        else:
            status_placeholder.success("Inference running normally.")

        faces_placeholder.empty()
        with faces_placeholder.container():
            st.write("### Face Cards")
            if not results:
                st.write("No faces detected in the latest analyzed frame.")
            else:
                for face_result in results:
                    _render_face_card(st.container(), face_result)
                    st.markdown("---")

        time.sleep(0.35)
else:
    with faces_placeholder.container():
        st.write("### Face Cards")
        st.write("Start the stream to see per-face predictions.")
