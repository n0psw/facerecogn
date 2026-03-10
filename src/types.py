"""Common typed structures shared across modules."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FaceBox:
    """Absolute bounding box in image pixel coordinates."""

    x1: int
    y1: int
    x2: int
    y2: int
    score: float


@dataclass(frozen=True)
class FacePrediction:
    """Predicted attributes for one detected face."""

    age_group: str
    gender: str
    emotion: str
    probs: dict

