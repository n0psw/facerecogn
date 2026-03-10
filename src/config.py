"""Central configuration for the face analysis project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


SEED = 42

# Age bins are inclusive ranges and the last bin is open-ended with upper bound 200.
AGE_BINS: list[tuple[int, int]] = [
    (0, 9),
    (10, 19),
    (20, 29),
    (30, 39),
    (40, 49),
    (50, 59),
    (60, 69),
    (70, 200),
]

AGE_GROUP_LABELS: list[str] = ["0-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]
GENDER_LABELS: list[str] = ["male", "female"]
EMOTION_LABELS: list[str] = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


@dataclass(frozen=True)
class TrainConfig:
    """Training defaults optimized for Colab free GPU usage."""

    batch_size: int = 64
    age_gender_epochs: int = 30
    emotion_epochs: int = 40
    initial_lr: float = 1e-3
    early_stopping_patience: int = 5
    reduce_lr_patience: int = 2
    reduce_lr_factor: float = 0.5
    min_lr: float = 1e-6
    detector_confidence: float = 0.5


@dataclass(frozen=True)
class PathConfig:
    """Default path layout used by notebook and modules."""

    project_root: Path = Path(".")
    data_root: Path = Path("./data")
    artifacts_root: Path = Path("./artifacts")
    checkpoints_root: Path = Path("./artifacts/checkpoints")
    logs_root: Path = Path("./artifacts/logs")

    utkface_dirname: str = "utkface"
    fer_csv_relpath: str = "fer2013/fer2013.csv"

    def utkface_path(self) -> Path:
        return self.data_root / self.utkface_dirname

    def fer_csv_path(self) -> Path:
        return self.data_root / self.fer_csv_relpath

