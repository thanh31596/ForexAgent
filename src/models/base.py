"""Abstract classical model interface (the ML ``BaseModel`` from the architecture brief)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd


class ClassicalModel(ABC):
    """Minimal contract for LightGBM / LSTM (and future) predictors."""

    name: ClassVar[str]

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train on aligned feature matrix and target."""

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return point predictions (same length as ``X``)."""

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist to ``path`` (file or directory)."""

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> ClassicalModel:
        """Load from ``path``."""

    def feature_importance(self) -> dict[str, float] | None:
        """Optional diagnostic hook for tree models."""
        return None
