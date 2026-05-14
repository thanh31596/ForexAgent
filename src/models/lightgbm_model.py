"""LightGBM regressor behind :class:`~src.models.base.ClassicalModel`."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import (
    LGBM_EARLY_STOPPING_ROUNDS,
    LGBM_NUM_BOOST_ROUND,
    LGBM_PARAMS,
    RANDOM_SEED,
)
from src.models.base import ClassicalModel


class LightGBMForexModel(ClassicalModel):
    name: ClassVar[str] = "lightgbm"

    def __init__(self) -> None:
        self._booster: lgb.Booster | None = None
        self._feature_cols: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._feature_cols = list(X.columns)
        params = dict(LGBM_PARAMS)
        params["seed"] = RANDOM_SEED
        if len(X) < 80:
            dtrain = lgb.Dataset(X, label=y)
            self._booster = lgb.train(
                params,
                dtrain,
                num_boost_round=min(LGBM_NUM_BOOST_ROUND, 200),
            )
            return
        cut = max(int(len(X) * 0.85), len(X) - 50)
        X_tr, y_tr = X.iloc[:cut], y.iloc[:cut]
        X_val, y_val = X.iloc[cut:], y.iloc[cut:]
        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval = lgb.Dataset(X_val, label=y_val)
        self._booster = lgb.train(
            params,
            dtrain,
            num_boost_round=LGBM_NUM_BOOST_ROUND,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(LGBM_EARLY_STOPPING_ROUNDS, verbose=False)],
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._booster is None:
            raise RuntimeError("Model not fitted")
        return np.asarray(self._booster.predict(X[self._feature_cols]), dtype=np.float64)

    def save(self, path: Path) -> None:
        if self._booster is None:
            raise RuntimeError("Model not fitted")
        path.parent.mkdir(parents=True, exist_ok=True)
        self._booster.save_model(str(path))

    @classmethod
    def load(cls, path: Path) -> LightGBMForexModel:
        m = cls()
        m._booster = lgb.Booster(model_file=str(path))
        m._feature_cols = list(m._booster.feature_name() or [])
        return m

    def feature_importance(self) -> dict[str, float] | None:
        if self._booster is None:
            return None
        names = self._booster.feature_name() or self._feature_cols
        imp = self._booster.feature_importance(importance_type="gain")
        return {n: float(i) for n, i in zip(names, imp, strict=True)}
