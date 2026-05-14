"""Walk-forward training orchestration + MLflow tracking."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from src.config import (
    DEFAULT_BACKFILL_DAYS,
    LSTM_SEQUENCE_LENGTH,
    MLFLOW_EXPERIMENT_NAME,
    MODEL_DIR,
    WALK_FORWARD_GAP,
    WALK_FORWARD_N_SPLITS,
    WALK_FORWARD_STRATEGY,
    WALK_FORWARD_TEST_SIZE,
    CurrencyPair,
    DataInterval,
    settings,
)
from src.data.features import add_features, feature_columns, walk_forward_indices
from src.data.fetcher import fetch_and_persist
from src.models.base import ClassicalModel
from src.models.lightgbm_model import LightGBMForexModel
from src.models.lstm_model import LSTMForexModel
from src.observability.logger import get_logger
from src.observability.metrics import latency_block

_log = get_logger(__name__)

BEST_MODEL_META = MODEL_DIR / "best_model.json"


@dataclass
class FoldMetrics:
    rmse: float
    dir_acc: float


def _directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() == 0:
        return 0.0
    yt = y_true[mask]
    yp = y_pred[mask]
    return float(np.mean(np.sign(yt) == np.sign(yp)))


def _evaluate(model: ClassicalModel, X: pd.DataFrame, y: pd.Series) -> FoldMetrics:
    pred = model.predict(X)
    m = mean_squared_error(y, pred)
    rmse = float(np.sqrt(m))
    dir_acc = _directional_accuracy(y.to_numpy(), pred)
    return FoldMetrics(rmse=rmse, dir_acc=dir_acc)


def walk_forward_train(
    df: pd.DataFrame,
    model_factory: type[ClassicalModel],
) -> tuple[ClassicalModel, dict[str, Any]]:
    feats = add_features(df)
    feats = feats.dropna()
    cols = feature_columns(feats)
    X_full = feats[cols]
    y_full = feats["y"]
    n = len(feats)
    test_size = min(WALK_FORWARD_TEST_SIZE, max(50, n // 4))
    fold_scores: list[FoldMetrics] = []
    last_trained: ClassicalModel | None = None

    for train_idx, test_idx in walk_forward_indices(
        n,
        n_splits=WALK_FORWARD_N_SPLITS,
        test_size=test_size,
        gap=WALK_FORWARD_GAP,
        strategy=WALK_FORWARD_STRATEGY,
    ):
        X_tr, y_tr = X_full.iloc[train_idx], y_full.iloc[train_idx]
        X_te, y_te = X_full.iloc[test_idx], y_full.iloc[test_idx]
        if len(X_tr) < 50 or len(X_te) < 10:
            continue
        m = model_factory()
        if m.name == "lstm" and len(X_tr) < LSTM_SEQUENCE_LENGTH + 30:
            continue
        try:
            m.fit(X_tr, y_tr)
        except Exception as exc:  # noqa: BLE001
            _log.warning("model_fit_failed", extra={"model": m.name, "error": repr(exc)})
            continue
        fold_scores.append(_evaluate(m, X_te, y_te))
        last_trained = m

    if last_trained is None:
        cut = max(int(0.75 * n), 50)
        if n - cut < 10 or cut < 30:
            raise RuntimeError(
                "Not enough rows after feature engineering for a minimal train/val split"
            )
        X_tr, y_tr = X_full.iloc[:cut], y_full.iloc[:cut]
        X_te, y_te = X_full.iloc[cut:], y_full.iloc[cut:]
        m = model_factory()
        if m.name == "lstm" and len(X_tr) < LSTM_SEQUENCE_LENGTH + 30:
            raise RuntimeError("Not enough rows for LSTM training")
        m.fit(X_tr, y_tr)
        fold_scores = [_evaluate(m, X_te, y_te)]
        last_trained = m

    # Refit on full sample for serving artifact
    final = model_factory()
    final.fit(X_full, y_full)
    metrics = {
        "folds": len(fold_scores),
        "mean_rmse": float(np.mean([f.rmse for f in fold_scores])) if fold_scores else None,
        "mean_dir_acc": float(np.mean([f.dir_acc for f in fold_scores])) if fold_scores else None,
    }
    return final, metrics


def train_all(pair: str, interval: str, *, days: int) -> Path:
    with latency_block("train.fetch", extra={"pair": pair}):
        raw = fetch_and_persist(pair, interval, days=days)
    if raw.empty:
        raise RuntimeError("No market data available — check network or DB")

    results: dict[str, Any] = {}
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    best_name: str | None = None
    best_score = -1.0
    best_path: Path | None = None

    for factory in (LightGBMForexModel, LSTMForexModel):
        name = factory.name
        try:
            with mlflow.start_run(run_name=f"{pair}-{name}"):
                model, metrics = walk_forward_train(raw, factory)
                mlflow.log_params({"pair": pair, "interval": interval, "model": name})
                mlflow.log_metrics(
                    {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
                )
                out_path = MODEL_DIR / f"{name}_{pair.replace('=', '_')}.artifact"
                if name == "lightgbm":
                    model.save(out_path)
                else:
                    model.save(out_path)
                mlflow.log_artifact(str(out_path))
                score = float(metrics.get("mean_dir_acc") or 0.0)
                results[name] = {"metrics": metrics, "artifact": str(out_path)}
                if score >= best_score:
                    best_score = score
                    best_name = name
                    best_path = out_path
        except Exception as exc:  # noqa: BLE001
            _log.error("train_run_failed", extra={"model": name, "error": repr(exc)})
            results[name] = {"error": repr(exc)}

    if not best_name or not best_path:
        raise RuntimeError("Training failed for all models")

    BEST_MODEL_META.write_text(
        json.dumps({"model": best_name, "path": str(best_path), "pair": pair}, indent=2),
        encoding="utf-8",
    )
    _log.info("train_complete", extra={"best": best_name, "path": str(best_path)})
    return best_path


def main() -> None:
    p = argparse.ArgumentParser(description="Train ForexAgent classical models")
    p.add_argument("--pair", default=CurrencyPair.EUR_USD.value)
    p.add_argument("--interval", default=DataInterval.ONE_DAY.value)
    p.add_argument("--days", type=int, default=DEFAULT_BACKFILL_DAYS)
    args = p.parse_args()
    train_all(args.pair, args.interval, days=args.days)


if __name__ == "__main__":
    main()
