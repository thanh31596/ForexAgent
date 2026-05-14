"""Population stability index (PSI) + Sharpe degradation helpers."""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from src.config import DRIFT_PSI_THRESHOLD, DRIFT_SHARPE_DEGRADATION_PCT


def psi(expected: np.ndarray, actual: np.ndarray, *, buckets: int = 10) -> float:
    """Compute PSI between two 1-D samples using quantile buckets on ``expected``."""
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]
    if len(expected) < buckets * 5 or len(actual) < buckets * 5:
        return 0.0
    qs = np.quantile(expected, np.linspace(0, 1, buckets + 1))
    qs[0] = -np.inf
    qs[-1] = np.inf
    exp_counts, _ = np.histogram(expected, bins=qs)
    act_counts, _ = np.histogram(actual, bins=qs)
    exp_pct = np.clip(exp_counts / exp_counts.sum(), 1e-6, 1.0)
    act_pct = np.clip(act_counts / act_counts.sum(), 1e-6, 1.0)
    return cast(float, np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def sharpe_degraded(ref_returns: np.ndarray, recent_returns: np.ndarray) -> bool:
    """True if recent Sharpe < (1 - threshold) * reference Sharpe (both annualized)."""
    ref_returns = ref_returns[np.isfinite(ref_returns)]
    recent_returns = recent_returns[np.isfinite(recent_returns)]
    if len(ref_returns) < 10 or len(recent_returns) < 10:
        return False

    def _sharpe(r: np.ndarray) -> float:
        mu, sig = float(np.mean(r)), float(np.std(r, ddof=1))
        if sig == 0.0:
            return 0.0
        return float((mu / sig) * np.sqrt(252.0))

    s_ref = _sharpe(ref_returns)
    s_new = _sharpe(recent_returns)
    if s_ref <= 0:
        return s_new < 0
    return s_new < (1.0 - DRIFT_SHARPE_DEGRADATION_PCT) * s_ref


def drift_flags(feature_df: pd.DataFrame, *, ref_rows: int = 500) -> dict[str, bool | float]:
    """Cheap drift screen: max PSI across numeric columns + Sharpe on ``Close`` returns."""
    if len(feature_df) < ref_rows + 50:
        return {"drift_detected": False, "max_psi": 0.0, "sharpe_degraded": False}
    ref = feature_df.iloc[:ref_rows]
    recent = feature_df.iloc[-ref_rows:]
    psis: list[float] = []
    for col in feature_df.select_dtypes(include=[np.number]).columns:
        if col.lower() == "y":
            continue
        psis.append(psi(ref[col].to_numpy(), recent[col].to_numpy()))
    max_psi = float(max(psis) if psis else 0.0)
    close = feature_df["Close"].pct_change().dropna().to_numpy()
    sharpe_bad = sharpe_degraded(close[:ref_rows], close[-ref_rows:])
    drift = max_psi > DRIFT_PSI_THRESHOLD or sharpe_bad
    return {"drift_detected": drift, "max_psi": max_psi, "sharpe_degraded": sharpe_bad}
