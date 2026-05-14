"""Rolling metrics, baselines, and Markdown/JSON reporting."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from src.config import (
    EVAL_ROLLING_WINDOW,
    EVAL_TRADING_COST_BPS,
    EVAL_TRADING_DAYS_PER_YEAR,
    METRICS_DIR,
    PREDICTION_HORIZON,
)
from src.data.features import add_features
from src.evaluation.drift import drift_flags
from src.observability.logger import get_logger
from src.observability.metrics import append_metric

_log = get_logger(__name__)


@dataclass
class BaselineResult:
    name: str
    sharpe: float
    cumulative_return: float


def _cost_adjusted_returns(signal: np.ndarray, fwd_ret: np.ndarray) -> np.ndarray:
    cost = EVAL_TRADING_COST_BPS / 10000.0
    pos = np.clip(signal, -1.0, 1.0)
    turnover = np.abs(np.diff(pos, prepend=0.0))
    gross = pos * fwd_ret
    out = gross - turnover * cost
    return np.asarray(out, dtype=np.float64)


def buy_and_hold_returns(fwd_ret: np.ndarray) -> np.ndarray:
    return _cost_adjusted_returns(np.ones_like(fwd_ret), fwd_ret)


def ma_crossover_returns(close: pd.Series, fwd_ret: np.ndarray) -> np.ndarray:
    fast = close.rolling(10).mean()
    slow = close.rolling(50).mean()
    sig = np.where(fast > slow, 1.0, -1.0)[-len(fwd_ret) :]
    return _cost_adjusted_returns(sig, fwd_ret)


def random_returns(fwd_ret: np.ndarray, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sig = rng.choice([-1.0, 0.0, 1.0], size=len(fwd_ret), replace=True)
    return _cost_adjusted_returns(sig, fwd_ret)


def sharpe(returns: np.ndarray) -> float:
    r = returns[np.isfinite(returns)]
    if len(r) < 5:
        return 0.0
    mu, sig = float(np.mean(r)), float(np.std(r, ddof=1))
    if sig == 0.0:
        return 0.0
    return float((mu / sig) * np.sqrt(float(EVAL_TRADING_DAYS_PER_YEAR)))


def cumulative(returns: np.ndarray) -> float:
    r = returns[np.isfinite(returns)]
    if len(r) == 0:
        return 0.0
    return float(np.prod(1.0 + r) - 1.0)


def evaluate_frame(df: pd.DataFrame) -> dict[str, object]:
    feats = add_features(df).dropna()
    close = feats["Close"]
    fwd = close.pct_change(PREDICTION_HORIZON).shift(-PREDICTION_HORIZON).to_numpy()
    fwd = fwd[~np.isnan(fwd)]
    n = len(fwd)
    if n < EVAL_ROLLING_WINDOW:
        raise ValueError("Not enough rows for evaluation")
    close_tail = close.iloc[-n:]
    bh = buy_and_hold_returns(fwd)
    ma = ma_crossover_returns(close_tail, fwd)
    rnd = random_returns(fwd)
    baselines = [
        BaselineResult("buy_and_hold", sharpe(bh), cumulative(bh)),
        BaselineResult("ma_crossover", sharpe(ma), cumulative(ma)),
        BaselineResult("random", sharpe(rnd), cumulative(rnd)),
    ]
    drift = drift_flags(feats.select_dtypes(include=[np.number]))
    report = {
        "n": n,
        "rolling_window": EVAL_ROLLING_WINDOW,
        "baselines": [asdict(b) for b in baselines],
        "drift": drift,
    }
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / "evaluation_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    md = ["# Evaluation report", "", "## Baselines", ""]
    for b in baselines:
        md.append(f"- **{b.name}**: Sharpe={b.sharpe:.3f}, cum_ret={b.cumulative_return:.3f}")
    drift_lines = [
        "",
        "## Drift",
        "",
        f"- drift_detected: {drift['drift_detected']}",
        f"- max_psi: {drift['max_psi']:.4f}",
        "",
    ]
    md.extend(drift_lines)
    (METRICS_DIR / "evaluation_report.md").write_text("\n".join(md), encoding="utf-8")
    append_metric({"event": "evaluation", **report})
    _log.info("evaluation_complete", extra={"path": str(METRICS_DIR / "evaluation_report.json")})
    return report


def main() -> None:
    from src.config import CurrencyPair, DataInterval
    from src.data.fetcher import fetch_and_persist

    df = fetch_and_persist(CurrencyPair.EUR_USD.value, DataInterval.ONE_DAY.value, days=800)
    evaluate_frame(df)


if __name__ == "__main__":
    main()
