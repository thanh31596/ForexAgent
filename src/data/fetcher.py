"""Market data fetchers: Alpha Vantage primary, yfinance fallback."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from typing import Any, cast

import pandas as pd
import requests
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import DEFAULT_BACKFILL_DAYS, settings
from src.data.db import load_ohlcv, upsert_ohlcv
from src.observability.logger import get_logger
from src.observability.metrics import latency_block

_log = get_logger(__name__)


class MarketDataFetcher(ABC):
    """Abstract market-data backend."""

    name: str

    @abstractmethod
    def fetch_history(self, pair: str, interval: str, *, days: int) -> pd.DataFrame:
        """Return OHLCV with yfinance-style columns and DatetimeIndex."""


class YFinanceFetcher(MarketDataFetcher):
    name = "yfinance"

    def fetch_history(self, pair: str, interval: str, *, days: int) -> pd.DataFrame:
        with latency_block("data.yfinance.fetch", extra={"pair": pair, "interval": interval}):
            t = yf.Ticker(pair)
            df = t.history(period=f"{days}d", interval=interval, auto_adjust=False)
        if df.empty:
            _log.warning("yfinance_empty", extra={"pair": pair})
            return cast(pd.DataFrame, df)
        cols_of_interest = ["Open", "High", "Low", "Close", "Volume"]
        if any(c not in df.columns for c in cols_of_interest):
            _log.warning("yfinance_missing_cols", extra={"cols": list(df.columns)})
            return pd.DataFrame()
        out = cast(pd.DataFrame, df[cols_of_interest].copy())
        idx = out.index
        if isinstance(idx, pd.DatetimeIndex) and idx.tz is not None:
            out = out.copy()
            tz_idx: pd.DatetimeIndex = idx
            out.index = tz_idx.tz_convert("UTC").tz_localize(None)
        return out


class AlphaVantageFetcher(MarketDataFetcher):
    name = "alpha_vantage"

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    @staticmethod
    def _av_symbol(pair: str) -> str:
        return pair.replace("=X", "").replace("/", "")

    @retry(wait=wait_exponential(multiplier=1, min=2, max=30), stop=stop_after_attempt(4))
    def _get(self, params: dict[str, Any]) -> dict[str, Any]:
        r = requests.get(
            "https://www.alphavantage.co/query",
            params=params,
            timeout=30,
        )
        r.raise_for_status()
        return cast(dict[str, Any], r.json())

    def fetch_history(self, pair: str, interval: str, *, days: int) -> pd.DataFrame:
        sym = self._av_symbol(pair)
        if len(sym) < 6:
            return pd.DataFrame()
        from_s, to_s = sym[:3], sym[3:6]
        is_daily = interval == "1d"
        fn = "FX_DAILY" if is_daily else "FX_INTRADAY"
        params: dict[str, Any] = {
            "function": fn,
            "from_symbol": from_s,
            "to_symbol": to_s,
            "apikey": self._api_key,
        }
        if not is_daily:
            params["interval"] = "60min"
            params["outputsize"] = "full"
        else:
            params["outputsize"] = "full"
        params = {k: v for k, v in params.items() if v is not None}
        with latency_block("data.alphavantage.fetch", extra={"pair": pair}):
            raw = self._get(params)
        key = "Time Series FX (Daily)" if is_daily else "Time Series FX (60min)"
        series = raw.get(key)
        if not isinstance(series, dict):
            _log.warning("alphavantage_parse_fail", extra={"keys": list(raw.keys())})
            return pd.DataFrame()
        rows = []
        for ts, ohlc in series.items():
            rows.append(
                {
                    "Datetime": pd.Timestamp(ts),
                    "Open": float(ohlc["1. open"]),
                    "High": float(ohlc["2. high"]),
                    "Low": float(ohlc["3. low"]),
                    "Close": float(ohlc["4. close"]),
                    "Volume": 0.0,
                }
            )
        df = pd.DataFrame(rows).set_index("Datetime").sort_index()
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).replace(tzinfo=None)
        df = df[df.index >= pd.Timestamp(cutoff)]
        return df


def get_default_fetcher() -> MarketDataFetcher:
    key = settings.alpha_vantage_api_key
    if key is not None and (secret := key.get_secret_value()):
        return AlphaVantageFetcher(secret)
    return YFinanceFetcher()


def fetch_and_persist(
    pair: str,
    interval: str,
    *,
    days: int = DEFAULT_BACKFILL_DAYS,
    stale: bool = False,
) -> pd.DataFrame:
    """Fetch via Alpha Vantage if configured, else yfinance; persist; return merged frame."""
    fetcher = get_default_fetcher()
    try:
        remote = fetcher.fetch_history(pair, interval, days=days)
        source = fetcher.name
    except Exception as exc:  # noqa: BLE001
        _log.warning("fetch_failed_use_db", extra={"error": repr(exc)})
        remote = pd.DataFrame()
        source = "cache"
        stale = True

    if not remote.empty:
        upsert_ohlcv(remote, pair=pair, interval=interval, source=source, stale=stale)

    local = load_ohlcv(pair, interval)
    if remote.empty and local.empty:
        return remote
    if remote.empty:
        return local
    if local.empty:
        return remote
    return remote.combine_first(local).sort_index()
