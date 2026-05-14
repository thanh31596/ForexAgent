"""Persistence for OHLCV bars (SQLite dev, Postgres in compose)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, cast

import pandas as pd
from sqlalchemy import DateTime, Float, String, UniqueConstraint, create_engine, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from src.config import settings
from src.observability.logger import get_logger

_log = get_logger(__name__)


class Base(DeclarativeBase):
    pass


class OhlcvBar(Base):
    __tablename__ = "ohlcv_bars"
    __table_args__ = (UniqueConstraint("pair", "interval", "ts", name="uq_bar"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    pair: Mapped[str] = mapped_column(String(32), index=True)
    interval: Mapped[str] = mapped_column(String(8), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=False), index=True)
    open_: Mapped[float] = mapped_column("open", Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    volume: Mapped[float] = mapped_column(Float, default=0.0)
    source: Mapped[str] = mapped_column(String(32))
    stale: Mapped[str] = mapped_column(String(8), default="false")


_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = create_engine(settings.database_url, future=True)
    return _engine


def get_session() -> Session:
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(
            bind=get_engine(), autoflush=False, autocommit=False, future=True
        )
    return _session_factory()


def init_db() -> None:
    Base.metadata.create_all(bind=get_engine())
    _log.info("db_init", extra={"url_prefix": settings.database_url.split(":", 1)[0]})


def _normalize_ts(ts: object) -> datetime:
    t = pd.Timestamp(cast(Any, ts))
    if t.tzinfo is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    return t.to_pydatetime()


def upsert_ohlcv(df: pd.DataFrame, *, pair: str, interval: str, source: str, stale: bool) -> int:
    """Upsert bars from a DataFrame with yfinance-style columns and DatetimeIndex."""
    if df.empty:
        return 0
    init_db()
    rows = 0
    stale_s = "true" if stale else "false"
    with get_session() as session:
        for ts, row in df.iterrows():
            ts_dt = _normalize_ts(ts)
            stmt = select(OhlcvBar).where(
                OhlcvBar.pair == pair,
                OhlcvBar.interval == interval,
                OhlcvBar.ts == ts_dt,
            )
            existing = session.scalars(stmt).first()
            vol = float(row["Volume"]) if "Volume" in row and pd.notna(row["Volume"]) else 0.0
            if existing:
                existing.open_ = float(row["Open"])
                existing.high = float(row["High"])
                existing.low = float(row["Low"])
                existing.close = float(row["Close"])
                existing.volume = vol
                existing.source = source
                existing.stale = stale_s
            else:
                session.add(
                    OhlcvBar(
                        pair=pair,
                        interval=interval,
                        ts=ts_dt,
                        open_=float(row["Open"]),
                        high=float(row["High"]),
                        low=float(row["Low"]),
                        close=float(row["Close"]),
                        volume=vol,
                        source=source,
                        stale=stale_s,
                    )
                )
            rows += 1
        session.commit()
    return rows


def load_ohlcv(pair: str, interval: str) -> pd.DataFrame:
    """Load stored bars as OHLCV with DatetimeIndex (naive local)."""
    init_db()
    with get_session() as session:
        stmt = (
            select(OhlcvBar)
            .where(OhlcvBar.pair == pair, OhlcvBar.interval == interval)
            .order_by(OhlcvBar.ts)
        )
        bars = session.scalars(stmt).all()
    if not bars:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    records: list[dict[str, Any]] = []
    for b in bars:
        records.append(
            {
                "Datetime": b.ts,
                "Open": b.open_,
                "High": b.high,
                "Low": b.low,
                "Close": b.close,
                "Volume": b.volume,
            }
        )
    out = pd.DataFrame.from_records(records, index="Datetime")
    out.index = pd.to_datetime(out.index)
    return out
