"""Market-data ingestion and feature engineering.

Modules
-------
fetcher    Abstract ``MarketDataFetcher`` interface with Alpha Vantage and
           yfinance backends; handles retries, rate limits, and caching.
features   Pure functions for technical indicators (RSI, MACD, Bollinger,
           moving averages, volatility, ATR) and walk-forward splits.
"""
