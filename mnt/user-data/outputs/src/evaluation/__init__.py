"""Automated evaluation framework.

Modules
-------
evaluator   Rolling-window metric computation (directional accuracy,
            precision/recall, Sharpe ratio) and comparison against
            baselines (buy-and-hold, MA-crossover, random). Emits
            JSON / Markdown reports.
drift       Drift detection via PSI on feature distributions and
            relative degradation of rolling Sharpe versus a historical
            reference window.
"""
