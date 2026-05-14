"""Small PyTorch LSTM regressor behind :class:`~src.models.base.ClassicalModel`."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.config import (
    LSTM_BATCH_SIZE,
    LSTM_DROPOUT,
    LSTM_EPOCHS,
    LSTM_HIDDEN_SIZE,
    LSTM_LEARNING_RATE,
    LSTM_NUM_LAYERS,
    LSTM_PATIENCE,
    LSTM_SEQUENCE_LENGTH,
    RANDOM_SEED,
)
from src.models.base import ClassicalModel


class _LSTMNet(nn.Module):
    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=LSTM_HIDDEN_SIZE,
            num_layers=LSTM_NUM_LAYERS,
            batch_first=True,
            dropout=LSTM_DROPOUT if LSTM_NUM_LAYERS > 1 else 0.0,
        )
        self.head = nn.Linear(LSTM_HIDDEN_SIZE, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last: torch.Tensor = out[:, -1, :]
        logits: torch.Tensor = self.head(last).squeeze(-1)
        return logits


class LSTMForexModel(ClassicalModel):
    name: ClassVar[str] = "lstm"

    def __init__(self) -> None:
        self._net: _LSTMNet | None = None
        self._feature_cols: list[str] = []
        self._seq = LSTM_SEQUENCE_LENGTH

    def _make_sequences(
        self, X: pd.DataFrame, y: pd.Series | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        arr = X[self._feature_cols].to_numpy(dtype=np.float32)
        xs: list[np.ndarray] = []
        ys: list[float] = []
        for i in range(self._seq, len(arr)):
            xs.append(arr[i - self._seq : i])
            if y is not None:
                ys.append(float(y.iloc[i]))
        if not xs:
            return torch.zeros((0, self._seq, arr.shape[1])), None if y is None else torch.zeros(0)
        tx = torch.tensor(np.stack(xs), dtype=torch.float32)
        if y is None:
            return tx, None
        ty = torch.tensor(ys, dtype=torch.float32)
        return tx, ty

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        torch.manual_seed(RANDOM_SEED)
        self._feature_cols = list(X.columns)
        self._net = _LSTMNet(n_features=len(self._feature_cols))
        tx, ty = self._make_sequences(X, y)
        if ty is None:
            raise RuntimeError("internal: ty required for training")
        if len(tx) < 8:
            raise ValueError("Not enough rows to train LSTM for configured sequence length")
        opt = torch.optim.Adam(self._net.parameters(), lr=LSTM_LEARNING_RATE)
        loss_fn = nn.MSELoss()
        best = float("inf")
        bad = 0
        max_epochs = min(LSTM_EPOCHS, 12)
        patience = min(LSTM_PATIENCE, 4)
        for epoch in range(max_epochs):
            self._net.train()
            perm = torch.randperm(len(tx))
            total_loss = 0.0
            for start in range(0, len(tx), LSTM_BATCH_SIZE):
                idx = perm[start : start + LSTM_BATCH_SIZE]
                xb = tx[idx]
                yb = ty[idx]
                opt.zero_grad()
                pred = self._net(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()
                total_loss += float(loss.detach())
            avg = total_loss / max(1, len(range(0, len(tx), LSTM_BATCH_SIZE)))
            if avg < best - 1e-6:
                best = avg
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._net is None:
            raise RuntimeError("Model not fitted")
        self._net.eval()
        net = self._net
        out = np.full(len(X), np.nan, dtype=np.float64)
        tx, _ = self._make_sequences(X, None)
        if len(tx) == 0:
            return out
        with torch.no_grad():
            preds = net(tx).cpu().numpy()
        out[self._seq : self._seq + len(preds)] = preds
        return out

    def save(self, path: Path) -> None:
        if self._net is None:
            raise RuntimeError("Model not fitted")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state": self._net.state_dict(),
                "feature_cols": self._feature_cols,
                "seq": self._seq,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> LSTMForexModel:
        m = cls()
        try:
            blob = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            blob = torch.load(path, map_location="cpu")
        m._feature_cols = list(blob["feature_cols"])
        m._seq = int(blob["seq"])
        m._net = _LSTMNet(n_features=len(m._feature_cols))
        m._net.load_state_dict(blob["state"])
        m._net.eval()
        return m
