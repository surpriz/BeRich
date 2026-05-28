"""LSTM classifier behind the :class:`~berich.models.base.Model` protocol.

A two-layer LSTM with a linear head over the canonical 24-feature panel. The model
maintains per-ticker history so that lookback windows never mix bars from different
symbols even though the public ``predict_proba`` API receives the date-sorted panel
as one frame: each (ticker, date) sample is predicted from the *previous* ``lookback``
bars of that *same* ticker, stitched together from whatever history the model has
already seen (training tail at OOF prediction time, full panel at signals time).

Features are standardized using statistics computed on the train fold only. A small
time-respecting validation tail (last 10% of each ticker's train data) drives early
stopping so we don't overfit and don't peek at the test fold.

Important: the input ``x`` must be already filtered to ``FEATURE_COLUMNS`` order
(this is what :func:`berich.datasets.build_dataset` produces). ``tickers`` is a
Series aligned to ``x.index`` whose values identify which symbol each row belongs
to. When ``tickers`` is ``None`` the whole frame is treated as a single stream.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

from berich.datasets.scaling import StandardScaler

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class LSTMConfig:
    """Hyperparameters for :class:`LSTMModel` — small enough to be exhaustively logged."""

    lookback: int = 60
    hidden: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    lr: float = 1e-3
    batch_size: int = 256
    epochs: int = 50
    patience: int = 5
    val_frac: float = 0.1
    device: str | None = None
    seed: int = 42

    def as_dict(self) -> dict[str, Any]:
        return {
            "lookback": self.lookback,
            "hidden": self.hidden,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "patience": self.patience,
            "val_frac": self.val_frac,
            "seed": self.seed,
        }


class _LSTMNet(nn.Module):
    """Stacked LSTM with dropout between layers + linear head to a scalar logit."""

    def __init__(self, *, n_features: int, hidden: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            # PyTorch only applies inter-layer dropout when num_layers > 1.
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head_dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(self.head_dropout(last)).squeeze(-1)


def _ticker_groups(
    n: int,
    tickers: pd.Series | None,
) -> list[tuple[str, np.ndarray]]:
    """Group positional indices by ticker, preserving each ticker's within-group order.

    The panel from :func:`berich.datasets.build_dataset` is sorted by date across all
    tickers and has duplicate dates (one per ticker per day). Using *positional* groups
    avoids the label-based pitfalls of grouping a DatetimeIndex with duplicates.
    """
    if tickers is None:
        return [("__all__", np.arange(n, dtype=int))]
    if len(tickers) != n:
        msg = f"tickers length {len(tickers)} does not match x length {n}"
        raise ValueError(msg)
    aligned = tickers.reset_index(drop=True)
    return [
        (str(t), np.asarray(idx, dtype=int))
        for t, idx in aligned.groupby(aligned, sort=False).groups.items()
    ]


def _build_sequences(
    arr: np.ndarray,
    lookback: int,
) -> np.ndarray:
    """Build overlapping windows of length ``lookback`` from a 2-D array.

    Returns ``(n_windows, lookback, n_features)``. A row at position ``i >= lookback - 1``
    in ``arr`` becomes the *end* of window ``i - (lookback - 1)``, so the returned
    windows align to the last ``len(arr) - lookback + 1`` rows of ``arr``.
    """
    n = len(arr)
    if n < lookback:
        return np.empty((0, lookback, arr.shape[1] if arr.ndim == 2 else 0), dtype=np.float32)  # noqa: PLR2004
    windows = [arr[i - lookback + 1 : i + 1] for i in range(lookback - 1, n)]
    return np.stack(windows).astype(np.float32, copy=False)


class LSTMModel:
    """LSTM classifier exposing the :class:`Model` protocol."""

    def __init__(self, config: LSTMConfig | None = None, **overrides: Any) -> None:
        cfg = config or LSTMConfig()
        if overrides:
            cfg = LSTMConfig(**{**cfg.as_dict(), **overrides})
        self.cfg = cfg

        device_str = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device_str)
        self.scaler: StandardScaler = StandardScaler()
        self.net: _LSTMNet | None = None
        self._n_features: int | None = None
        # Per-ticker tail of (scaled) training rows kept so predict_proba can bootstrap
        # a lookback window without re-seeing train data.
        self._tail: dict[str, np.ndarray] = {}
        self._best_val_auc: float = float("nan")
        self._best_epoch: int = -1

    # ------------------------------------------------------------------ fit ----

    def fit(
        self,
        x: pd.DataFrame,
        y: pd.Series,
        sample_weight: pd.Series | None = None,
        *,
        tickers: pd.Series | None = None,
    ) -> LSTMModel:
        """Train on the per-ticker streams of ``x``; early-stops on a held-out tail."""
        del sample_weight  # sequence model — sample-weighting handled at the loss if needed

        self._n_features = x.shape[1]
        torch.manual_seed(self.cfg.seed)
        np.random.default_rng(self.cfg.seed)

        # Fit the scaler on all train rows (this is one walk-forward fold's train).
        self.scaler.fit(x.to_numpy())

        train_seqs, train_y = [], []
        val_seqs, val_y = [], []
        self._tail = {}

        y_arr = y.to_numpy().astype(np.float32, copy=False)
        for ticker, pos in _ticker_groups(len(x), tickers):
            sub = x.iloc[pos]
            arr = self.scaler.transform(sub.to_numpy()).astype(np.float32, copy=False)
            labels = y_arr[pos]
            seqs = _build_sequences(arr, self.cfg.lookback)
            seq_labels = labels[self.cfg.lookback - 1 :]
            if len(seqs) == 0:
                continue
            # Time-respecting val split: last ``val_frac`` of *this ticker* goes to val.
            n_val = max(1, int(len(seqs) * self.cfg.val_frac))
            train_seqs.append(seqs[:-n_val])
            train_y.append(seq_labels[:-n_val])
            val_seqs.append(seqs[-n_val:])
            val_y.append(seq_labels[-n_val:])
            # Keep the last lookback-1 rows so predict can stitch a window onto test rows.
            self._tail[ticker] = arr[-(self.cfg.lookback - 1) :]

        if not train_seqs:
            msg = "no training sequences could be built — increase data or shrink lookback"
            raise ValueError(msg)

        x_tr = np.concatenate(train_seqs)
        y_tr = np.concatenate(train_y)
        x_val = np.concatenate(val_seqs)
        y_val = np.concatenate(val_y)

        self.net = _LSTMNet(
            n_features=self._n_features,
            hidden=self.cfg.hidden,
            num_layers=self.cfg.num_layers,
            dropout=self.cfg.dropout,
        ).to(self.device)
        # BCEWithLogits is more numerically stable than sigmoid + BCE for binary heads.
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.cfg.lr)

        self._train_loop(x_tr, y_tr, x_val, y_val, criterion, optimizer)
        return self

    def _train_loop(
        self,
        x_tr: np.ndarray,
        y_tr: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        assert self.net is not None  # noqa: S101 — set in fit() before this call
        x_val_t = torch.from_numpy(x_val).to(self.device)
        y_val_t = torch.from_numpy(y_val).to(self.device)

        best_state: dict[str, torch.Tensor] | None = None
        best_val_loss = float("inf")
        best_val_auc = float("nan")
        best_epoch = -1
        epochs_since_best = 0
        n_train = len(x_tr)

        for epoch in range(self.cfg.epochs):
            self.net.train()
            perm = np.random.default_rng(self.cfg.seed + epoch).permutation(n_train)
            running = 0.0
            for start in range(0, n_train, self.cfg.batch_size):
                batch_idx = perm[start : start + self.cfg.batch_size]
                xb = torch.from_numpy(x_tr[batch_idx]).to(self.device)
                yb = torch.from_numpy(y_tr[batch_idx]).to(self.device)
                optimizer.zero_grad(set_to_none=True)
                logits = self.net(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                running += float(loss.item()) * len(batch_idx)
            train_loss = running / max(n_train, 1)

            self.net.eval()
            with torch.no_grad():
                val_logits = self.net(x_val_t)
                val_loss = float(criterion(val_logits, y_val_t).item())
                val_proba = torch.sigmoid(val_logits).cpu().numpy()
            val_auc = _safe_auc(y_val, val_proba)
            logger.info(
                "epoch %d/%d  train_loss=%.4f  val_loss=%.4f  val_auc=%.4f",
                epoch + 1,
                self.cfg.epochs,
                train_loss,
                val_loss,
                val_auc,
            )

            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_val_auc = val_auc
                best_epoch = epoch
                epochs_since_best = 0
                best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            else:
                epochs_since_best += 1
                if epochs_since_best >= self.cfg.patience:
                    logger.info("early stop at epoch %d (best %d)", epoch + 1, best_epoch + 1)
                    break

        if best_state is not None:
            self.net.load_state_dict(best_state)
        self._best_val_auc = best_val_auc
        self._best_epoch = best_epoch

    # ----------------------------------------------------------- predict_proba ----

    def predict_proba(
        self,
        x: pd.DataFrame,
        *,
        tickers: pd.Series | None = None,
    ) -> np.ndarray:
        if self.net is None:
            msg = "LSTMModel must be fit before predict_proba"
            raise RuntimeError(msg)

        # Position-indexed output so duplicate dates across tickers in the panel
        # are handled correctly (label-based .loc would broadcast wrongly).
        out = np.full(len(x), 0.5, dtype=float)
        self.net.eval()

        for ticker, pos in _ticker_groups(len(x), tickers):
            # Preserve within-ticker chronological order even if the panel mixed them.
            # x is date-sorted across tickers but a single ticker's rows keep relative order.
            sub = x.iloc[pos]
            arr = self.scaler.transform(sub.to_numpy()).astype(np.float32, copy=False)
            tail = self._tail.get(ticker)
            if tail is None or len(tail) < self.cfg.lookback - 1:
                logger.debug("ticker %s has no train history; emitting 0.5 probas", ticker)
                continue
            stitched = np.concatenate([tail, arr], axis=0)
            seqs = _build_sequences(stitched, self.cfg.lookback)
            test_seqs = seqs[-len(arr) :]
            if len(test_seqs) == 0:
                continue
            with torch.no_grad():
                probs = self._batched_proba(test_seqs)
            out[pos] = probs

        return out

    def _batched_proba(self, seqs: np.ndarray) -> np.ndarray:
        assert self.net is not None  # noqa: S101
        out: list[np.ndarray] = []
        for start in range(0, len(seqs), self.cfg.batch_size):
            chunk = seqs[start : start + self.cfg.batch_size]
            x_t = torch.from_numpy(chunk).to(self.device)
            logits = self.net(x_t)
            out.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(out) if out else np.empty(0)

    # ------------------------------------------------------------- diagnostics ----

    @property
    def best_val_auc(self) -> float:
        return self._best_val_auc

    @property
    def best_epoch(self) -> int:
        return self._best_epoch


def _safe_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """ROC-AUC that returns NaN when only one class is present (avoids sklearn raise)."""
    classes = np.unique(y_true)
    if classes.size < 2:  # noqa: PLR2004
        return float("nan")
    return float(roc_auc_score(y_true, y_proba))


__all__ = ["LSTMConfig", "LSTMModel"]
