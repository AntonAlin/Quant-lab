"""PyTorch sequence models: LSTM, GRU and a small encoder-only transformer.

Windowing happens *inside* fit/predict on whatever contiguous slice the caller
hands over. The walk-forward loop hands over the train fold and the test fold
separately, so a window can never contain bars from both sides of the split.
The price of that purity: the first seq_len-1 bars of every test fold get no
prediction (NaN -> flat in the backtest). It is stated in the UI. Live with it.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from quantlab.features.registry import ParamSpec
from quantlab.models.base import ModelAdapter, ProgressFn, make_preprocessor

P = ParamSpec


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_windows(X: np.ndarray, seq_len: int) -> np.ndarray:
    """(n, f) -> (n - seq_len + 1, seq_len, f), window i ends at row i + seq_len - 1."""
    n, f = X.shape
    if n < seq_len:
        return np.zeros((0, seq_len, f), dtype=np.float32)
    # Strided view, then copy so torch does not see a weird stride.
    s0, s1 = X.strides
    out = np.lib.stride_tricks.as_strided(X, shape=(n - seq_len + 1, seq_len, f), strides=(s0, s0, s1))
    return np.ascontiguousarray(out, dtype=np.float32)


class _RNN(nn.Module):
    def __init__(self, kind: str, n_features: int, hidden: int, layers: int, dropout: float, bidirectional: bool, n_classes: int) -> None:
        super().__init__()
        cls = nn.LSTM if kind == "lstm" else nn.GRU
        self.rnn = cls(
            n_features,
            hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        d = hidden * (2 if bidirectional else 1)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Dropout(dropout), nn.Linear(d, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :])


class _Transformer(nn.Module):
    def __init__(self, n_features: int, d_model: int, n_heads: int, n_layers: int, dropout: float, seq_len: int, n_classes: int) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads}).")
        self.proj = nn.Linear(n_features, d_model)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=4 * d_model, dropout=dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x) + self.pos[:, : x.shape[1], :]
        h = self.enc(h)
        # Last token carries the "now"; mean-pooling would let the model average away recency.
        return self.head(h[:, -1, :])


class TorchSequenceAdapter(ModelAdapter):
    is_sequence_model = True
    kind: str = "lstm"
    name = "lstm"
    param_schema = (
        P("seq_len", "int", 20, 2, 250, 1),
        P("hidden_size", "int", 32, 4, 512, 4),
        P("num_layers", "int", 1, 1, 6, 1),
        P("dropout", "float", 0.2, 0.0, 0.9, 0.05),
        P("bidirectional", "bool", False),
        P("epochs", "int", 40, 1, 500, 1),
        P("batch_size", "int", 64, 8, 2048, 8),
        P("lr", "float", 1e-3, 1e-5, 1e-1, 1e-4),
        P("weight_decay", "float", 1e-4, 0.0, 1.0, 1e-4),
        P("patience", "int", 6, 1, 100, 1, help="Early-stopping patience in epochs, on the validation tail."),
        P("val_fraction", "float", 0.15, 0.05, 0.5, 0.05, help="Chronological tail of the train fold held out for early stopping."),
    )

    def _build_net(self, n_features: int, n_classes: int) -> nn.Module:
        p = self.params
        return _RNN(self.kind, n_features, p["hidden_size"], p["num_layers"], p["dropout"], p["bidirectional"], n_classes)

    def param_distributions(self, rng: np.random.Generator) -> dict[str, Any]:
        return {
            "hidden_size": int(rng.choice([16, 32, 64, 128])),
            "num_layers": int(rng.integers(1, 3)),
            "dropout": float(rng.choice([0.1, 0.2, 0.3, 0.5])),
            "lr": float(10 ** rng.uniform(-4, -2)),
        }

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight: pd.Series | None = None, progress: ProgressFn | None = None) -> "TorchSequenceAdapter":
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        p = self.params
        seq_len = p["seq_len"]
        if not X.index.equals(y.index):
            raise ValueError("X and y must share an index.")
        self.feature_names_ = list(X.columns)
        yv = y.to_numpy(dtype=float)
        self.classes_ = np.unique(yv[~np.isnan(yv)]).astype(int)
        if len(self.classes_) < 2:
            raise ValueError(f"{self.name}: single class in training fold.")
        # Preprocess on the whole train slice (fit here, and only here).
        self.pre_ = make_preprocessor(self.scaler, self.imputer)
        Xp = self.pre_.fit_transform(X).astype(np.float32)
        Xp = np.nan_to_num(Xp, nan=0.0)
        W = make_windows(Xp, seq_len)
        # Window i ends at row i + seq_len - 1; that row's label is the target.
        targets = yv[seq_len - 1 :]
        weights = np.ones(len(targets)) if sample_weight is None else np.nan_to_num(sample_weight.reindex(X.index).to_numpy(dtype=float)[seq_len - 1 :], nan=0.0)
        ok = ~np.isnan(targets)
        W, targets, weights = W[ok], targets[ok], weights[ok]
        if len(W) < 50:
            raise ValueError(f"{self.name}: only {len(W)} windows after seq_len={seq_len}. Reduce seq_len or enlarge the train window.")
        t_idx = np.searchsorted(self.classes_, targets.astype(int))
        n_val = max(10, int(len(W) * p["val_fraction"]))
        n_tr = len(W) - n_val
        # Validation windows that overlap the training tail share bars with it;
        # drop seq_len of them so early stopping is not judged on memorised bars.
        val_start = n_tr + seq_len - 1
        if val_start >= len(W):
            val_start = n_tr
        device = _device()
        net = self._build_net(W.shape[2], len(self.classes_)).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
        loss_fn = nn.CrossEntropyLoss(reduction="none")
        Xtr, ytr, wtr = (torch.tensor(a).to(device) for a in (W[:n_tr], t_idx[:n_tr], weights[:n_tr].astype(np.float32)))
        Xva, yva = (torch.tensor(a).to(device) for a in (W[val_start:], t_idx[val_start:]))
        best_state, best_val, bad = copy.deepcopy(net.state_dict()), np.inf, 0
        g = torch.Generator(device="cpu").manual_seed(self.seed)
        self.history_: list[dict[str, float]] = []
        for epoch in range(p["epochs"]):
            net.train()
            perm = torch.randperm(n_tr, generator=g)
            tot = 0.0
            for s in range(0, n_tr, p["batch_size"]):
                b = perm[s : s + p["batch_size"]].to(device)
                opt.zero_grad()
                l = loss_fn(net(Xtr[b]), ytr[b])
                l = (l * wtr[b]).sum() / wtr[b].sum().clamp_min(1e-8)
                l.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                tot += float(l) * len(b)
            net.eval()
            with torch.no_grad():
                val = float(loss_fn(net(Xva), yva).mean()) if len(Xva) else np.nan
            rec = {"epoch": epoch + 1, "train_loss": tot / n_tr, "val_loss": val, "best_val": min(best_val, val)}
            self.history_.append(rec)
            if progress:
                progress({"stage": "epoch", "model": self.name, **rec})
            if val < best_val - 1e-5:
                best_val, bad, best_state = val, 0, copy.deepcopy(net.state_dict())
            else:
                bad += 1
                if bad >= p["patience"]:
                    break
        net.load_state_dict(best_state)
        net.eval()
        self.net_ = net.cpu()
        self.fitted_ = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        self._check_columns(X)
        seq_len = self.params["seq_len"]
        Xp = np.nan_to_num(self.pre_.transform(X).astype(np.float32), nan=0.0)
        W = make_windows(Xp, seq_len)
        out = pd.DataFrame(np.nan, index=X.index, columns=[int(c) for c in self.classes_], dtype=float)
        if len(W) == 0:
            return out
        with torch.no_grad():
            logits = self.net_(torch.tensor(W))
            proba = torch.softmax(logits, dim=1).numpy()
        out.iloc[seq_len - 1 :, :] = proba
        return out

    # torch modules pickle fine on CPU, but keep the device out of the file.
    def __getstate__(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        if "net_" in d:
            d["net_"] = d["net_"].cpu()
        return d


class LSTMAdapter(TorchSequenceAdapter):
    kind = "lstm"
    name = "lstm"


class GRUAdapter(TorchSequenceAdapter):
    kind = "gru"
    name = "gru"


class TransformerAdapter(TorchSequenceAdapter):
    kind = "transformer"
    name = "transformer"
    param_schema = (
        P("seq_len", "int", 20, 2, 250, 1),
        P("d_model", "int", 32, 8, 512, 8),
        P("n_heads", "int", 4, 1, 16, 1),
        P("n_layers", "int", 2, 1, 8, 1),
        P("dropout", "float", 0.2, 0.0, 0.9, 0.05),
        P("epochs", "int", 40, 1, 500, 1),
        P("batch_size", "int", 64, 8, 2048, 8),
        P("lr", "float", 5e-4, 1e-5, 1e-1, 1e-4),
        P("weight_decay", "float", 1e-4, 0.0, 1.0, 1e-4),
        P("patience", "int", 6, 1, 100, 1),
        P("val_fraction", "float", 0.15, 0.05, 0.5, 0.05),
    )

    def _build_net(self, n_features: int, n_classes: int) -> nn.Module:
        p = self.params
        return _Transformer(n_features, p["d_model"], p["n_heads"], p["n_layers"], p["dropout"], p["seq_len"], n_classes)

    def param_distributions(self, rng: np.random.Generator) -> dict[str, Any]:
        return {
            "d_model": int(rng.choice([16, 32, 64])),
            "n_layers": int(rng.integers(1, 4)),
            "dropout": float(rng.choice([0.1, 0.2, 0.3])),
            "lr": float(10 ** rng.uniform(-4, -2.5)),
        }


DEEP: dict[str, type[TorchSequenceAdapter]] = {a.name: a for a in (LSTMAdapter, GRUAdapter, TransformerAdapter)}
