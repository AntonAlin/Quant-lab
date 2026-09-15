"""Global determinism plumbing.

Why this exists: "I ran it again and got a different Sharpe" is the second most
common bug report in quant research, right after "it worked in-sample". Every
random consumer we depend on gets seeded from one place so nobody forgets one.
"""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int) -> None:
    """Seed python, numpy, torch (if installed) and the hash env var.

    sklearn, xgboost and lightgbm take `random_state` per estimator, so they are
    seeded at construction time in the model adapters rather than here.
    """
    if not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a non-negative int, got {seed!r}")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # noqa: WPS433 - optional heavy import, only when present

        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover - no GPU in CI
            torch.cuda.manual_seed_all(seed)
        # Deterministic cuDNN is slower, but "slower" beats "different every run".
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:  # pragma: no cover - torch is optional at import time
        pass
