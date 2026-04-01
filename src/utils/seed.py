from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed all random number generators (Python, NumPy, PyTorch) for reproducibility.

    Args:
        seed: Integer seed value to use across all RNG backends.
        deterministic: If True, enables cuDNN deterministic mode and
            ``torch.use_deterministic_algorithms`` for fully reproducible GPU ops.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            # Older torch builds may not support this API; keep best-effort determinism.
            pass

