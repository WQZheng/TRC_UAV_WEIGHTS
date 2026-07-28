"""Global reproducibility utilities.

Call set_seed(SEED) at the start of every script so that two runs on the
same hardware / library versions produce identical results. We fix the
Python, NumPy and PyTorch RNGs and enable deterministic cuDNN.

Reproducibility scope (verified):
  * On the SAME GPU model + CUDA/cuDNN + PyTorch version, two runs with
    the same seed produce bit-identical training curves (verified: two
    Stage-2 runs with seed=999 gave identical per-iter loss values).
  * Across DIFFERENT GPU models / library versions, tiny floating-point
    differences can appear, but metrics match closely. We use float64 in
    the planner/dynamics, which reduces cross-hardware drift.

NOTE: we intentionally do NOT call
torch.use_deterministic_algorithms(True): it makes the cuDNN GRU either
error out or hang and slows everything ~5x. cudnn.deterministic=True plus
fixed RNG seeds already give reproducible results on fixed hardware.
"""
from __future__ import annotations
import os
import random
import numpy as np
import torch

GLOBAL_SEED = 12345  # single source of truth for the whole pipeline


def set_seed(seed: int = GLOBAL_SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    set_seed()
    print("seed set to", GLOBAL_SEED)
    print("sample:", torch.randn(3).tolist())
