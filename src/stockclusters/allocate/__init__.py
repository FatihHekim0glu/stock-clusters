"""Allocators: HRP, IVP, naive 1/N, and the Markowitz adapter.

Each allocator maps a (shared) covariance — and, for max-Sharpe, a shrunk mu —
to simplex weights. Importing this subpackage has no side effects (cvxpy is
imported lazily only inside the max-Sharpe function).
"""

from __future__ import annotations

from stockclusters.allocate.hrp import (
    HRPResult,
    get_cluster_var,
    get_rec_bipart,
    hrp_allocate,
)
from stockclusters.allocate.ivp import ivp_weights
from stockclusters.allocate.markowitz_adapter import max_sharpe_weights, min_var_weights
from stockclusters.allocate.naive import naive_weights

__all__ = [
    "HRPResult",
    "get_cluster_var",
    "get_rec_bipart",
    "hrp_allocate",
    "ivp_weights",
    "max_sharpe_weights",
    "min_var_weights",
    "naive_weights",
]
