"""Cluster-aware allocation schemes and the diversification result bundle.

Each scheme maps a cluster labeling (and, where relevant, a covariance estimate)
to a weight vector on the simplex:

- :func:`one_over_n_weights` — naive equal weight across all assets (the honest
  benchmark the clustering must beat).
- :func:`cluster_equal_weight` — equal weight per cluster, split equally within.
- :func:`stripped_hrp_weights` — inverse-variance within cluster, equal across
  clusters ("stripped-HRP": the cross-cluster recursive bisection of HRP replaced
  by a flat equal split).

All schemes return labelled simplex weights; the walk-forward engine applies them
with ``shift(1)``.

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from stockclusters._typing import MatrixLike

__all__ = [
    "DiversificationResult",
    "cluster_equal_weight",
    "one_over_n_weights",
    "stripped_hrp_weights",
]


@dataclass(frozen=True, slots=True)
class DiversificationResult:
    """Immutable result of the cluster-vs-1/N diversification horse race.

    Attributes
    ----------
    one_over_n_sharpe:
        Annualized OOS Sharpe of the naive 1/N portfolio.
    cluster_ew_sharpe:
        Annualized OOS Sharpe of the equal-weight-across-cluster portfolio.
    stripped_hrp_sharpe:
        Annualized OOS Sharpe of the stripped-HRP portfolio.
    sharpe_diff_vs_1overN:
        The best cluster-aware Sharpe minus the 1/N Sharpe.
    memmel_jk_pvalue:
        Jobson-Korkie-Memmel two-sided p-value on that Sharpe gap.
    deflated_sharpe:
        Deflated Sharpe of the selected cluster-aware strategy under the FULL
        trial count.
    n_trials:
        The full DSR trial count (product of every swept axis).
    cost_bps:
        The per-side transaction cost (basis points) used in the OOS backtest.
    """

    one_over_n_sharpe: float
    cluster_ew_sharpe: float
    stripped_hrp_sharpe: float
    sharpe_diff_vs_1overN: float
    memmel_jk_pvalue: float
    deflated_sharpe: float
    n_trials: int
    cost_bps: float
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this result."""
        out = asdict(self)
        for key in (
            "one_over_n_sharpe",
            "cluster_ew_sharpe",
            "stripped_hrp_sharpe",
            "sharpe_diff_vs_1overN",
            "memmel_jk_pvalue",
            "deflated_sharpe",
            "cost_bps",
        ):
            out[key] = float(getattr(self, key))
        out["n_trials"] = int(self.n_trials)
        return out


def one_over_n_weights(assets: list[str]) -> pd.Series:
    r"""Naive 1/N equal-weight portfolio (the honest benchmark).

    Parameters
    ----------
    assets:
        The asset tickers in the universe.

    Returns
    -------
    pandas.Series
        Equal weights ``1 / N`` indexed by asset, summing to one.

    Raises
    ------
    ValidationError
        If ``assets`` is empty.
    """
    raise NotImplementedError


def cluster_equal_weight(labels: pd.Series) -> pd.Series:
    r"""Equal weight across clusters, equal weight within each cluster.

    Allocates ``1 / K`` to each of the ``K`` clusters and splits each cluster's
    budget equally among its members.

    Parameters
    ----------
    labels:
        Integer cluster labels indexed by asset ticker.

    Returns
    -------
    pandas.Series
        Simplex weights indexed by asset, summing to one.

    Raises
    ------
    ValidationError
        If ``labels`` is empty.
    """
    raise NotImplementedError


def stripped_hrp_weights(labels: pd.Series, cov: MatrixLike) -> pd.Series:
    r"""Stripped-HRP: inverse-variance within cluster, equal across clusters.

    Allocates ``1 / K`` to each cluster and, within each cluster, weights members
    inversely to their variances (the within-cluster step of HRP) — but replaces
    HRP's cross-cluster recursive bisection with a flat equal split across
    clusters ("stripped").

    Parameters
    ----------
    labels:
        Integer cluster labels indexed by asset ticker.
    cov:
        The ``N x N`` covariance matrix (provides per-asset variances), labelled
        by asset.

    Returns
    -------
    pandas.Series
        Simplex weights indexed by asset, summing to one.

    Raises
    ------
    ValidationError
        If ``labels`` and ``cov`` labels disagree or ``cov`` has a non-positive
        diagonal.
    """
    raise NotImplementedError
