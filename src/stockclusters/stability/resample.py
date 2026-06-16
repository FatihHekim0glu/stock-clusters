"""Rolling-window cluster re-fitting and the StabilityResult bundle.

Re-fits clusters on a sequence of rolling in-sample windows and summarizes how
stable the assignment is over time. The frozen :class:`StabilityResult` carries
the per-window labelings, the headline adjacent-window ARI, and birth/death events.

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from stockclusters._typing import ReturnsLike

__all__ = ["StabilityResult", "rolling_stability"]


@dataclass(frozen=True, slots=True)
class StabilityResult:
    """Immutable result of a rolling-window cluster-stability analysis.

    Attributes
    ----------
    window_labels:
        Per-window aligned label Series, ordered earliest-to-latest.
    window_dates:
        The (end) date of each window, parallel to ``window_labels``.
    ari_mean:
        The headline mean adjacent-window ARI.
    ari_series:
        The per-adjacent-pair ARI values (``len(window_labels) - 1`` entries).
    n_windows:
        The number of rolling windows evaluated.
    """

    window_labels: list[pd.Series]
    window_dates: list[str]
    ari_mean: float
    ari_series: list[float]
    n_windows: int
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this result.

        Each per-window labeling is rendered as a ``{ticker: cluster_id}`` mapping
        and ``ari_mean`` scrubbed of non-finite values downstream.
        """
        out = asdict(self)
        out["window_labels"] = [
            {str(k): int(v) for k, v in labels.items()} for labels in self.window_labels
        ]
        out["window_dates"] = [str(d) for d in self.window_dates]
        out["ari_mean"] = float(self.ari_mean)
        out["ari_series"] = [float(v) for v in self.ari_series]
        out["n_windows"] = int(self.n_windows)
        return out


def rolling_stability(
    returns: ReturnsLike,
    *,
    n_clusters: int,
    train_window: int = 504,
    step: int = 21,
    method: str = "average",
    denoise: bool = True,
) -> StabilityResult:
    r"""Re-fit clusters on rolling windows and summarize temporal stability.

    Slides a ``train_window``-length in-sample window across ``returns`` in
    ``step``-sized increments, re-fitting clusters on each window (optionally
    RMT-denoised), aligning labels across windows, and computing the headline
    adjacent-window ARI.

    NO-LOOKAHEAD REQUIREMENT: each window's clustering uses only the observations
    inside that window; windows do not peek forward.

    Parameters
    ----------
    returns:
        A wide panel of asset returns (rows = time, columns = asset).
    n_clusters:
        The (fixed) number of clusters per window.
    train_window:
        The in-sample window length in observations.
    step:
        The stride between consecutive windows in observations.
    method:
        Linkage method for the per-window clustering.
    denoise:
        Whether to RMT-denoise each window's covariance before clustering.

    Returns
    -------
    StabilityResult
        The frozen stability bundle.

    Raises
    ------
    ValidationError
        If ``train_window`` exceeds the available observations or ``step < 1``.
    """
    raise NotImplementedError
