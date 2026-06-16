"""Number-of-clusters selection: gap statistic vs a phase-randomized null.

The pre-registered selector is the Tibshirani (2001) gap statistic evaluated
against a **phase-randomized** null reference distribution, with the **1-SE rule**
choosing the smallest ``k`` whose gap is within one standard error of the maximum.
Silhouette and MST modularity are reported as cross-checks but do NOT override the
gap selection.

The phase-randomized null FFT-phase-scrambles each asset's return series
independently: it preserves each asset's marginal power spectrum (autocorrelation)
while destroying cross-asset correlation, giving a null with realistic single-asset
dynamics but no genuine cluster structure.

DSR REQUIREMENT: every ``k`` candidate evaluated here is a swept axis; the count of
candidates feeds the DSR ``n_trials`` product. All trials are recorded on
:class:`GapResult`.

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from stockclusters._typing import MatrixLike, ReturnsLike

__all__ = ["GapResult", "phase_randomize", "select_k_gap"]


@dataclass(frozen=True, slots=True)
class GapResult:
    """Immutable result of gap-statistic cluster-number selection.

    Attributes
    ----------
    k_selected:
        The selected number of clusters under the Tibshirani 1-SE rule.
    k_candidates:
        The full list of ``k`` values evaluated (a swept DSR axis).
    gap:
        The gap statistic ``E*[log W_k] - log W_k`` at each candidate ``k``.
    gap_se:
        The standard error ``s_k`` of the gap at each candidate ``k``.
    log_wk:
        The observed ``log W_k`` (pooled within-cluster dispersion) at each ``k``.
    selection_rule:
        The selector identifier (``"tibshirani_1se"``).
    n_trials:
        The number of cluster-number candidates evaluated (``len(k_candidates)``);
        a multiplicative factor in the DSR trial count.
    """

    k_selected: int
    k_candidates: list[int]
    gap: list[float]
    gap_se: list[float]
    log_wk: list[float]
    selection_rule: str = "tibshirani_1se"
    n_trials: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this result."""
        out = asdict(self)
        out["k_selected"] = int(self.k_selected)
        out["k_candidates"] = [int(k) for k in self.k_candidates]
        out["gap"] = [float(v) for v in self.gap]
        out["gap_se"] = [float(v) for v in self.gap_se]
        out["log_wk"] = [float(v) for v in self.log_wk]
        out["n_trials"] = int(self.n_trials)
        return out


def phase_randomize(returns: ReturnsLike, *, seed: int = 0) -> np.ndarray:
    r"""FFT phase-randomized surrogate of a returns panel.

    Independently phase-scrambles each asset's return series in the Fourier
    domain: the amplitude spectrum (hence the marginal autocorrelation) is
    preserved while the phases are randomized, destroying cross-asset correlation.
    This is the null reference for the gap statistic.

    REPRODUCIBILITY: phases are drawn from a seeded PCG64 generator, so a fixed
    ``seed`` reproduces the surrogate byte-for-byte.

    Parameters
    ----------
    returns:
        A wide panel of asset returns (rows = time, columns = asset).
    seed:
        Master seed for the random phases.

    Returns
    -------
    numpy.ndarray
        A surrogate returns array of the same shape, with preserved per-asset
        marginal spectra and destroyed cross-correlation.

    Raises
    ------
    ValidationError
        If ``returns`` has fewer than two observations.
    """
    raise NotImplementedError


def select_k_gap(
    returns: ReturnsLike,
    dist: MatrixLike,
    *,
    k_min: int = 2,
    k_max: int = 20,
    method: str = "average",
    n_references: int = 20,
    seed: int = 0,
) -> GapResult:
    r"""Select the number of clusters via the gap statistic (1-SE rule).

    For each ``k`` in ``[k_min, k_max]`` computes the observed pooled
    within-cluster dispersion ``log W_k`` and its expectation under ``n_references``
    phase-randomized surrogates, forms the gap ``E*[log W_k] - log W_k`` and its
    standard error ``s_k``, and applies the Tibshirani 1-SE rule: the smallest ``k``
    with ``gap(k) >= gap(k+1) - s_{k+1}``.

    HONESTY REQUIREMENT: the null is the phase-randomized reference (preserves
    marginal spectra, destroys cross-correlation), NOT a uniform-box null. The
    uniform null is used only to validate the gap code path against a reference,
    never to select ``k``. Silhouette / MST modularity are reported cross-checks,
    not selectors.

    DSR REQUIREMENT: ``GapResult.n_trials`` records ``len(k_candidates)`` so the
    caller can fold it into the full DSR trial-count product.

    Parameters
    ----------
    returns:
        The in-sample returns panel used to draw phase-randomized surrogates.
    dist:
        The Mantegna distance matrix the clustering is performed on.
    k_min, k_max:
        Inclusive range of cluster counts to evaluate.
    method:
        Linkage method used when computing ``W_k`` (defaults to ``"average"``).
    n_references:
        Number of phase-randomized reference draws ``B``.
    seed:
        Master seed for the reference draws.

    Returns
    -------
    GapResult
        The frozen selection bundle, including all evaluated candidates.

    Raises
    ------
    ValidationError
        If ``k_min < 1``, ``k_max < k_min``, or ``k_max`` exceeds the asset count.
    """
    raise NotImplementedError
