"""Pure-function verdict derivation for the clustering horse race.

The headline verdict is a PURE FUNCTION of the inference outputs
``(memmel_jk_pvalue, deflated_sharpe, sharpe_diff)`` from a fixed enum. It is
structurally incapable of emitting ``clusters_beat_1n`` while the Memmel-JK test is
insignificant or the deflated Sharpe is non-positive — the truth table is
unit-tested. This is what keeps the README honest: the verdict is derived from the
evidence, never narrated.

Importing this module has no side effects.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = ["ClusteringVerdict", "derive_clustering_verdict"]


class ClusteringVerdict(StrEnum):
    """Possible headline verdicts for the cluster-aware-vs-1/N comparison.

    The values are stable string identifiers safe to serialize across the API
    boundary and render in the frontend.
    """

    #: The OOS Sharpe gap is positive AND statistically significant (significant
    #: Memmel-JK test AND positive deflated Sharpe).
    CLUSTERS_BEAT_1N = "clusters_beat_1n"

    #: The OOS Sharpe gap is negative AND statistically significant.
    CLUSTERS_LOSE_TO_1N = "clusters_lose_to_1n"

    #: The gap is not statistically distinguishable from zero — the expected,
    #: literature-consistent outcome (clustering is diagnostic, not alpha).
    NO_SIGNIFICANT_DIFFERENCE = "no_significant_difference"


def derive_clustering_verdict(
    memmel_jk_pvalue: float,
    deflated_sharpe: float,
    sharpe_diff: float,
    *,
    alpha: float = 0.05,
    dsr_threshold: float = 0.0,
) -> ClusteringVerdict:
    r"""Derive the headline clustering verdict (pure function).

    Decision rule (truth-table unit-tested):

    1. If the Memmel-JK test is insignificant (``memmel_jk_pvalue >= alpha``) OR the
       deflated Sharpe fails its threshold (``deflated_sharpe <= dsr_threshold``),
       return :attr:`ClusteringVerdict.NO_SIGNIFICANT_DIFFERENCE`. (A directional
       claim requires BOTH the test AND the DSR to support it.)
    2. Otherwise, if ``sharpe_diff > 0``, return
       :attr:`ClusteringVerdict.CLUSTERS_BEAT_1N`.
    3. Otherwise (``sharpe_diff < 0``), return
       :attr:`ClusteringVerdict.CLUSTERS_LOSE_TO_1N`.

    HONESTY REQUIREMENT: this function MUST NOT return
    :attr:`ClusteringVerdict.CLUSTERS_BEAT_1N` while the Memmel-JK p-value is
    insignificant or the deflated Sharpe is non-positive, regardless of the point
    estimate. The verdict is a deterministic consequence of the evidence.

    Parameters
    ----------
    memmel_jk_pvalue:
        The Jobson-Korkie-Memmel two-sided p-value for the Sharpe gap.
    deflated_sharpe:
        The deflated Sharpe ratio (FULL-grid ``n_trials``) of the selected
        cluster-aware strategy.
    sharpe_diff:
        The OOS Sharpe gap (best cluster-aware strategy minus 1/N).
    alpha:
        Significance level for the Memmel-JK test (default ``0.05``).
    dsr_threshold:
        Minimum deflated Sharpe required to support a positive claim (default
        ``0.0``: the DSR must be strictly positive).

    Returns
    -------
    ClusteringVerdict
        The derived headline verdict.

    Raises
    ------
    ValidationError
        If ``memmel_jk_pvalue`` is outside ``[0, 1]`` or not finite.
    """
    raise NotImplementedError
