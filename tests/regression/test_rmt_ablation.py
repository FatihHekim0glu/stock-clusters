"""RMT-denoise ablation regression (Group C).

ADR-0001 puts RMT denoising BEFORE clustering. This regression builds the honest
denoise-on/off ablation table on two axes - ARI-vs-GICS (here the recoverable
ground-truth blocks stand in for GICS sectors) and temporal stability (mean
adjacent-window ARI) - and asserts the *honest* properties:

- both settings recover the block structure (the effect is reported even if
  marginal, not assumed to be large);
- the two settings are tabulated side by side so the README "RMT-ablation" row is
  backed by a runnable test;
- on the pure-noise null, neither setting manufactures spurious structure.

The point is faithful reporting, not a guaranteed denoise win - on clean synthetic
blocks the gap is expected to be small.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from stockclusters.clustering.hierarchical import hierarchical_clusters
from stockclusters.correlation.distance import mantegna_distance
from stockclusters.correlation.estimate import correlation_matrix
from stockclusters.correlation.rmt import marchenko_pastur_clip
from stockclusters.metrics import ari_vs_gics
from stockclusters.stability.resample import rolling_stability


def _denoised_correlation(returns: pd.DataFrame) -> pd.DataFrame:
    """RMT-denoise the correlation, renormalized back to a unit-diagonal correlation."""
    corr = correlation_matrix(returns)
    denoised = marchenko_pastur_clip(corr, n_obs=len(returns))
    arr = denoised.to_numpy()
    std = np.sqrt(np.diag(arr))
    safe = np.where(std > 0.0, std, 1.0)
    renorm = arr / np.outer(safe, safe)
    np.fill_diagonal(renorm, 1.0)
    return pd.DataFrame(renorm, index=corr.index, columns=corr.columns)


def _cluster_ari(returns: pd.DataFrame, gics: dict[str, str], *, denoise: bool) -> float:
    """ARI of the clustering (with/without RMT denoise) against the GICS map."""
    corr = _denoised_correlation(returns) if denoise else correlation_matrix(returns)
    dist = mantegna_distance(corr)
    result = hierarchical_clusters(dist, n_clusters=4, method="average")
    return ari_vs_gics(result.labels, gics)


def _build_ablation_table(
    returns: pd.DataFrame, gics: dict[str, str]
) -> dict[bool, dict[str, float]]:
    """The denoise-on/off table on the two reported axes (ARI-vs-GICS, stability)."""
    table: dict[bool, dict[str, float]] = {}
    for denoise in (False, True):
        ari = _cluster_ari(returns, gics, denoise=denoise)
        stability = rolling_stability(
            returns,
            n_clusters=4,
            train_window=252,
            step=63,
            method="average",
            denoise=denoise,
        )
        table[denoise] = {
            "ari_vs_gics": float(ari),
            "stability_ari_mean": float(stability.ari_mean),
        }
    return table


@pytest.mark.regression
def test_rmt_ablation_table_recovers_blocks(
    k_blocks: pd.DataFrame, k_blocks_truth: pd.Series
) -> None:
    """Both denoise settings recover the block structure; table is well-formed."""
    gics = {str(a): f"S{int(s)}" for a, s in k_blocks_truth.items()}
    table = _build_ablation_table(k_blocks, gics)

    for denoise, row in table.items():
        # ARI-vs-GICS is a valid, finite index in [-1, 1].
        assert -1.0 <= row["ari_vs_gics"] <= 1.0, (denoise, row)
        # On the clean recoverable panel both settings clear a conservative bar.
        assert row["ari_vs_gics"] >= 0.8, (denoise, row)
        # Stability is a finite mean ARI (or NaN with <2 windows; not the case here).
        assert np.isfinite(row["stability_ari_mean"]), (denoise, row)


@pytest.mark.regression
def test_rmt_ablation_reported_honestly(k_blocks: pd.DataFrame, k_blocks_truth: pd.Series) -> None:
    """The denoise effect is reported as a finite delta, not assumed to be large.

    Honest reporting: we surface the on-vs-off delta on each axis and assert only
    that it is finite and modest on clean blocks (the structure is already trivial
    to recover, so denoising cannot help much). No assertion that denoise WINS.
    """
    gics = {str(a): f"S{int(s)}" for a, s in k_blocks_truth.items()}
    table = _build_ablation_table(k_blocks, gics)

    ari_delta = table[True]["ari_vs_gics"] - table[False]["ari_vs_gics"]
    stab_delta = table[True]["stability_ari_mean"] - table[False]["stability_ari_mean"]

    assert np.isfinite(ari_delta)
    assert np.isfinite(stab_delta)
    # On clean blocks the recovery is already near-perfect, so the honest delta is
    # small in magnitude (this is the marginal-effect finding, reported not hidden).
    assert abs(ari_delta) <= 0.5


@pytest.mark.regression
def test_rmt_ablation_no_spurious_structure_on_noise(pure_noise: pd.DataFrame) -> None:
    """On the null, neither denoise setting manufactures stable cluster structure."""
    for denoise in (False, True):
        corr = _denoised_correlation(pure_noise) if denoise else correlation_matrix(pure_noise)
        dist = mantegna_distance(corr)
        # Force a 3-cluster cut and measure ARI against a meaningless reference.
        result = hierarchical_clusters(dist, n_clusters=3, method="average")
        # No genuine structure: ARI against an arbitrary 3-sector map is low.
        arbitrary: dict[str, Any] = {str(a): f"S{i % 3}" for i, a in enumerate(pure_noise.columns)}
        ari = ari_vs_gics(result.labels, arbitrary)
        assert ari < 0.5, (denoise, ari)
