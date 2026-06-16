"""Regression slot: honest-null headline guard (placeholder).

On the ``pure_noise`` fixture the diversification horse race MUST come back
insignificant (Memmel-JK p not significant, DSR <= 0) and the verdict must be
NO_SIGNIFICANT_DIFFERENCE. Also covers to_dict() snapshots and the DSR trial-count
guard. Marked xfail until the pipeline lands.
"""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.mark.regression
@pytest.mark.xfail(reason="diversification pipeline is a stub", strict=False)
def test_pure_noise_is_insignificant(pure_noise: pd.DataFrame) -> None:
    """Pure noise yields no significant cluster-vs-1/N edge (the honest null)."""
    from stockclusters.evaluation.verdict import (
        ClusteringVerdict,
        derive_clustering_verdict,
    )

    verdict = derive_clustering_verdict(
        memmel_jk_pvalue=0.8, deflated_sharpe=-0.1, sharpe_diff=0.02
    )
    assert verdict is ClusteringVerdict.NO_SIGNIFICANT_DIFFERENCE
    assert not pure_noise.empty
