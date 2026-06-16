"""Parity-oracle slot: scipy/sklearn references (placeholder).

The parity suite pins linkage/cophenetic vs scipy (1e-10), silhouette (1e-10) and
ARI (1e-12) vs sklearn, K-means inertia (1e-8), RMT edges vs analytic MP (1e-10),
DSR/PSR (1e-8), and the gap machinery. Marked xfail until kernels land.
"""

from __future__ import annotations

import pytest


@pytest.mark.parity
@pytest.mark.xfail(reason="distance kernel is a stub", strict=False)
def test_ari_matches_sklearn() -> None:
    """adjusted_rand_index matches sklearn.metrics.adjusted_rand_score (1e-12)."""
    import pandas as pd
    from sklearn.metrics import adjusted_rand_score

    from stockclusters.stability.ari import adjusted_rand_index

    a = pd.Series([0, 0, 1, 1], index=list("abcd"))
    b = pd.Series([1, 1, 0, 0], index=list("abcd"))
    assert abs(adjusted_rand_index(a, b) - adjusted_rand_score(a, b)) < 1e-12
