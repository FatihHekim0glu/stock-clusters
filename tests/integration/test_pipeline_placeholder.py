"""Integration slot: end-to-end clustering pipeline (placeholder).

Exercises the full fit-on-train, freeze, apply-OOS pipeline on the seeded
``k_blocks`` fixture and asserts the recovered clusters match truth above the
pinned ARI threshold. Marked xfail until the pipeline lands.
"""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.mark.integration
@pytest.mark.xfail(reason="clustering pipeline is a stub", strict=False)
def test_k_blocks_recovers_four_clusters(
    k_blocks: pd.DataFrame, k_blocks_truth: pd.Series
) -> None:
    """The k_blocks panel recovers its four planted clusters."""
    from stockclusters.clustering.hierarchical import hierarchical_clusters
    from stockclusters.correlation.distance import mantegna_distance
    from stockclusters.correlation.estimate import correlation_matrix

    corr = correlation_matrix(k_blocks)
    dist = mantegna_distance(corr)
    result = hierarchical_clusters(dist, n_clusters=4, method="average")
    assert result.n_clusters == 4
