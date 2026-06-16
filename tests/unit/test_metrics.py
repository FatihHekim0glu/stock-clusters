"""Unit tests for the diagnostic metrics (Group C).

Covers:

- ``silhouette_score`` parity vs ``sklearn.metrics.silhouette_score`` (precomputed)
  to ``1e-10``.
- ``cophenetic_correlation`` parity vs ``scipy.cluster.hierarchy.cophenet`` to
  ``1e-10``.
- ``modularity`` sanity: a partition matching the block structure scores
  meaningfully higher than a random partition, and clean blocks give a sizable
  positive ``Q``.
- ``ari_vs_gics`` correctness on the ``k_blocks`` recovery fixture: the
  ARI-vs-(ground-truth-as-GICS) clears a pinned threshold; perfect alignment is
  exactly ``1.0``; orthogonal labelings are ``~0``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockclusters._exceptions import ValidationError
from stockclusters._rng import make_rng
from stockclusters.clustering.hierarchical import hierarchical_clusters
from stockclusters.correlation.distance import mantegna_distance
from stockclusters.correlation.estimate import correlation_matrix
from stockclusters.metrics import (
    ari_vs_gics,
    cophenetic_correlation,
    modularity,
    silhouette_score,
)

#: Pinned recovery threshold: clusters on the clean k_blocks panel should
#: re-discover the (truth-as-GICS) partition near-perfectly. Conservative so the
#: guard is robust to sampling noise.
_ARI_RECOVERY_THRESHOLD = 0.8


def _block_distance(n: int, block_size: int, *, seed: int) -> tuple[pd.DataFrame, pd.Series]:
    """A random symmetric distance matrix and a block labeling for parity tests."""
    gen = make_rng(seed)
    assets = [f"A{i:02d}" for i in range(n)]
    m = gen.random((n, n))
    dist = (m + m.T) / 2.0
    np.fill_diagonal(dist, 0.0)
    labels = pd.Series([i // block_size for i in range(n)], index=assets, dtype=int)
    return pd.DataFrame(dist, index=assets, columns=assets), labels


@pytest.mark.unit
def test_silhouette_matches_sklearn() -> None:
    """silhouette_score matches sklearn (precomputed) to 1e-10."""
    from sklearn.metrics import silhouette_score as sk_silhouette

    dist, labels = _block_distance(12, 3, seed=11)
    mine = silhouette_score(dist, labels)
    theirs = float(sk_silhouette(dist.to_numpy(), labels.to_numpy(), metric="precomputed"))
    assert abs(mine - theirs) < 1e-10


@pytest.mark.unit
def test_silhouette_label_order_independent() -> None:
    """Shuffling the label Series index leaves the silhouette unchanged."""
    from sklearn.metrics import silhouette_score as sk_silhouette

    dist, labels = _block_distance(9, 3, seed=12)
    shuffled = labels.sample(frac=1.0, random_state=3)
    mine = silhouette_score(dist, shuffled)
    theirs = float(sk_silhouette(dist.to_numpy(), labels.to_numpy(), metric="precomputed"))
    assert abs(mine - theirs) < 1e-10


@pytest.mark.unit
def test_silhouette_rejects_single_cluster() -> None:
    """A one-cluster labeling has no nearest-other-cluster: must raise."""
    dist, _ = _block_distance(6, 3, seed=13)
    one = pd.Series(0, index=dist.index, dtype=int)
    with pytest.raises(ValidationError):
        silhouette_score(dist, one)


@pytest.mark.unit
def test_cophenetic_matches_scipy() -> None:
    """cophenetic_correlation matches scipy.cluster.hierarchy.cophenet to 1e-10."""
    from scipy.cluster.hierarchy import cophenet
    from scipy.cluster.hierarchy import linkage as sp_linkage
    from scipy.spatial.distance import squareform

    dist, _ = _block_distance(10, 2, seed=14)
    condensed = squareform(dist.to_numpy(), checks=False)
    for method in ("average", "single", "complete"):
        link = sp_linkage(condensed, method=method)
        mine = cophenetic_correlation(link, dist)
        theirs, _ = cophenet(link, condensed)
        assert abs(mine - float(theirs)) < 1e-10, method


@pytest.mark.unit
def test_cophenetic_rejects_wrong_linkage_shape() -> None:
    """A linkage with the wrong row count for the matrix size must raise."""
    from scipy.cluster.hierarchy import linkage as sp_linkage
    from scipy.spatial.distance import squareform

    dist, _ = _block_distance(8, 2, seed=15)
    link = sp_linkage(squareform(dist.to_numpy(), checks=False), method="average")
    smaller, _ = _block_distance(6, 2, seed=16)
    with pytest.raises(ValidationError):
        cophenetic_correlation(link, smaller)


@pytest.mark.unit
def test_modularity_sane_on_block_structure() -> None:
    """Block-aligned labels score higher Q than random labels; clean blocks > 0.3."""
    n, block_size = 12, 3
    assets = [f"A{i:02d}" for i in range(n)]
    corr = np.full((n, n), 0.05)
    for b in range(0, n, block_size):
        corr[b : b + block_size, b : b + block_size] = 0.8
    np.fill_diagonal(corr, 1.0)
    corr_df = pd.DataFrame(corr, index=assets, columns=assets)

    truth = pd.Series([i // block_size for i in range(n)], index=assets, dtype=int)
    gen = make_rng(17)
    random_labels = pd.Series(gen.integers(0, 4, n), index=assets, dtype=int)

    q_truth = modularity(truth, corr_df)
    q_random = modularity(random_labels, corr_df)

    assert q_truth > q_random
    assert q_truth > 0.3


@pytest.mark.unit
def test_modularity_zero_on_no_positive_edges() -> None:
    """A correlation matrix with no positive off-diagonal weight gives Q == 0."""
    assets = list("abcd")
    arr = np.full((4, 4), -0.2)
    np.fill_diagonal(arr, 1.0)
    corr = pd.DataFrame(arr, index=assets, columns=assets)
    labels = pd.Series([0, 0, 1, 1], index=assets, dtype=int)
    # All off-diagonal correlations are negative, so the thresholded graph has no
    # positive edges: Newman modularity is exactly zero.
    assert modularity(labels, corr) == 0.0


@pytest.mark.unit
def test_ari_vs_gics_recovers_truth_on_k_blocks(
    k_blocks: pd.DataFrame, k_blocks_truth: pd.Series
) -> None:
    """Clustering the k_blocks panel re-discovers the truth partition (ARI >= pinned)."""
    corr = correlation_matrix(k_blocks)
    dist = mantegna_distance(corr)
    result = hierarchical_clusters(dist, n_clusters=4, method="average")

    # Treat the ground-truth blocks as the "GICS sectors" the clusters re-discover.
    gics = {str(a): f"S{int(s)}" for a, s in k_blocks_truth.items()}
    ari = ari_vs_gics(result.labels, gics)
    assert ari >= _ARI_RECOVERY_THRESHOLD


@pytest.mark.unit
def test_ari_vs_gics_perfect_and_orthogonal() -> None:
    """Perfect cluster/GICS alignment -> 1.0; an orthogonal labeling -> near 0."""
    assets = [f"A{i:02d}" for i in range(12)]
    labels = pd.Series([i // 3 for i in range(12)], index=assets, dtype=int)

    perfect = {a: f"S{labels.loc[a]}" for a in assets}
    assert abs(ari_vs_gics(labels, perfect) - 1.0) < 1e-12

    # A GICS map that cuts across every cluster (4 clusters vs 3 cyclic sectors)
    # has no real agreement: the chance-corrected ARI is near zero (and may go
    # slightly negative, i.e. worse than random) — well below any "agreement" bar.
    orthogonal = {a: f"S{i % 3}" for i, a in enumerate(assets)}
    assert ari_vs_gics(labels, orthogonal) < 0.3


@pytest.mark.unit
def test_ari_vs_gics_uses_only_shared_assets() -> None:
    """Assets missing from the GICS map are dropped; too few shared assets raises."""
    assets = [f"A{i:02d}" for i in range(6)]
    labels = pd.Series([0, 0, 1, 1, 2, 2], index=assets, dtype=int)

    # Only one asset carries a sector -> cannot form a partition -> raise.
    with pytest.raises(ValidationError):
        ari_vs_gics(labels, {"A00": "Tech"})
