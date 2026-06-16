"""Unit tests for the correlation + clustering core (Group A).

Covers correctness details the property/parity suites do not pin directly:

- log-return estimation never forward-fills before differencing (the honesty
  requirement) and rejects non-positive prices;
- correlation-matrix shape/domain guarantees and insufficient-data guards;
- the recovery guard: clustering recovers the planted k-block structure;
- the gap statistic recovers ``k = 4`` on the k_blocks fixture;
- validation errors on malformed inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score

from stockclusters._exceptions import InsufficientDataError, ValidationError
from stockclusters.clustering.embedding import rmt_signal_embedding
from stockclusters.clustering.hierarchical import hierarchical_clusters
from stockclusters.clustering.kmeans import kmeans_clusters
from stockclusters.clustering.selection import select_k_gap
from stockclusters.correlation.distance import (
    mantegna_distance,
    minimum_spanning_tree,
    subdominant_ultrametric,
)
from stockclusters.correlation.estimate import correlation_matrix, log_returns


@pytest.mark.unit
def test_log_returns_does_not_forward_fill() -> None:
    """A price gap (NaN) must stay NaN, NOT become a spurious zero return."""
    prices = pd.DataFrame(
        {"A": [100.0, np.nan, 110.0], "B": [50.0, 51.0, 52.0]},
        index=pd.date_range("2021-01-01", periods=3, freq="B"),
    )
    rets = log_returns(prices)
    # The differenced panel keeps the gap as NaN for asset A on both affected
    # rows; if we had forward-filled, A would show a fabricated 0.0 return.
    assert bool(rets["A"].isna().any())
    # Asset B has no gap and its returns are finite.
    assert bool(rets["B"].notna().all())


@pytest.mark.unit
def test_log_returns_exact_value() -> None:
    """log return == log(p_t / p_{t-1})."""
    prices = pd.DataFrame({"A": [100.0, 110.0]})
    rets = log_returns(prices)
    assert rets["A"].iloc[0] == pytest.approx(np.log(110.0 / 100.0), abs=1e-12)


@pytest.mark.unit
def test_log_returns_rejects_non_positive() -> None:
    """Non-positive prices are a data error (log of a ratio undefined)."""
    prices = pd.DataFrame({"A": [100.0, 0.0, 90.0]})
    with pytest.raises(ValidationError):
        log_returns(prices)


@pytest.mark.unit
def test_correlation_matrix_shape_and_domain(k_blocks: pd.DataFrame) -> None:
    """N x N, symmetric, unit diagonal, entries in [-1, 1]."""
    c = correlation_matrix(k_blocks)
    n = k_blocks.shape[1]
    assert c.shape == (n, n)
    assert np.allclose(c.to_numpy(), c.to_numpy().T)
    assert np.allclose(np.diag(c.to_numpy()), 1.0)
    assert float(c.to_numpy().min()) >= -1.0
    assert float(c.to_numpy().max()) <= 1.0


@pytest.mark.unit
def test_correlation_matrix_requires_two_assets() -> None:
    """A single-asset panel cannot form a correlation matrix."""
    with pytest.raises(ValidationError):
        correlation_matrix(pd.DataFrame({"A": [0.1, 0.2, 0.3]}))


@pytest.mark.unit
def test_correlation_matrix_min_periods_guard() -> None:
    """Too few overlapping observations raises InsufficientDataError."""
    df = pd.DataFrame({"A": [0.1, np.nan, np.nan], "B": [np.nan, 0.2, 0.3]})
    with pytest.raises(InsufficientDataError):
        correlation_matrix(df, min_periods=2)


@pytest.mark.unit
def test_mantegna_distance_rejects_out_of_domain() -> None:
    """Entries outside [-1, 1] are rejected."""
    bad = pd.DataFrame([[1.0, 1.5], [1.5, 1.0]], index=list("ab"), columns=list("ab"))
    with pytest.raises(ValidationError):
        mantegna_distance(bad)


@pytest.mark.unit
def test_mst_edge_count_and_sorting(k_blocks: pd.DataFrame) -> None:
    """The MST has N-1 edges, sorted by ascending weight."""
    dist = mantegna_distance(correlation_matrix(k_blocks))
    mst = minimum_spanning_tree(dist)
    assert len(mst) == k_blocks.shape[1] - 1
    assert list(mst.columns) == ["source", "target", "weight"]
    weights = mst["weight"].to_numpy()
    assert np.all(np.diff(weights) >= -1e-12)


@pytest.mark.unit
def test_subdominant_ultrametric_is_ultrametric(k_blocks: pd.DataFrame) -> None:
    """The subdominant ultrametric satisfies the strong (ultrametric) inequality."""
    dist = mantegna_distance(correlation_matrix(k_blocks))
    u = subdominant_ultrametric(dist).to_numpy()
    n = u.shape[0]
    # u_ik <= max(u_ij, u_jk) for all triples.
    for i in range(n):
        for j in range(n):
            for k in range(n):
                assert u[i, k] <= max(u[i, j], u[j, k]) + 1e-9


@pytest.mark.unit
def test_hierarchical_recovers_k_blocks(k_blocks: pd.DataFrame, k_blocks_truth: pd.Series) -> None:
    """Recovery guard: hierarchical clustering recovers the planted 4 blocks."""
    dist = mantegna_distance(correlation_matrix(k_blocks))
    res = hierarchical_clusters(dist, n_clusters=4, method="average")
    truth = k_blocks_truth.reindex(res.labels.index)
    ari = adjusted_rand_score(truth.to_numpy(), res.labels.to_numpy())
    assert ari >= 0.9  # planted structure is strong; expect near-perfect recovery


@pytest.mark.unit
def test_kmeans_recovers_k_blocks(k_blocks: pd.DataFrame, k_blocks_truth: pd.Series) -> None:
    """K-means on the RMT embedding also recovers the planted blocks."""
    emb = rmt_signal_embedding(correlation_matrix(k_blocks), n_obs=len(k_blocks))
    res = kmeans_clusters(emb, n_clusters=4, seed=0)
    truth = k_blocks_truth.reindex(res.labels.index)
    ari = adjusted_rand_score(truth.to_numpy(), res.labels.to_numpy())
    assert ari >= 0.8


@pytest.mark.unit
def test_gap_recovers_k_on_k_blocks(k_blocks: pd.DataFrame) -> None:
    """The pre-registered gap selector recovers k=4 on the 4-block fixture."""
    dist = mantegna_distance(correlation_matrix(k_blocks))
    g = select_k_gap(k_blocks, dist, k_min=2, k_max=8, n_references=10, seed=0)
    assert g.k_selected == 4
    assert g.n_trials == len(g.k_candidates) == 7


@pytest.mark.unit
def test_one_block_is_one_cluster(one_block_correlation: pd.DataFrame) -> None:
    """A single common factor should not be split by the gap statistic."""
    dist = mantegna_distance(correlation_matrix(one_block_correlation))
    g = select_k_gap(one_block_correlation, dist, k_min=1, k_max=6, n_references=8, seed=0)
    assert g.k_selected == 1


@pytest.mark.unit
def test_cluster_result_to_dict_is_jsonable(k_blocks: pd.DataFrame) -> None:
    """ClusterResult.to_dict() is plain-Python and JSON-serializable."""
    import json

    dist = mantegna_distance(correlation_matrix(k_blocks))
    res = hierarchical_clusters(dist, n_clusters=4, method="average")
    d = res.to_dict()
    assert isinstance(d["labels"], dict)
    assert all(isinstance(v, int) for v in d["labels"].values())
    assert isinstance(d["silhouette"], float)
    assert isinstance(d["linkage"], list)
    json.dumps(d)  # must not raise


@pytest.mark.unit
def test_hierarchical_rejects_bad_method(k_blocks: pd.DataFrame) -> None:
    """An unknown linkage method is rejected, never silently substituted."""
    dist = mantegna_distance(correlation_matrix(k_blocks))
    with pytest.raises(ValidationError):
        hierarchical_clusters(dist, n_clusters=4, method="complete")


@pytest.mark.unit
def test_hierarchical_rejects_out_of_range_k(k_blocks: pd.DataFrame) -> None:
    """k must satisfy 1 <= k <= N."""
    dist = mantegna_distance(correlation_matrix(k_blocks))
    with pytest.raises(ValidationError):
        hierarchical_clusters(dist, n_clusters=99, method="average")


@pytest.mark.unit
def test_select_k_gap_rejects_bad_range(k_blocks: pd.DataFrame) -> None:
    """k_max < k_min is rejected."""
    dist = mantegna_distance(correlation_matrix(k_blocks))
    with pytest.raises(ValidationError):
        select_k_gap(k_blocks, dist, k_min=5, k_max=2, n_references=2, seed=0)


@pytest.mark.unit
def test_embedding_n_components_cap(k_blocks: pd.DataFrame) -> None:
    """n_components caps the embedding dimension."""
    corr = correlation_matrix(k_blocks)
    emb = rmt_signal_embedding(corr, n_obs=len(k_blocks), n_components=2)
    assert emb.shape == (k_blocks.shape[1], 2)


@pytest.mark.unit
def test_embedding_keeps_market_mode_when_requested(k_blocks: pd.DataFrame) -> None:
    """drop_market_mode=False retains one extra (market) component."""
    corr = correlation_matrix(k_blocks)
    with_mm = rmt_signal_embedding(corr, n_obs=len(k_blocks), drop_market_mode=False)
    without_mm = rmt_signal_embedding(corr, n_obs=len(k_blocks), drop_market_mode=True)
    assert with_mm.shape[1] == without_mm.shape[1] + 1


@pytest.mark.unit
def test_embedding_rejects_bad_inputs(k_blocks: pd.DataFrame) -> None:
    """Non-positive n_obs and n_components are rejected; non-square corr too."""
    corr = correlation_matrix(k_blocks)
    with pytest.raises(ValidationError):
        rmt_signal_embedding(corr, n_obs=0)
    with pytest.raises(ValidationError):
        rmt_signal_embedding(corr, n_obs=len(k_blocks), n_components=0)
    with pytest.raises(ValidationError):
        rmt_signal_embedding(np.ones((3, 4)), n_obs=100)


@pytest.mark.unit
def test_embedding_fallback_on_pure_noise(pure_noise: pd.DataFrame) -> None:
    """Pure noise yields no signal eigenvectors -> a single fallback component."""
    corr = correlation_matrix(pure_noise)
    emb = rmt_signal_embedding(corr, n_obs=len(pure_noise), drop_market_mode=True)
    assert emb.shape[1] >= 1  # fallback guarantees a usable embedding


@pytest.mark.unit
def test_kmeans_rejects_out_of_range_k(k_blocks: pd.DataFrame) -> None:
    """K-means k must satisfy 1 <= k <= n_samples."""
    emb = rmt_signal_embedding(correlation_matrix(k_blocks), n_obs=len(k_blocks))
    with pytest.raises(ValidationError):
        kmeans_clusters(emb, n_clusters=999, seed=0)


@pytest.mark.unit
def test_select_k_gap_validation_guards(k_blocks: pd.DataFrame) -> None:
    """select_k_gap rejects mismatched assets, k_max>N, and n_references<1."""
    dist = mantegna_distance(correlation_matrix(k_blocks))
    n = k_blocks.shape[1]
    with pytest.raises(ValidationError):  # k_max exceeds asset count
        select_k_gap(k_blocks, dist, k_min=2, k_max=n + 5, n_references=2, seed=0)
    with pytest.raises(ValidationError):  # n_references < 1
        select_k_gap(k_blocks, dist, k_min=2, k_max=4, n_references=0, seed=0)
    with pytest.raises(ValidationError):  # k_min < 1
        select_k_gap(k_blocks, dist, k_min=0, k_max=4, n_references=2, seed=0)
    with pytest.raises(ValidationError):  # returns/dist asset-count mismatch
        select_k_gap(k_blocks.iloc[:, :3], dist, k_min=2, k_max=4, n_references=2, seed=0)


@pytest.mark.unit
def test_pooled_within_dispersion_singleton_skipped() -> None:
    """A singleton cluster contributes zero to W_k (no within-pair distances)."""
    from stockclusters.clustering.selection import pooled_within_dispersion

    d = np.array([[0.0, 2.0, 10.0], [2.0, 0.0, 10.0], [10.0, 10.0, 0.0]])
    labels = np.array([0, 0, 1])  # cluster 1 is a singleton
    # Only cluster 0 contributes: D_0 = 4, W = 4 / (2*2) = 1.
    assert pooled_within_dispersion(d, labels) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.unit
def test_distance_rejects_asymmetric_for_mst() -> None:
    """MST and ultrametric reject an asymmetric distance matrix."""
    bad = pd.DataFrame(
        [[0.0, 1.0, 2.0], [3.0, 0.0, 1.0], [2.0, 1.0, 0.0]],
        index=list("abc"),
        columns=list("abc"),
    )
    with pytest.raises(ValidationError):
        minimum_spanning_tree(bad)
    with pytest.raises(ValidationError):
        subdominant_ultrametric(bad)
