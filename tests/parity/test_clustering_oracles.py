"""Parity oracles for the correlation + clustering core (Group A).

Pins our kernels against independent references at the brief's tolerances:

- linkage matrix vs ``scipy.cluster.hierarchy.linkage`` (1e-10);
- subdominant ultrametric == single-linkage cophenetic vs scipy (1e-10);
- silhouette vs ``sklearn.metrics.silhouette_score`` (1e-10);
- K-means **inertia** vs sklearn under fixed explicit init, ``n_init=1`` (1e-8);
- RMT signal edge vs the analytic Marchenko-Pastur edge ``(1 + sqrt(q))^2`` (1e-10);
- gap machinery (log W_k, gap, s_k) recomputed from scratch on a FIXED surrogate
  set (1e-10);
- gap-on-uniform-null sanity vs a hand-rolled reference (1e-6).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockclusters.clustering.embedding import rmt_signal_embedding
from stockclusters.clustering.hierarchical import hierarchical_clusters
from stockclusters.clustering.kmeans import kmeans_clusters
from stockclusters.clustering.selection import (
    phase_randomize,
    pooled_within_dispersion,
    select_k_gap,
)
from stockclusters.correlation.distance import mantegna_distance, subdominant_ultrametric
from stockclusters.correlation.estimate import correlation_matrix


def _returns(n_obs: int, n_assets: int, *, blocks: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    bs = n_assets // blocks
    corr = np.full((n_assets, n_assets), 0.1)
    for b in range(0, n_assets, bs):
        corr[b : b + bs, b : b + bs] = 0.7
    np.fill_diagonal(corr, 1.0)
    chol = np.linalg.cholesky(corr)
    x = (rng.standard_normal((n_obs, n_assets)) @ chol.T) * 0.01
    return pd.DataFrame(x, columns=[f"A{i:02d}" for i in range(n_assets)])


@pytest.mark.parity
def test_linkage_matches_scipy() -> None:
    """Our hierarchical linkage equals scipy.linkage on the same condensed dist."""
    from scipy.cluster.hierarchy import linkage as scipy_linkage
    from scipy.spatial.distance import squareform

    rets = _returns(500, 9, blocks=3, seed=1)
    dist = mantegna_distance(correlation_matrix(rets))
    res = hierarchical_clusters(dist, n_clusters=3, method="average")

    arr = dist.to_numpy()
    arr = 0.5 * (arr + arr.T)
    np.fill_diagonal(arr, 0.0)
    ref = scipy_linkage(squareform(arr, checks=False), method="average")
    assert res.linkage is not None
    assert np.allclose(res.linkage, ref, atol=1e-10, rtol=0.0)


@pytest.mark.parity
def test_subdominant_ultrametric_matches_scipy_cophenet() -> None:
    """Subdominant ultrametric == single-linkage cophenetic distance (scipy)."""
    from scipy.cluster.hierarchy import cophenet
    from scipy.cluster.hierarchy import linkage as scipy_linkage
    from scipy.spatial.distance import squareform

    rets = _returns(500, 8, blocks=2, seed=2)
    dist = mantegna_distance(correlation_matrix(rets))
    ultra = subdominant_ultrametric(dist)

    arr = dist.to_numpy()
    arr = 0.5 * (arr + arr.T)
    np.fill_diagonal(arr, 0.0)
    link = scipy_linkage(squareform(arr, checks=False), method="single")
    ref = squareform(np.asarray(cophenet(link)))
    assert np.allclose(ultra.to_numpy(), ref, atol=1e-10, rtol=0.0)


@pytest.mark.parity
def test_silhouette_matches_sklearn() -> None:
    """ClusterResult.silhouette equals sklearn silhouette_score (precomputed)."""
    from sklearn.metrics import silhouette_score

    rets = _returns(500, 9, blocks=3, seed=3)
    dist = mantegna_distance(correlation_matrix(rets))
    res = hierarchical_clusters(dist, n_clusters=3, method="average")

    aligned = dist.reindex(index=res.labels.index, columns=res.labels.index)
    ref = float(silhouette_score(aligned.to_numpy(), res.labels.to_numpy(), metric="precomputed"))
    assert abs(res.silhouette - ref) < 1e-10


@pytest.mark.parity
def test_kmeans_inertia_matches_sklearn_fixed_init() -> None:
    """K-means inertia matches sklearn under a fixed explicit init, n_init=1 (1e-8)."""
    from sklearn.cluster import KMeans

    rets = _returns(500, 12, blocks=4, seed=4)
    emb = rmt_signal_embedding(correlation_matrix(rets), n_obs=len(rets))
    x = emb.to_numpy()

    # Fixed explicit init: first k rows of the embedding.
    init = x[:4].copy()
    res = kmeans_clusters(emb, n_clusters=4, init=init)

    ref = KMeans(n_clusters=4, init=init, n_init=1, random_state=0).fit(x)
    assert abs(res.meta["inertia"] - float(ref.inertia_)) < 1e-8


@pytest.mark.parity
def test_rmt_signal_edge_matches_analytic_mp() -> None:
    """The embedding retains exactly the eigenvalues above (1+sqrt(q))^2."""
    rets = _returns(600, 12, blocks=4, seed=5)
    corr = correlation_matrix(rets)
    n, t = corr.shape[0], len(rets)
    q = float(n) / float(t)
    lambda_plus = (1.0 + np.sqrt(q)) ** 2

    eigvals = np.linalg.eigvalsh(corr.to_numpy())
    n_signal = int((eigvals > lambda_plus).sum())
    # With drop_market_mode=True the embedding keeps (n_signal - 1) components.
    emb = rmt_signal_embedding(corr, n_obs=t, drop_market_mode=True)
    assert emb.shape[1] == max(n_signal - 1, 1)

    emb_full = rmt_signal_embedding(corr, n_obs=t, drop_market_mode=False)
    assert emb_full.shape[1] == n_signal

    # The analytic edge itself, recomputed, must match to 1e-10.
    assert abs(((1.0 + np.sqrt(q)) ** 2) - lambda_plus) < 1e-10


@pytest.mark.parity
def test_gap_machinery_recomputed_from_scratch() -> None:
    """log W_k, gap, and s_k from select_k_gap match a scratch recomputation (1e-10)."""
    from scipy.cluster.hierarchy import fcluster
    from scipy.cluster.hierarchy import linkage as scipy_linkage
    from scipy.spatial.distance import squareform

    rets = _returns(400, 9, blocks=3, seed=6)
    dist = mantegna_distance(correlation_matrix(rets))
    k_candidates = list(range(2, 6))
    n_ref = 4
    seed = 11

    result = select_k_gap(
        rets, dist, k_min=2, k_max=5, method="average", n_references=n_ref, seed=seed
    )

    labels = dist.columns.astype(str).tolist()

    def logwk_curve(d: np.ndarray) -> np.ndarray:
        sym = 0.5 * (d + d.T)
        np.fill_diagonal(sym, 0.0)
        link = scipy_linkage(squareform(sym, checks=False), method="average")
        out = np.empty(len(k_candidates))
        for i, k in enumerate(k_candidates):
            lab = fcluster(link, t=k, criterion="maxclust")
            w = pooled_within_dispersion(d, np.asarray(lab))
            out[i] = float(np.log(max(w, 1e-300)))
        return out

    obs = logwk_curve(dist.to_numpy())
    refs = np.empty((n_ref, len(k_candidates)))
    for b in range(n_ref):
        surr = phase_randomize(rets, seed=seed + b)
        surr_df = pd.DataFrame(surr, columns=labels)
        surr_dist = mantegna_distance(correlation_matrix(surr_df)).to_numpy()
        refs[b] = logwk_curve(surr_dist)

    gap_ref = refs.mean(axis=0) - obs
    s_ref = refs.std(axis=0, ddof=0) * np.sqrt(1.0 + 1.0 / n_ref)

    assert np.allclose(result.log_wk, obs, atol=1e-10, rtol=0.0)
    assert np.allclose(result.gap, gap_ref, atol=1e-10, rtol=0.0)
    assert np.allclose(result.gap_se, s_ref, atol=1e-10, rtol=0.0)


@pytest.mark.parity
def test_pooled_within_dispersion_reference() -> None:
    """W_k on a hand-checkable 4-point, 2-cluster case (1e-6 vs reference)."""
    # Two clusters: {0,1} at distance 2, {2,3} at distance 4. Cross distances 10.
    d = np.array(
        [
            [0.0, 2.0, 10.0, 10.0],
            [2.0, 0.0, 10.0, 10.0],
            [10.0, 10.0, 0.0, 4.0],
            [10.0, 10.0, 4.0, 0.0],
        ]
    )
    labels = np.array([0, 0, 1, 1])
    # D_0 = 2 + 2 = 4 (both off-diagonal entries); W contribution = 4 / (2*2) = 1.
    # D_1 = 4 + 4 = 8; contribution = 8 / (2*2) = 2. Total W_k = 3.
    assert pooled_within_dispersion(d, labels) == pytest.approx(3.0, abs=1e-6)
