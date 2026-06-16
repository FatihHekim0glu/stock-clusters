"""Property tests for the correlation + clustering core (Group A).

Covers the invariants the brief pre-registers for the clustering science:

- permutation equivariance: relabelling/reordering assets permutes the cluster
  labels but leaves the partition (and cophenetic structure) invariant;
- scale invariance: rescaling each asset's returns by a positive constant leaves
  the correlation matrix, distance, and clustering unchanged;
- monotonicity of the dendrogram cut: ``k = N`` -> singletons, ``k = 1`` -> one
  cluster, and a finer cut never lowers the cluster count;
- seed determinism: the phase-randomized null, the gap selection, and K-means are
  byte-for-byte reproducible under a fixed seed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score

from stockclusters.clustering.embedding import rmt_signal_embedding
from stockclusters.clustering.hierarchical import cut_tree, hierarchical_clusters
from stockclusters.clustering.kmeans import kmeans_clusters
from stockclusters.clustering.selection import phase_randomize, select_k_gap
from stockclusters.correlation.distance import mantegna_distance
from stockclusters.correlation.estimate import correlation_matrix


def _dist(returns: pd.DataFrame) -> pd.DataFrame:
    return mantegna_distance(correlation_matrix(returns))


@pytest.mark.property
def test_permutation_equivariance_partition(k_blocks: pd.DataFrame) -> None:
    """Reordering assets permutes labels but preserves the partition (ARI == 1)."""
    dist = _dist(k_blocks)
    base = hierarchical_clusters(dist, n_clusters=4, method="average")

    perm = list(reversed(dist.columns.tolist()))
    permuted = k_blocks[perm]
    dist_perm = _dist(permuted)
    other = hierarchical_clusters(dist_perm, n_clusters=4, method="average")

    common = base.labels.index
    assert adjusted_rand_score(
        base.labels.loc[common].to_numpy(),
        other.labels.loc[common].to_numpy(),
    ) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.property
def test_scale_invariance(k_blocks: pd.DataFrame) -> None:
    """Per-asset positive rescaling leaves correlation, distance, labels unchanged."""
    rng = np.random.default_rng(7)
    scales = rng.uniform(0.5, 5.0, size=k_blocks.shape[1])
    scaled = k_blocks * scales

    c0 = correlation_matrix(k_blocks)
    c1 = correlation_matrix(scaled)
    assert np.allclose(c0.to_numpy(), c1.to_numpy(), atol=1e-12)

    r0 = hierarchical_clusters(mantegna_distance(c0), n_clusters=4, method="average")
    r1 = hierarchical_clusters(mantegna_distance(c1), n_clusters=4, method="average")
    assert adjusted_rand_score(r0.labels.to_numpy(), r1.labels.to_numpy()) == pytest.approx(
        1.0, abs=1e-12
    )


@pytest.mark.property
def test_cut_monotonicity(k_blocks: pd.DataFrame) -> None:
    """k=N -> singletons, k=1 -> one cluster, finer cut never lowers the count."""
    dist = _dist(k_blocks)
    n = dist.shape[0]
    labels = dist.columns.astype(str).tolist()
    link = hierarchical_clusters(dist, n_clusters=2, method="average").linkage
    assert link is not None

    singletons = cut_tree(link, n_clusters=n, labels=labels)
    assert int(singletons.nunique()) == n

    one = cut_tree(link, n_clusters=1, labels=labels)
    assert int(one.nunique()) == 1

    prev = 0
    for k in range(1, n + 1):
        count = int(cut_tree(link, n_clusters=k, labels=labels).nunique())
        assert count >= prev  # finer cut never decreases the cluster count
        assert count <= k
        prev = count


@pytest.mark.property
def test_phase_randomize_seed_determinism(k_blocks: pd.DataFrame) -> None:
    """Same seed -> identical surrogate; different seed -> different surrogate."""
    a = phase_randomize(k_blocks, seed=3)
    b = phase_randomize(k_blocks, seed=3)
    c = phase_randomize(k_blocks, seed=4)
    assert np.allclose(a, b, atol=0.0)  # exact reproduction
    assert not np.allclose(a, c)


@pytest.mark.property
def test_phase_randomize_preserves_marginal_spectrum(k_blocks: pd.DataFrame) -> None:
    """Surrogate preserves each asset's amplitude spectrum, kills cross-corr."""
    surr = phase_randomize(k_blocks, seed=0)
    amp_orig = np.abs(np.fft.rfft(k_blocks.to_numpy(), axis=0))
    amp_surr = np.abs(np.fft.rfft(surr, axis=0))
    assert np.allclose(amp_orig, amp_surr, atol=1e-8)

    # Within-block correlation present in the original is largely destroyed.
    orig_corr = float(np.corrcoef(k_blocks.to_numpy().T)[0, 1])
    surr_corr = float(np.corrcoef(surr.T)[0, 1])
    assert orig_corr > 0.4
    assert abs(surr_corr) < 0.25


@pytest.mark.property
def test_select_k_gap_seed_determinism(k_blocks: pd.DataFrame) -> None:
    """Gap selection is reproducible under a fixed seed."""
    dist = _dist(k_blocks)
    g1 = select_k_gap(k_blocks, dist, k_min=2, k_max=6, n_references=5, seed=0)
    g2 = select_k_gap(k_blocks, dist, k_min=2, k_max=6, n_references=5, seed=0)
    assert g1.k_selected == g2.k_selected
    assert g1.gap == g2.gap
    assert g1.gap_se == g2.gap_se
    assert g1.log_wk == g2.log_wk


@pytest.mark.property
def test_kmeans_seed_determinism(k_blocks: pd.DataFrame) -> None:
    """Fixed seed reproduces the same K-means inertia and partition."""
    emb = rmt_signal_embedding(correlation_matrix(k_blocks), n_obs=len(k_blocks))
    a = kmeans_clusters(emb, n_clusters=4, seed=0)
    b = kmeans_clusters(emb, n_clusters=4, seed=0)
    assert a.meta["inertia"] == pytest.approx(b.meta["inertia"], abs=1e-12)
    assert adjusted_rand_score(a.labels.to_numpy(), b.labels.to_numpy()) == pytest.approx(
        1.0, abs=1e-12
    )


@pytest.mark.property
def test_gap_selects_one_cluster_on_pure_noise(pure_noise: pd.DataFrame) -> None:
    """Honest null: with k_min=1 the gap statistic should not split pure noise."""
    dist = _dist(pure_noise)
    g = select_k_gap(pure_noise, dist, k_min=1, k_max=6, n_references=8, seed=0)
    assert g.k_selected == 1
