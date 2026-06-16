"""K-means clustering on the RMT-signal embedding.

K-means runs on the eigenvector embedding from
:mod:`stockclusters.clustering.embedding`, NOT on the raw Mantegna distances
(K-means assumes Euclidean geometry, which the embedding provides and the distance
matrix does not). Results are returned as a :class:`ClusterResult`.

REPRODUCIBILITY: seeding flows through ``seed`` so labels and inertia are
deterministic. The parity oracle pins **inertia** (not label identity) against
``sklearn`` under a fixed explicit init.

Importing this module has no side effects.
"""

from __future__ import annotations

from stockclusters._typing import MatrixLike
from stockclusters.clustering.hierarchical import ClusterResult

__all__ = ["kmeans_clusters"]


def kmeans_clusters(
    embedding: MatrixLike,
    *,
    n_clusters: int,
    seed: int = 0,
    n_init: int = 10,
) -> ClusterResult:
    r"""K-means clustering on an RMT-signal embedding.

    Runs ``sklearn.cluster.KMeans`` on the Euclidean ``embedding`` (rows = assets)
    and packages the result as a :class:`ClusterResult` with ``method="kmeans"``
    and ``linkage=None``.

    HONESTY REQUIREMENT: K-means runs on the embedding, never on the raw distance
    matrix. Determinism is seeded so a fixed ``seed`` reproduces the same inertia.

    Parameters
    ----------
    embedding:
        An ``N x d`` Euclidean embedding (output of
        :func:`stockclusters.clustering.embedding.rmt_signal_embedding`).
    n_clusters:
        The number of clusters ``k``.
    seed:
        Master seed for K-means initialization.
    n_init:
        Number of random initializations; the lowest-inertia run is kept.

    Returns
    -------
    ClusterResult
        The frozen clustering bundle (``linkage=None``).

    Raises
    ------
    ValidationError
        If ``embedding`` is empty or ``n_clusters`` is out of range.
    """
    raise NotImplementedError
