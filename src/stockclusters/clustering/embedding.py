"""RMT-signal eigenvector embedding of the correlation matrix.

Embeds assets into a low-dimensional Euclidean space spanned by the
*signal* eigenvectors of the correlation matrix — those whose eigenvalues exceed
the Marchenko-Pastur upper edge — excluding the market-mode eigenvector. K-means
(see :mod:`stockclusters.clustering.kmeans`) then runs on this embedding rather
than on the raw distances.

Importing this module has no side effects.
"""

from __future__ import annotations

import pandas as pd

from stockclusters._typing import MatrixLike

__all__ = ["rmt_signal_embedding"]


def rmt_signal_embedding(
    corr: MatrixLike,
    *,
    n_obs: int,
    drop_market_mode: bool = True,
    n_components: int | None = None,
) -> pd.DataFrame:
    r"""Embed assets using the RMT-signal eigenvectors of the correlation matrix.

    Eigendecomposes ``corr`` and keeps the eigenvectors whose eigenvalues exceed
    the Marchenko-Pastur upper edge :math:`(1 + \sqrt{q})^2` (the signal subspace),
    optionally dropping the top "market mode" eigenvector. Each asset is embedded
    at its (eigenvalue-scaled) loadings on the retained eigenvectors.

    Parameters
    ----------
    corr:
        An ``N x N`` correlation matrix.
    n_obs:
        The number of in-sample observations ``T`` (sets the MP edge via
        ``q = N / T``).
    drop_market_mode:
        If ``True``, exclude the largest-eigenvalue (market) eigenvector so the
        embedding captures sector/industry structure rather than the common mode.
    n_components:
        Optional explicit cap on the number of embedding dimensions; when ``None``
        all RMT-signal eigenvectors (post market-mode drop) are used.

    Returns
    -------
    pandas.DataFrame
        An ``N x d`` embedding, rows indexed by asset ticker, columns the retained
        signal components.

    Raises
    ------
    ValidationError
        If ``corr`` is not square or ``n_obs`` is non-positive.
    """
    raise NotImplementedError
