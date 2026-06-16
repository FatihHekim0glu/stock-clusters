r"""Mantegna correlation distance and network helpers.

The Mantegna (1999) distance

.. math::

    d_{ij} = \sqrt{2\,(1 - \rho_{ij})}

is a TRUE metric on the unit sphere: it satisfies non-negativity, symmetry, the
identity of indiscernibles (zero diagonal), and the triangle inequality. It is the
distance used to build correlation-based minimum spanning trees and the
subdominant ultrametric (single-linkage hierarchy).

HONESTY REQUIREMENT: the metric is :math:`\sqrt{2(1 - \rho)}`, NOT ``1 - rho`` and
NOT ``1 - |rho|``. The metric axioms are asserted with a Hypothesis property test.

Importing this module has no side effects.
"""

from __future__ import annotations

import pandas as pd

from stockclusters._typing import MatrixLike

__all__ = [
    "mantegna_distance",
    "minimum_spanning_tree",
    "subdominant_ultrametric",
]


def mantegna_distance(corr: MatrixLike) -> pd.DataFrame:
    r"""Mantegna distance :math:`d_{ij} = \sqrt{2(1 - \rho_{ij})}`.

    Maps a correlation matrix to a true-metric distance matrix:
    :math:`\rho = +1 \to d = 0`, :math:`\rho = 0 \to d = \sqrt{2}`,
    :math:`\rho = -1 \to d = 2`.

    HONESTY REQUIREMENT: this is :math:`\sqrt{2(1 - \rho)}`, the Mantegna metric —
    NOT ``1 - rho`` (not a metric) and NOT ``1 - |rho|`` (collapses anti-correlated
    pairs). The triangle inequality and the other metric axioms are property-tested.

    Parameters
    ----------
    corr:
        An ``N x N`` correlation matrix with unit diagonal and entries in
        ``[-1, 1]``.

    Returns
    -------
    pandas.DataFrame
        An ``N x N`` symmetric distance matrix with an exactly zero diagonal,
        labelled by asset.

    Raises
    ------
    ValidationError
        If ``corr`` is not square or contains entries outside ``[-1, 1]``.
    """
    raise NotImplementedError


def minimum_spanning_tree(dist: MatrixLike) -> pd.DataFrame:
    r"""Minimum spanning tree edge list of a distance matrix.

    Builds the MST of the fully-connected distance graph (Mantegna's asset graph).
    The MST is the backbone of the correlation network and the basis of the
    subdominant ultrametric (single linkage).

    Parameters
    ----------
    dist:
        An ``N x N`` symmetric distance matrix (output of
        :func:`mantegna_distance`).

    Returns
    -------
    pandas.DataFrame
        Edge list with columns ``["source", "target", "weight"]`` (``N - 1`` rows),
        sorted by ascending weight.

    Raises
    ------
    ValidationError
        If ``dist`` is not square/symmetric.
    """
    raise NotImplementedError


def subdominant_ultrametric(dist: MatrixLike) -> pd.DataFrame:
    r"""Subdominant ultrametric (single-linkage cophenetic) distance matrix.

    The subdominant ultrametric is the largest ultrametric dominated by ``dist``;
    it equals the single-linkage cophenetic distance and underlies the MST-based
    hierarchical clustering of the correlation network.

    Parameters
    ----------
    dist:
        An ``N x N`` symmetric distance matrix (output of
        :func:`mantegna_distance`).

    Returns
    -------
    pandas.DataFrame
        The ``N x N`` ultrametric distance matrix, symmetric with a zero diagonal,
        labelled by asset.

    Raises
    ------
    ValidationError
        If ``dist`` is not square/symmetric.
    """
    raise NotImplementedError
