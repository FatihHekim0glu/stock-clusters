"""Adjusted Rand Index — the headline cluster-stability scalar.

The Adjusted Rand Index (ARI) measures agreement between two labelings, corrected
for chance. The adjacent-window ARI (mean ARI between clusterings fit on
consecutive rolling windows) is the headline stability number surfaced in the
summary. ARI is also reused post-hoc to compare clusters against GICS sectors
(see :mod:`stockclusters.metrics`).

PARITY: validated against ``sklearn.metrics.adjusted_rand_score`` to ``1e-12``.

Importing this module has no side effects.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

__all__ = ["adjacent_window_ari", "adjusted_rand_index"]


def adjusted_rand_index(labels_a: pd.Series, labels_b: pd.Series) -> float:
    r"""Adjusted Rand Index between two labelings (chance-corrected).

    Aligns the two label Series on their shared index and computes the ARI:
    ``1.0`` for identical partitions, ``~0.0`` for random agreement, possibly
    negative for worse-than-random.

    PARITY REQUIREMENT: matches ``sklearn.metrics.adjusted_rand_score`` to
    ``1e-12`` on the shared support.

    Parameters
    ----------
    labels_a, labels_b:
        Two integer-label Series indexed by asset ticker. Compared on their
        intersected index.

    Returns
    -------
    float
        The Adjusted Rand Index.

    Raises
    ------
    ValidationError
        If the two labelings share fewer than two common assets.
    """
    raise NotImplementedError


def adjacent_window_ari(window_labels: Sequence[pd.Series]) -> float:
    r"""Mean ARI between clusterings of consecutive windows (headline scalar).

    Given an ordered sequence of per-window label Series, computes the ARI between
    each adjacent pair and returns their mean — the headline temporal-stability
    number.

    Parameters
    ----------
    window_labels:
        An ordered sequence of integer-label Series (one per rolling window),
        each indexed by asset ticker.

    Returns
    -------
    float
        The mean adjacent-window ARI. ``NaN`` (rendered ``None`` downstream) when
        fewer than two windows are supplied.

    Raises
    ------
    ValidationError
        If any element is not a label Series.
    """
    raise NotImplementedError
