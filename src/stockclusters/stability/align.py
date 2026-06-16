"""Cross-window cluster-label alignment.

Cluster integer labels are arbitrary, so comparing labelings across windows
requires matching cluster ids by membership overlap. This module aligns labels via
the Hungarian algorithm on a max-Jaccard cost matrix, and records the birth/death
of clusters when ``k`` changes between windows.

Importing this module has no side effects.
"""

from __future__ import annotations

import pandas as pd

__all__ = ["align_labels", "births_and_deaths"]


def align_labels(
    reference: pd.Series,
    target: pd.Series,
) -> pd.Series:
    r"""Relabel ``target`` clusters to best match ``reference`` (max-Jaccard).

    Builds a cluster-vs-cluster Jaccard-overlap matrix on the shared assets and
    solves the assignment (Hungarian) that maximizes total overlap, then relabels
    ``target`` accordingly. Unmatched target clusters (when ``k`` grew) receive
    fresh ids beyond the reference range.

    Parameters
    ----------
    reference:
        The reference labeling (e.g. the earlier window), indexed by asset.
    target:
        The labeling to align to ``reference``, indexed by asset.

    Returns
    -------
    pandas.Series
        ``target`` relabeled to align with ``reference``, indexed by asset.

    Raises
    ------
    ValidationError
        If the two labelings share fewer than one common asset.
    """
    raise NotImplementedError


def births_and_deaths(
    reference: pd.Series,
    target: pd.Series,
) -> dict[str, list[int]]:
    r"""Identify clusters that appear or disappear between two windows.

    After alignment, returns the cluster ids present in ``target`` but not matched
    in ``reference`` (births) and those present in ``reference`` but absent in
    ``target`` (deaths).

    Parameters
    ----------
    reference:
        The earlier-window labeling, indexed by asset.
    target:
        The later-window labeling, indexed by asset.

    Returns
    -------
    dict
        ``{"births": [...], "deaths": [...]}`` with cluster ids.

    Raises
    ------
    ValidationError
        If the two labelings share no common assets.
    """
    raise NotImplementedError
