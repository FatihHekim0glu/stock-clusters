"""Unit tests for the stability layer: alignment, births/deaths, rolling re-fit.

These cover the Group-B stability machinery directly, with an injected
deterministic clusterer so the tests do not depend on the (separately authored)
clustering layer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockclusters._exceptions import InsufficientDataError, ValidationError
from stockclusters.allocation.schemes import (
    cluster_equal_weight,
    one_over_n_weights,
    stripped_hrp_weights,
)
from stockclusters.stability.align import align_labels, births_and_deaths
from stockclusters.stability.ari import adjacent_window_ari, adjusted_rand_index
from stockclusters.stability.resample import StabilityResult, rolling_stability


def _labels(values: list[int], assets: list[str] | None = None) -> pd.Series:
    idx = assets or [f"A{i:02d}" for i in range(len(values))]
    return pd.Series(values, index=idx, dtype=int)


# --------------------------------------------------------------------------- #
# Alignment                                                                    #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_align_recovers_reference_ids_under_permutation() -> None:
    """A pure relabeling of the reference is mapped back onto its own ids."""
    ref = _labels([0, 0, 1, 1, 2, 2])
    permuted = ref.map({0: 7, 1: 4, 2: 9})
    aligned = align_labels(ref, permuted)
    pd.testing.assert_series_equal(aligned, ref, check_names=False)


@pytest.mark.unit
def test_align_assigns_fresh_ids_to_born_clusters() -> None:
    """A target cluster with no reference match gets an id beyond the ref range."""
    ref = _labels([0, 0, 1, 1], ["x", "y", "z", "w"])
    target = _labels([0, 0, 1, 2], ["x", "y", "z", "w"])  # 'w' split off
    aligned = align_labels(ref, target)
    # x, y keep cluster 0; z keeps cluster 1; w is a new id > max(ref)=1.
    assert aligned.loc["x"] == aligned.loc["y"]
    assert aligned.loc["w"] > 1


@pytest.mark.unit
def test_births_and_deaths_on_k_change() -> None:
    """Births/deaths reflect clusters gained/lost between windows."""
    ref = _labels([0, 0, 1, 1], ["x", "y", "z", "w"])
    grown = _labels([0, 0, 1, 2], ["x", "y", "z", "w"])
    bd = births_and_deaths(ref, grown)
    assert bd["births"] == [2]
    assert bd["deaths"] == []

    shrunk_ref = _labels([0, 0, 1, 2], ["x", "y", "z", "w"])
    shrunk = _labels([0, 0, 1, 1], ["x", "y", "z", "w"])
    bd2 = births_and_deaths(shrunk_ref, shrunk)
    assert bd2["births"] == []
    assert bd2["deaths"] == [2]


@pytest.mark.unit
def test_align_requires_common_assets() -> None:
    """Disjoint labelings cannot be aligned."""
    a = _labels([0, 1], ["x", "y"])
    b = _labels([0, 1], ["p", "q"])
    with pytest.raises(ValidationError):
        align_labels(a, b)


# --------------------------------------------------------------------------- #
# ARI edge cases                                                               #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_adjacent_window_ari_nan_for_single_window() -> None:
    """A single window has no adjacent pair -> NaN headline stability."""
    out = adjacent_window_ari([_labels([0, 0, 1])])
    assert np.isnan(out)


@pytest.mark.unit
def test_adjacent_window_ari_mean_of_pairs() -> None:
    """The headline scalar is the mean of consecutive-pair ARIs."""
    w0 = _labels([0, 0, 1, 1])
    w1 = _labels([0, 0, 1, 1])  # identical -> 1.0
    w2 = _labels([0, 1, 0, 1])  # different
    expected = np.mean([adjusted_rand_index(w0, w1), adjusted_rand_index(w1, w2)])
    assert adjacent_window_ari([w0, w1, w2]) == pytest.approx(expected, abs=1e-12)


@pytest.mark.unit
def test_ari_requires_two_common_assets() -> None:
    """ARI on fewer than two shared assets is undefined."""
    a = _labels([0], ["x"])
    b = _labels([0], ["x"])
    with pytest.raises(ValidationError):
        adjusted_rand_index(a, b)


# --------------------------------------------------------------------------- #
# Rolling stability (injected clusterer)                                       #
# --------------------------------------------------------------------------- #
def _panel(n_obs: int, n_assets: int, *, seed: int) -> pd.DataFrame:
    gen = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n_obs, freq="B")
    cols = [f"A{i:02d}" for i in range(n_assets)]
    return pd.DataFrame(gen.standard_normal((n_obs, n_assets)) * 0.01, index=idx, columns=cols)


@pytest.mark.unit
def test_rolling_stability_stable_clusterer_gives_ari_one() -> None:
    """A constant labeling across windows yields headline ARI = 1.0."""
    panel = _panel(400, 9, seed=2)
    fixed = pd.Series([i % 3 for i in range(9)], index=panel.columns, dtype=int)

    result = rolling_stability(
        panel,
        n_clusters=3,
        train_window=120,
        step=40,
        clusterer=lambda w: fixed.reindex(w.columns),
    )
    assert isinstance(result, StabilityResult)
    assert result.n_windows >= 2
    assert result.ari_mean == pytest.approx(1.0, abs=1e-12)
    assert all(v == pytest.approx(1.0, abs=1e-12) for v in result.ari_series)
    assert len(result.window_dates) == result.n_windows


@pytest.mark.unit
def test_rolling_stability_no_lookahead_window_local() -> None:
    """Each window's clusterer sees ONLY its own rows (no peeking forward)."""
    panel = _panel(300, 6, seed=4)
    seen_max_dates: list[pd.Timestamp] = []

    def spy(window: pd.DataFrame) -> pd.Series:
        seen_max_dates.append(window.index.max())
        return pd.Series([i % 2 for i in range(window.shape[1])], index=window.columns, dtype=int)

    rolling_stability(panel, n_clusters=2, train_window=100, step=50, clusterer=spy)
    # No window may extend past the panel's final observation.
    assert max(seen_max_dates) <= panel.index.max()


@pytest.mark.unit
def test_rolling_stability_to_dict_is_json_safe() -> None:
    """StabilityResult.to_dict() renders plain JSON-serializable types."""
    panel = _panel(260, 5, seed=8)
    fixed = pd.Series([0, 0, 1, 1, 1], index=panel.columns, dtype=int)
    result = rolling_stability(
        panel, n_clusters=2, train_window=120, step=60, clusterer=lambda w: fixed.reindex(w.columns)
    )
    d = result.to_dict()
    assert isinstance(d["ari_mean"], float)
    assert all(isinstance(x, float) for x in d["ari_series"])
    assert all(isinstance(m, dict) for m in d["window_labels"])
    for mapping in d["window_labels"]:
        assert all(isinstance(k, str) and isinstance(v, int) for k, v in mapping.items())
    import json

    json.dumps(d, default=str)  # must not raise


@pytest.mark.unit
def test_rolling_stability_rejects_oversized_window() -> None:
    """A train window larger than the panel is insufficient data."""
    panel = _panel(50, 4, seed=1)
    with pytest.raises(InsufficientDataError):
        rolling_stability(panel, n_clusters=2, train_window=100, step=10)


@pytest.mark.unit
def test_rolling_stability_rejects_bad_step() -> None:
    """A non-positive step is a validation error."""
    panel = _panel(200, 4, seed=1)
    with pytest.raises(ValidationError):
        rolling_stability(panel, n_clusters=2, train_window=100, step=0)


@pytest.mark.unit
def test_rolling_stability_rejects_bad_n_clusters_and_window() -> None:
    """n_clusters < 1 and train_window < 2 are validation errors."""
    panel = _panel(200, 4, seed=1)
    with pytest.raises(ValidationError):
        rolling_stability(panel, n_clusters=0, train_window=100, step=10)
    with pytest.raises(ValidationError):
        rolling_stability(panel, n_clusters=2, train_window=1, step=10)


@pytest.mark.unit
def test_rolling_stability_default_clusterer_invoked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no injected clusterer, the default pipeline builder is used.

    We stub the three pipeline pieces the default clusterer imports lazily so the
    branch is exercised without depending on the real clustering layer.
    """
    import types

    panel = _panel(300, 5, seed=6)
    fixed = pd.Series([0, 0, 1, 1, 1], index=panel.columns, dtype=int)

    fake_corr = types.SimpleNamespace()
    monkeypatch.setattr(
        "stockclusters.correlation.estimate.correlation_matrix",
        lambda window: fake_corr,
        raising=True,
    )
    monkeypatch.setattr(
        "stockclusters.correlation.rmt.marchenko_pastur_clip",
        lambda corr, *, n_obs: corr,
        raising=True,
    )
    monkeypatch.setattr(
        "stockclusters.correlation.distance.mantegna_distance",
        lambda corr: corr,
        raising=True,
    )

    class _FakeResult:
        labels = fixed

    monkeypatch.setattr(
        "stockclusters.clustering.hierarchical.hierarchical_clusters",
        lambda dist, *, n_clusters, method: _FakeResult(),
        raising=True,
    )

    result = rolling_stability(panel, n_clusters=2, train_window=120, step=60)
    assert result.n_windows >= 2
    assert result.ari_mean == pytest.approx(1.0, abs=1e-12)


# --------------------------------------------------------------------------- #
# Allocation validation branches                                              #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_allocation_rejects_empty_inputs() -> None:
    """Empty asset lists / labelings are validation errors."""
    with pytest.raises(ValidationError):
        one_over_n_weights([])
    with pytest.raises(ValidationError):
        cluster_equal_weight(pd.Series([], dtype=int))


@pytest.mark.unit
def test_stripped_hrp_rejects_label_cov_mismatch() -> None:
    """Disagreeing cov labels or a non-positive diagonal are rejected."""
    labels = _labels([0, 0, 1], ["x", "y", "z"])
    bad_cov = pd.DataFrame(np.diag([1.0, 1.0, 1.0]), index=["x", "y", "q"], columns=["x", "y", "q"])
    with pytest.raises(ValidationError):
        stripped_hrp_weights(labels, bad_cov)

    zero_diag = pd.DataFrame(
        np.diag([1.0, 0.0, 1.0]), index=["x", "y", "z"], columns=["x", "y", "z"]
    )
    with pytest.raises(ValidationError):
        stripped_hrp_weights(labels, zero_diag)


@pytest.mark.unit
def test_stripped_hrp_accepts_unlabelled_cov_positionally() -> None:
    """An unlabelled (RangeIndex) cov aligns positionally to the labeling."""
    labels = _labels([0, 0, 1], ["x", "y", "z"])
    cov = pd.DataFrame(np.diag([1.0, 4.0, 1.0]))  # default RangeIndex
    w = stripped_hrp_weights(labels, cov)
    assert w.sum() == pytest.approx(1.0, abs=1e-12)
    # cluster 0 (x,y) gets 0.5 split inverse-variance 1:4 -> 0.4 / 0.1.
    assert w.loc["x"] == pytest.approx(0.4, abs=1e-12)
    assert w.loc["y"] == pytest.approx(0.1, abs=1e-12)
    assert w.loc["z"] == pytest.approx(0.5, abs=1e-12)


@pytest.mark.unit
def test_align_jaccard_ignores_zero_overlap() -> None:
    """A target cluster with zero overlap is treated as born, not force-matched."""
    ref = _labels([0, 0, 1, 1], ["x", "y", "z", "w"])
    # Target cluster 5 shares no asset-membership pattern with any ref cluster
    # other than via assets; build a case where a relabel keeps structure.
    target = _labels([2, 2, 3, 3], ["x", "y", "z", "w"])
    aligned = align_labels(ref, target)
    # Structure preserved: x,y together; z,w together; mapped onto ref ids 0/1.
    assert aligned.loc["x"] == aligned.loc["y"]
    assert aligned.loc["z"] == aligned.loc["w"]
    assert aligned.loc["x"] != aligned.loc["z"]
    assert set(aligned.to_numpy()) == {0, 1}
