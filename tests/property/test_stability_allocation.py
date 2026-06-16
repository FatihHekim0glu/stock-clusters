"""Property tests for the stability + allocation + verdict layer (Group B).

Covers the invariants the brief pins for this layer:

- ARI matches ``sklearn.metrics.adjusted_rand_score`` to ``1e-12`` (Hypothesis).
- No-lookahead future-perturbation invariance: perturbing post-cutoff returns
  leaves the train-window labels AND the shift(1)-applied weights unchanged.
- The three diversification strategies share an IDENTICAL post-purge/embargo OOS
  date index, asserted BEFORE the Memmel-JK test runs.
- Allocation schemes return non-negative simplex weights; stripped-HRP is
  inverse-variance within cluster and equal across clusters.
- The verdict is a pure, honest function: it cannot return "beat" when the
  Memmel-JK p-value is insignificant or the deflated Sharpe is non-positive.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from sklearn.metrics import adjusted_rand_score

from stockclusters.allocation.schemes import (
    cluster_equal_weight,
    one_over_n_weights,
    run_diversification,
    stripped_hrp_weights,
)
from stockclusters.evaluation.verdict import ClusteringVerdict, derive_clustering_verdict
from stockclusters.stability.ari import adjusted_rand_index

_ARI_TOL = 1e-12


def _labels(values: list[int]) -> pd.Series:
    return pd.Series(values, index=[f"A{i:02d}" for i in range(len(values))], dtype=int)


# --------------------------------------------------------------------------- #
# ARI parity vs scikit-learn (1e-12)                                          #
# --------------------------------------------------------------------------- #
@pytest.mark.property
@given(
    arrays=hnp.arrays(
        dtype=np.int64,
        shape=st.integers(min_value=2, max_value=40),
        elements=st.integers(min_value=0, max_value=5),
    ),
    second=st.data(),
)
@settings(max_examples=200, deadline=None)
def test_ari_matches_sklearn_1e_12(arrays: np.ndarray, second: st.DataObject) -> None:
    """Hand-rolled ARI agrees with ``adjusted_rand_score`` to 1e-12."""
    n = arrays.shape[0]
    b = second.draw(
        hnp.arrays(
            dtype=np.int64,
            shape=n,
            elements=st.integers(min_value=0, max_value=5),
        )
    )
    labels_a = _labels(arrays.tolist())
    labels_b = _labels(b.tolist())

    mine = adjusted_rand_index(labels_a, labels_b)
    ref = float(adjusted_rand_score(arrays, b))
    assert abs(mine - ref) <= _ARI_TOL


@pytest.mark.property
def test_ari_identity_and_relabel_invariance() -> None:
    """ARI is 1.0 for identical partitions and invariant to id permutation."""
    a = _labels([0, 0, 1, 1, 2, 2, 2])
    assert adjusted_rand_index(a, a) == pytest.approx(1.0, abs=_ARI_TOL)

    relabeled = a.map({0: 5, 1: 9, 2: 0})
    assert adjusted_rand_index(a, relabeled) == pytest.approx(1.0, abs=_ARI_TOL)


# --------------------------------------------------------------------------- #
# Allocation: non-negative simplex weights                                    #
# --------------------------------------------------------------------------- #
@pytest.mark.property
@given(
    n_assets=st.integers(min_value=2, max_value=30),
    k=st.integers(min_value=1, max_value=8),
)
@settings(max_examples=100, deadline=None)
def test_one_over_n_and_cluster_ew_on_simplex(n_assets: int, k: int) -> None:
    """1/N and cluster-EW weights are non-negative and sum to one."""
    k = min(k, n_assets)
    labels = _labels([i % k for i in range(n_assets)])
    assets = list(labels.index)

    w1 = one_over_n_weights(assets)
    assert (w1 >= 0).all()
    assert w1.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.ptp(w1.to_numpy()) == pytest.approx(0.0, abs=1e-12)  # exactly equal

    wc = cluster_equal_weight(labels)
    assert (wc >= 0).all()
    assert wc.sum() == pytest.approx(1.0, abs=1e-12)
    # Each present cluster receives exactly 1/K of the budget.
    per_cluster = wc.groupby(labels).sum()
    assert np.allclose(per_cluster.to_numpy(), 1.0 / labels.nunique(), atol=1e-12)


@pytest.mark.property
@given(
    variances=hnp.arrays(
        dtype=np.float64,
        shape=st.integers(min_value=2, max_value=20),
        elements=st.floats(min_value=0.1, max_value=10.0, allow_nan=False),
    )
)
@settings(max_examples=100, deadline=None)
def test_stripped_hrp_is_inverse_variance_within_equal_across(
    variances: np.ndarray,
) -> None:
    """Stripped-HRP weights are inverse-variance within and equal across clusters."""
    n = variances.shape[0]
    # Two non-empty clusters: first asset in cluster 0, the rest in cluster 1,
    # so both clusters are always populated (even at n == 2).
    labels = _labels([0 if i == 0 else 1 for i in range(n)])
    assets = list(labels.index)
    cov = pd.DataFrame(np.diag(variances), index=assets, columns=assets)

    w = stripped_hrp_weights(labels, cov)
    assert (w >= 0).all()
    assert w.sum() == pytest.approx(1.0, abs=1e-12)

    # Equal budget across the two clusters.
    per_cluster = w.groupby(labels).sum()
    assert np.allclose(per_cluster.to_numpy(), 0.5, atol=1e-12)

    # Within each cluster the weights are proportional to 1 / variance.
    for cid in labels.unique():
        members = labels.index[labels == cid]
        sub_w = w.loc[members].to_numpy()
        sub_v = pd.Series(np.diag(cov.loc[members, members].to_numpy()))
        inv_v = (1.0 / sub_v).to_numpy()
        expected = 0.5 * inv_v / inv_v.sum()
        assert np.allclose(sub_w, expected, atol=1e-12)


# --------------------------------------------------------------------------- #
# Verdict honesty (pure function truth table)                                 #
# --------------------------------------------------------------------------- #
@pytest.mark.property
@given(
    p=st.floats(min_value=0.0, max_value=1.0),
    dsr=st.floats(min_value=-2.0, max_value=2.0),
    diff=st.floats(min_value=-2.0, max_value=2.0),
)
@settings(max_examples=300, deadline=None)
def test_verdict_cannot_beat_without_evidence(p: float, dsr: float, diff: float) -> None:
    """The verdict NEVER claims "beat" while p is insignificant or DSR <= 0."""
    verdict = derive_clustering_verdict(p, dsr, diff)
    if p >= 0.05 or dsr <= 0.0:
        assert verdict is ClusteringVerdict.NO_SIGNIFICANT_DIFFERENCE
    elif diff > 0:
        assert verdict is ClusteringVerdict.CLUSTERS_BEAT_1N
    else:
        assert verdict is ClusteringVerdict.CLUSTERS_LOSE_TO_1N
    # The forbidden state, stated directly.
    if verdict is ClusteringVerdict.CLUSTERS_BEAT_1N:
        assert p < 0.05 and dsr > 0.0 and diff > 0.0


# --------------------------------------------------------------------------- #
# No-lookahead future-perturbation invariance                                 #
# --------------------------------------------------------------------------- #
def _block_panel(n_obs: int, *, seed: int) -> tuple[pd.DataFrame, pd.Series]:
    """A seeded 4-block-of-3 return panel plus its ground-truth labels."""
    n_assets, bs = 12, 3
    within, across = 0.75, 0.10
    corr = np.full((n_assets, n_assets), across)
    for b in range(0, n_assets, bs):
        corr[b : b + bs, b : b + bs] = within
    np.fill_diagonal(corr, 1.0)
    gen = np.random.default_rng(seed)
    chol = np.linalg.cholesky(corr)
    z = gen.standard_normal((n_obs, n_assets))
    idx = pd.date_range("2020-01-01", periods=n_obs, freq="B")
    cols = [f"A{i:02d}" for i in range(n_assets)]
    panel = pd.DataFrame((z @ chol.T) * 0.01, index=idx, columns=cols)
    labels = pd.Series([i // bs for i in range(n_assets)], index=cols, dtype=int)
    return panel, labels


@pytest.mark.property
def test_no_lookahead_future_perturbation_leaves_weights_unchanged() -> None:
    """Perturbing post-cutoff returns must not change pre-cutoff applied weights.

    Runs the diversification backtest on a panel, then re-runs it on a panel whose
    rows STRICTLY AFTER a cutoff have been replaced by garbage. The weights applied
    (shift(1)) on every rebalance up to the cutoff must be byte-identical: the
    engine cannot have peeked forward.
    """
    from stockclusters.backtest.walk_forward import walk_forward_backtest

    panel, labels = _block_panel(360, seed=7)
    cutoff = 250

    def alloc_cluster_ew(window: pd.DataFrame) -> pd.Series:
        sub = labels.reindex(window.columns).dropna().astype(int)
        return cluster_equal_weight(sub).reindex(window.columns).fillna(0.0)

    base = walk_forward_backtest(panel, alloc_cluster_ew, lookback_window=60, cost_bps=5.0)

    perturbed = panel.copy()
    gen = np.random.default_rng(999)
    perturbed.iloc[cutoff:] = gen.standard_normal(perturbed.iloc[cutoff:].shape) * 0.5
    pert = walk_forward_backtest(perturbed, alloc_cluster_ew, lookback_window=60, cost_bps=5.0)

    # Weights decided at rebalances whose in-sample window ends at/before the
    # cutoff cannot depend on the perturbed (post-cutoff) rows.
    base_w = base.weights
    pert_w = pert.weights
    pre = [d for d in base_w.index if panel.index.get_loc(d) <= cutoff]
    assert pre, "expected at least one pre-cutoff rebalance"
    pd.testing.assert_frame_equal(base_w.loc[pre], pert_w.loc[pre])


@pytest.mark.property
def test_three_strategies_share_identical_oos_index() -> None:
    """run_diversification asserts an identical OOS index before Memmel-JK.

    If the guard were absent or the engine mis-aligned, the call would raise; a
    successful return proves the three net OOS series shared one index. We also
    assert the result's inference fields are finite scalars.
    """
    panel, labels = _block_panel(380, seed=11)
    res = run_diversification(panel, labels, lookback_window=60, n_trials=24, cost_bps=5.0)

    assert np.isfinite(res.one_over_n_sharpe)
    assert np.isfinite(res.cluster_ew_sharpe)
    assert np.isfinite(res.stripped_hrp_sharpe)
    assert 0.0 <= res.memmel_jk_pvalue <= 1.0
    assert 0.0 <= res.deflated_sharpe <= 1.0
    assert res.n_trials == 24


@pytest.mark.property
def test_identical_index_guard_fires_on_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """The identical-OOS-index guard raises when series indexes disagree.

    We monkeypatch the backtest at its source module (it is imported lazily inside
    the runner) so the stripped-HRP strategy returns a series on a shifted index;
    the guard must reject it with a ValidationError BEFORE any Memmel-JK call.
    """
    import stockclusters.backtest.walk_forward as wf
    from stockclusters._exceptions import ValidationError

    panel, labels = _block_panel(360, seed=3)

    real_backtest = wf.walk_forward_backtest
    calls = {"n": 0}

    def fake_backtest(returns: object, allocator: object, **kwargs: object) -> object:
        result = real_backtest(returns, allocator, **kwargs)  # type: ignore[arg-type]
        calls["n"] += 1
        if calls["n"] == 3:  # the third strategy (stripped-HRP) gets a bad index
            broken = result.oos_returns.copy()
            broken.index = broken.index + pd.Timedelta(days=1)
            object.__setattr__(result, "oos_returns", broken)
        return result

    monkeypatch.setattr(wf, "walk_forward_backtest", fake_backtest)
    with pytest.raises(ValidationError, match="non-identical OOS"):
        run_diversification(panel, labels, lookback_window=60, n_trials=8)
