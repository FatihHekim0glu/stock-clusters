"""Property tests for the Mantegna metric axioms.

The Mantegna distance ``d_ij = sqrt(2 (1 - rho_ij))`` is a TRUE metric on the unit
sphere. This suite asserts, via Hypothesis over random valid correlation matrices:

- non-negativity
- symmetry
- identity of indiscernibles (zero diagonal)
- the triangle inequality

plus the exact closed-form values at the canonical correlations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from stockclusters.correlation.distance import mantegna_distance


def _random_correlation(rng: np.random.Generator, n: int) -> np.ndarray:
    """A valid (PSD, unit-diagonal) correlation matrix of size ``n``."""
    a = rng.standard_normal((n, max(n, 3)))
    cov = a @ a.T
    d = np.sqrt(np.diag(cov))
    corr = cov / np.outer(d, d)
    corr = 0.5 * (corr + corr.T)
    np.clip(corr, -1.0, 1.0, out=corr)
    np.fill_diagonal(corr, 1.0)
    return corr


@pytest.mark.property
@given(seed=st.integers(min_value=0, max_value=2**31 - 1), n=st.integers(min_value=2, max_value=12))
@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_mantegna_metric_axioms(seed: int, n: int) -> None:
    """d_ij is non-negative, symmetric, zero-diagonal, and obeys the triangle ineq."""
    rng = np.random.default_rng(seed)
    corr = _random_correlation(rng, n)
    labels = [f"A{i}" for i in range(n)]
    dist = mantegna_distance(pd.DataFrame(corr, index=labels, columns=labels))
    d = dist.to_numpy()

    # Non-negativity.
    assert float(d.min()) >= -1e-12
    # Symmetry.
    assert np.allclose(d, d.T, atol=1e-12)
    # Identity of indiscernibles: exactly zero diagonal.
    assert np.allclose(np.diag(d), 0.0, atol=1e-12)
    # Triangle inequality: d_ik <= d_ij + d_jk for all i, j, k.
    triangle = d[:, None, :] - (d[:, :, None] + d[None, :, :])
    assert float(triangle.max()) <= 1e-9


@pytest.mark.property
def test_mantegna_exact_values() -> None:
    """Closed-form: rho=1->0, rho=0.5->1, rho=0->sqrt2, rho=-1->2."""
    corr = pd.DataFrame(
        [[1.0, 0.5, 0.0, -1.0], [0.5, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [-1.0, 0.0, 0.0, 1.0]],
        index=list("wxyz"),
        columns=list("wxyz"),
    )
    d = mantegna_distance(corr)
    assert d.iloc[0, 0] == pytest.approx(0.0, abs=1e-12)
    assert d.iloc[0, 1] == pytest.approx(1.0, abs=1e-12)
    assert d.iloc[0, 2] == pytest.approx(np.sqrt(2.0), abs=1e-12)
    assert d.iloc[0, 3] == pytest.approx(2.0, abs=1e-12)


@pytest.mark.property
def test_mantegna_is_not_one_minus_rho() -> None:
    """Guard the footgun: the metric is sqrt(2(1-rho)), never 1-rho or 1-|rho|."""
    corr = pd.DataFrame([[1.0, 0.5], [0.5, 1.0]], index=list("ab"), columns=list("ab"))
    d = mantegna_distance(corr).iloc[0, 1]
    assert d == pytest.approx(1.0, abs=1e-12)
    assert d != pytest.approx(0.5)  # 1 - rho would give 0.5
