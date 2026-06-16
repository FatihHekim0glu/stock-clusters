"""Property tests for the Mantegna metric axioms (placeholder slot).

The full Hypothesis property suite (metric axioms, no-lookahead invariance,
permutation equivariance, scale invariance, monotonicity, seed determinism,
identical OOS index) is authored here. Until the distance kernel is implemented the
single axiom check below is marked xfail(strict=False) so the partition collects.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.property
@pytest.mark.xfail(reason="mantegna_distance is a stub", strict=False)
def test_mantegna_zero_distance_on_identity() -> None:
    """A unit-diagonal identity correlation maps to a zero-diagonal distance."""
    from stockclusters.correlation.distance import mantegna_distance

    corr = np.eye(4)
    dist = mantegna_distance(corr)
    assert np.allclose(np.diag(dist.to_numpy()), 0.0)
