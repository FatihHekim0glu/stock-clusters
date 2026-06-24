# ADR-0006: DSR `n_trials` counts the full clustering configuration grid

- **Status:** Accepted
- **Date:** 2026-06-16
- **Deciders:** stock-clusters maintainers
- **Related:** [ADR-0004](0004-honest-1overN-null.md) (the verdict the DSR feeds), [ADR-0005](0005-k-selection-decision-rule.md) (k candidates as one axis)

## Context

The Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014) corrects an observed
Sharpe for **selection bias**: if you try many configurations and report the best,
the maximum Sharpe is inflated purely by chance, and the inflation grows with the
number of trials. The DSR deflates the observed Sharpe toward the Sharpe expected
from the *best of N independent trials* given the return distribution's skew and
kurtosis.

The integrity of this correction hinges entirely on **what counts as a trial**. A
tempting under-count is `n_trials = 1` ("we only ran one clustering") or counting
only the headline allocators. But every knob the pipeline *could* have turned and
reported the best of is a trial, the clustering family, the number of clusters
`k`, the cluster-aware weighting scheme selected, the RMT denoise toggle, and the
OOS cost level. Under-counting `n_trials` is exactly how a lucky clustering
configuration gets laundered into a "significant" edge over 1/N. **This is the top
correctness risk in the project.**

## Decision

The DSR `n_trials` counts the **full explored configuration grid**:

```
n_trials = #clustering-families     (hierarchical and kmeans give 2 when
                                     method="both", else 1; was loosely called
                                     "#linkages")
         × #k-candidates            (every k in [k_min, k_max], per ADR-0005)
         × #weighting-schemes       (the cluster-aware arms selected-best on OOS:
                                     cluster-EW and stripped-HRP → 2. 1/N is the
                                     fixed BENCHMARK, not a selected trial, so it
                                     is not a multiplicity factor.)
         × #denoise-settings-compared
         × #cost-grid-points-compared-on-OOS
```

The code mirrors this exactly: `_N_WEIGHTING_SCHEMES = 2` (the two cluster-aware
arms; 1/N is the baseline they are raced against, not a counted trial), the
clustering-family factor is `2` for `method="both"` and `1` otherwise, and the
single reported denoise setting and single OOS cost level each contribute `1`.

The DSR uses the full `(k + 2) / 4` kurtosis term (not a Gaussian simplification),
implemented in `evaluation/dsr.py`, and is parity-tested to 1e-8.

Two guards enforce the count:

- a **regression test** asserts `n_trials >= product of all swept axes`, so an
  accidental under-count fails CI;
- the DSR is property-tested **non-increasing in `n_trials`**, so adding an axis can
  only deflate, never inflate, the reported Sharpe.

`n_trials` is reported on the API and surfaced in the UI alongside the verdict, so
the multiplicity is visible to the reader, not buried.

## Consequences

- **Positive.** The "winning" Sharpe found by exploring families × `k` × schemes ×
  denoise × costs is correctly deflated. With the full grid counted, the
  cluster-vs-1/N DSR does not clear the `1 - alpha = 0.95` confidence gate, the
  honest finding (ADR-0004), rather than a spuriously significant one.
- **Positive.** Every other ADR that *adds* an axis (a clustering family via
  `method="both"`, a wider `k` range, a denoise comparison) does so knowing it
  raises `n_trials`. The cost of exploration is paid in the right place.
- **Positive.** Because the verdict is a pure function of the JKM `p`-value and the
  DSR (ADR-0004), a DSR that does not clear the `1 - alpha = 0.95` confidence gate
  mechanically blocks an over-claim.
- **Cost.** The verdict is conservative: a genuine small edge could be deflated
  below significance. We accept this, for a benchmark whose purpose is honesty, a
  false "no edge" is far cheaper than a false "clusters beat 1/N".
- **Risk addressed.** "Multiplicity / data-snooping inflating the Sharpe", the
  project's top risk, is countered by counting the full grid, a
  `n_trials >= product` guard, and a DSR-monotonicity property test.
