# ADR-0005: The pre-registered `k`-selection rule is Tibshirani's 1-SE rule

- **Status:** Accepted
- **Date:** 2026-06-16
- **Deciders:** stock-clusters maintainers
- **Related:** [ADR-0003](0003-gap-vs-phase-null.md) (the gap statistic this rule decides on), [ADR-0006](0006-dsr-multiplicity.md) (k candidates as a swept axis)

## Context

The gap statistic (ADR-0003) gives a `gap(k)` curve and a standard error `s_k` at
each candidate `k`. A curve still needs a **rule** to turn it into a single chosen
`k`. The naive `argmax_k gap(k)` is noisy — it chases the sampling fluctuation in
the reference draws and tends to pick a larger `k` than warranted.

There are several reported cross-checks (silhouette, MST/Newman modularity) that
also peak at "a good `k`", and it is tempting to pick whichever metric gives the
most flattering answer. That is exactly the multiple-comparisons trap: choosing the
selector *after* seeing the data manufactures structure and inflates every
downstream claim.

The rule must therefore be **pre-registered** (fixed before looking at the data) and
**parsimonious** (prefer fewer clusters unless the evidence clearly favours more).

## Decision

The **pre-registered selector** is Tibshirani's (2001) **1-standard-error rule**:
choose the *smallest* `k` whose gap is within one standard error of the next
candidate's gap,

```
k* = min { k : gap(k) >= gap(k + 1) - s_{k+1} }
```

implemented in `clustering/selection.py` and recorded on `GapResult` as
`selection_rule = "tibshirani_1se"`.

- **Silhouette and MST/Newman modularity are reported cross-checks only.** They are
  surfaced in the summary so the reader can see whether they agree, but they
  **never override** the gap+1-SE selection.
- **Determinism.** The selected `k` is a deterministic function of the seeded gap
  computation (property-tested for seed stability).
- **Multiplicity.** Every evaluated `k` in `[k_min, k_max]` is a swept axis; the
  count of candidates is recorded and folded into the DSR `n_trials` product
  (ADR-0006), so exploring more `k` values costs significance in the right place.
- **Fixed-`k` override.** A user-supplied `n_clusters` bypasses selection entirely
  (`selection_method = "fixed"`); the gap is then not consulted.

## Consequences

- **Positive.** The 1-SE rule biases toward **fewer, more robust** clusters, which
  is the conservative choice for a diagnostic map and is reproducible across seeds.
- **Positive.** Because the selector is pre-registered and the cross-checks cannot
  override it, the `k` choice is not cherry-picked from whichever metric flatters.
- **Positive.** Counting every `k` candidate as a trial means a wide `[k_min,
  k_max]` sweep is paid for in the DSR, not hidden.
- **Cost.** The 1-SE rule can under-split genuinely fine structure; we accept this
  for robustness and report silhouette/modularity so an over-merge is visible.
- **Risk addressed.** "Selector cherry-picking / argmax noise inflating `k`" is
  closed by a pre-registered, parsimonious, seed-deterministic rule with
  non-overriding cross-checks.
