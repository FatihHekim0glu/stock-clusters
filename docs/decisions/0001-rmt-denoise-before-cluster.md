# ADR-0001: RMT-denoise the correlation BEFORE clustering

- **Status:** Accepted
- **Date:** 2026-06-16
- **Deciders:** stock-clusters maintainers
- **Related:** [ADR-0002](0002-mantegna-metric.md) (the distance built on the denoised correlation)

## Context

An empirical correlation matrix estimated from `T` daily returns over `N` assets
is dominated by estimation noise: with the aspect ratio `q = N / T` away from
zero, most of its eigenvalues fall inside the Marchenko-Pastur (MP) bulk and are
statistically indistinguishable from those of a pure-noise matrix. Clustering the
*raw* correlation therefore partly clusters sampling noise, the dendrogram leaf
order, the cut into `k` clusters, and the temporal stability of the assignment all
inherit that noise.

Random-matrix theory gives a principled cleaning step. Eigenvalues at or below the
MP upper edge `lambda_+ = (1 + sqrt(q))^2` carry no signal; flattening them to
their common average (trace-preserving) while keeping the signal eigenvalues
yields a denoised correlation whose structure is what the clustering should see.

The open question is *ordering*: denoise the correlation **before** building the
distance and clustering, or cluster the raw correlation and treat denoising as an
optional post-hoc smoother. Doing it after the fact does not remove the noise the
linkage already consumed.

## Decision

RMT denoising runs **before** the Mantegna distance and the clustering, gated by
the `denoise` flag (default **on**). The pipeline is:

```
log-returns -> correlation -> [MP eigenvalue clip] -> renormalize to unit diagonal
            -> Mantegna distance -> linkage / embedding -> cut at k
```

The clip is the reused, parity-tested `marchenko_pastur_clip` (edge vs analytic
`(1 ± sqrt(q))^2` to 1e-10); the cleaned matrix is renormalized back to a unit
diagonal so it is a proper correlation before the distance is taken.

Crucially, the choice is **not asserted to help**, it is measured. The
denoise-on/off **ablation** is a runnable regression test
(`tests/regression/test_rmt_ablation.py`) that tabulates ARI-vs-GICS and mean
adjacent-window stability ARI for both settings, and the result is reported
honestly in the README validation table even when the effect is marginal.

## Consequences

- **Positive.** The clustering operates on the signal subspace, so the cut into
  `k` clusters and the temporal stability are less driven by sampling noise; the
  effect is largest at small `T` / large `N` (high `q`).
- **Positive.** Because denoising is a toggle with a published ablation, the claim
  "denoising helps" is evidence-backed, not assumed. On clean, well-separated
  synthetic blocks the gap is small (the structure is trivially recoverable either
  way), and the test asserts exactly that honest, modest delta.
- **Cost.** One eigendecomposition per fit, negligible at the universe sizes here,
  plus a renormalization pass to restore the unit diagonal.
- **Risk addressed.** "Clustering sampling noise" is countered by an MP clip whose
  edge is parity-tested and whose benefit is ablated, not hand-waved.
