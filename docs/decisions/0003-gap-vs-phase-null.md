# ADR-0003: Select `k` by the gap statistic vs a phase-randomized null

- **Status:** Accepted
- **Date:** 2026-06-16
- **Deciders:** stock-clusters maintainers
- **Related:** [ADR-0005](0005-k-selection-decision-rule.md) (the 1-SE decision rule on top of this gap)

## Context

The number of clusters `k` is the most consequential free parameter: choose it by
eyeballing a dendrogram and the whole "diagnostic" becomes a Rorschach test. The
Tibshirani (2001) **gap statistic** makes the choice principled: it compares the
observed pooled within-cluster dispersion `log W_k` against its expectation under a
**null reference** with no cluster structure, `gap(k) = E*[log W_k] - log W_k`, and
prefers the `k` at which the data pull furthest below the null.

Everything hinges on *which null*. The textbook default is a uniform box drawn over
the data's bounding range (or its principal-component box). For **financial
returns** that null is wrong: it destroys not only cross-asset correlation (which we
want gone) but also each asset's own autocorrelation, fat tails, and volatility
clustering, so the reference is "too easy", the gap is inflated, and the procedure
over-splits.

The honest null must preserve each asset's marginal dynamics while destroying only
the *cross-asset* structure that clustering is supposed to detect.

## Decision

`k` is selected by the gap statistic against a **phase-randomized (FFT
surrogate) null** (`clustering/selection.py:phase_randomize`):

- each asset's return series is independently Fourier-transformed, its phases are
  replaced by uniform random phases (conjugate-symmetric, so the surrogate is
  real), and inverse-transformed;
- this **preserves each asset's amplitude spectrum**, hence its autocorrelation,
  marginal variance, and (approximately) its spectral fingerprint, while
  **destroying cross-asset correlation**.

The reference `E*[log W_k]` and its standard error `s_k` are averaged over `B`
phase-randomized draws (`gap_B`, default 20; capped at 20 on the hosted sync path).
Every evaluated `k` candidate is a swept DSR axis and is recorded on `GapResult`.

A **uniform-box null is retained only as a code-path validator**: the gap machinery
(`W_k`, `log`-gap, `s_k`) is parity-checked against a reference implementation on a
fixed surrogate set, and the gap-on-uniform-null is checked to 1e-6 against a
reference, but the uniform null is **never** used to select `k`.

## Consequences

- **Positive.** The null has realistic single-asset dynamics, so the gap is not
  inflated by autocorrelation/fat-tails and `k` is not systematically over-split.
- **Positive.** Reproducible: phases are drawn from a seeded PCG64 generator, so a
  fixed seed reproduces the surrogate (and the selected `k`) byte-for-byte, and
  this is property-tested.
- **Positive.** The gap code path is still validated against the classical uniform
  null, so a bug in `W_k`/`s_k` is caught without contaminating the selector.
- **Cost.** `B` FFT surrogates × `k` candidates clusterings per fit; bounded by the
  `gap_B <= 20` and `k_max - k_min <= 12` caps on the hosted path.
- **Risk addressed.** "Wrong null inflating the gap and over-splitting `k`" is
  countered by phase-randomization; "silent gap-machinery bug" by the
  uniform-null parity check.
