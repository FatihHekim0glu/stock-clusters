# ADR-0004: The headline is an honest 1/N null with a structurally-enforced verdict

- **Status:** Accepted
- **Date:** 2026-06-16
- **Deciders:** stock-clusters maintainers
- **Related:** [ADR-0006](0006-dsr-multiplicity.md) (the full trial count that deflates the Sharpe)

## Context

The tempting (and dishonest) story for a clustering tool is "cluster-aware
allocation beats naive 1/N". The literature is clear that this is usually **not**
true out-of-sample after costs: naive `1/N` is a famously stubborn benchmark
(DeMiguel, Garlappi & Uppal, 2009), and cluster-aware schemes (equal-weight across
clusters, stripped-HRP) rarely clear it by a statistically significant, cost- and
multiplicity-adjusted margin.

A research tool earns trust by being **structurally incapable of over-claiming**.
It is not enough to "try to report honestly", a narration step can always drift.
The verdict must be a deterministic function of the inference outputs, so a
non-significant test or a deflated Sharpe that does not clear the `1 - alpha = 0.95`
confidence gate *mechanically* blocks a "beats 1/N" headline.

## Decision

The diversification horse race is run against the **1/N null** under three
guarantees:

1. **Fair comparison.** `1/N`, cluster-equal-weight, and stripped-HRP are evaluated
   on the **identical** post-purge/embargo OOS date index; a property test asserts
   index equality *before* the Jobson-Korkie-Memmel (JKM) test runs.
2. **Two complementary statistics.** The Sharpe gap (best cluster-aware minus 1/N)
   is tested with the Memmel-corrected JKM test *and* deflated with the Deflated
   Sharpe Ratio (DSR) over the full trial count (ADR-0006).
3. **Pure-function verdict.** `derive_clustering_verdict(memmel_jk_pvalue,
   deflated_sharpe, sharpe_diff)` returns one of `clusters_beat_1n`,
   `clusters_lose_to_1n`, or `no_significant_difference` from a fixed truth table.
   It returns `clusters_beat_1n` **only if** the JKM test is significant
   (`p < alpha`) **and** the deflated Sharpe clears the `1 - alpha = 0.95`
   confidence threshold (the DSR is a probability/CDF in `[0, 1]`; the gate fails if
   `DSR <= 0.95`). The truth table is unit-tested, and a regression test asserts
   that the **pure-noise** fixture yields a non-significant JKM `p` and a DSR that
   does not clear the `0.95` gate (hence `no_significant_difference`).

The README states the expected, literature-consistent outcome up front: clusters
re-discover GICS sectors (ARI ~0.4-0.7) and cluster-aware allocation does **not**
beat 1/N after costs.

## Consequences

- **Positive.** The headline cannot drift away from the evidence: a lucky point
  estimate with an insignificant test or a DSR below the `0.95` confidence gate can
  never print "beats 1/N".
- **Positive.** The honest null is a *feature*, the tool's value is diagnostic (a
  map of the diversification skeleton), not a false alpha claim.
- **Positive.** Because the comparison is on an identical OOS index, a Sharpe gap
  cannot be an artifact of mismatched samples.
- **Cost.** The verdict is conservative: a genuine small edge could be deflated
  below significance. For an honesty-first benchmark a false "no edge" is far
  cheaper than a false "clusters beat 1/N".
- **Risk addressed.** "Narrating an over-claim" is closed by making the verdict a
  pure function of the test `p`-value and the deflated Sharpe.
