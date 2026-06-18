# ADR-0002: Cluster on the Mantegna distance `sqrt(2(1 - rho))`, a true metric

- **Status:** Accepted
- **Date:** 2026-06-16
- **Deciders:** stock-clusters maintainers
- **Related:** [ADR-0001](0001-rmt-denoise-before-cluster.md) (the correlation this distance is built on)

## Context

Clustering, the minimum spanning tree (MST), and the subdominant ultrametric all
behave sensibly only on a **proper metric**, non-negative, zero iff the points
coincide, symmetric, and obeying the triangle inequality. The distance is derived
from the (RMT-denoised) correlation `rho_ij`.

Two widespread shortcuts are **not** metrics and are rejected:

- `d_ij = 1 - rho_ij` violates the triangle inequality and conflates
  anti-correlation with distance.
- `d_ij = 1 - |rho_ij|` collapses strongly anti-correlated pairs (`rho = -1`) onto
  zero distance, which is wrong for a diversification map: an asset and its hedge
  are maximally *useful* to hold together, not identical.

Mantegna (1999) defines

```
d_ij = sqrt( 2 * (1 - rho_ij) )
```

which maps `rho = +1 -> d = 0`, `rho = 0 -> d = sqrt(2)`, `rho = -1 -> d = 2`, is
the chord distance between unit vectors on the correlation sphere, and **is** a
true Euclidean-embeddable metric. It is the canonical distance for correlation-based
financial networks (the asset MST and its single-linkage subdominant ultrametric).

## Decision

The distance is **`d_ij = sqrt(2(1 - rho_ij))`**, implemented in
`correlation/distance.py`, and used directly by the agglomerative linkage, the MST,
and the subdominant ultrametric. The `1 - rho` and `1 - |rho|` forms are
**rejected** and never used anywhere in the codebase.

The line carries an inline `HONESTY-REQUIREMENT` comment naming the two footguns.
The metric axioms, non-negativity, zero diagonal, symmetry, and the triangle
inequality, are asserted with a **Hypothesis property test**, and the exact values
at `rho in {-1, 0, +1}` are pinned by a unit test. K-means does **not** run on this
distance (K-means assumes Euclidean coordinates); it runs on the RMT-signal
*embedding* instead (see ADR-0005), with the distance reserved for the
hierarchical/MST path.

## Consequences

- **Positive.** Linkage, MST, and ultrametric operate on a genuine metric, so the
  dendrogram leaf order and the network backbone are well-defined and match the
  reference oracles (scipy linkage/cophenet to 1e-10).
- **Positive.** Anti-correlated pairs are placed far apart (`d = 2`), which is the
  correct geometry for a *diversification* skeleton, the opposite of what
  `1 - |rho|` would do.
- **Positive.** The single most common silent correlation-clustering bug
  (`1 - rho`) is caught by a property test and a visible comment rather than
  slipping into the figures.
- **Cost.** None of note: the closed form is one elementwise operation.
- **Risk addressed.** "Non-metric distance distorting the dendrogram / collapsing
  hedges" is closed by the property suite and an inline honesty comment.
