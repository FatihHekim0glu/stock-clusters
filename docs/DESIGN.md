# Design

This document explains how `stock-clusters` is put together: the layering, the
data flow through a single train/OOS window, the invariants the compute core
guarantees, and the testing strategy that keeps the honest headline honest. For
*why* individual contested choices were made, see the numbered ADRs in
[`docs/decisions/`](decisions/).

## Goals and non-goals

**Goals**

- A pure, typed (`mypy --strict`, `py.typed`), side-effect-free compute core that
  can be audited line by line and vendored into a backend without dragging UI or
  network dependencies along.
- A faithful correlation-clustering pipeline (RMT denoise, Mantegna distance,
  agglomerative + embedding/K-means, gap-statistic `k`-selection) parity-tested to
  1e-10 against `scipy`/`sklearn`.
- A **fair** diversification horse race in which the allocation rule is the only
  treatment, evaluated on an identical post-purge/embargo OOS index, so any Sharpe
  difference cannot be blamed on a mismatched sample.
- A statistically defensible verdict that survives multiplicity correction and is
  *mechanically* prevented from over-claiming.

**Non-goals**

- Beating 1/N. The honest finding is that cluster-aware allocation does not, after
  costs, by a significant margin.
- A live trading system. This is a research/benchmark library.
- A generic clustering toolkit. The clustering exists to map the diversification
  skeleton of an equity universe and to test it honestly against 1/N.

## Layered architecture

The package is strictly layered; each layer imports only from the ones below it.
`src/` has **zero import-time side effects**, guarded by a subprocess import-purity
test.

```
                 cli.py (Typer)        plots.py (Plotly)        metrics.py
                      |                      |                       |
   ┌──────────────────┴──────────────────────┴───────────────────────┘
   │                          evaluation/
   │           dsr.py · comparison.py · verdict.py
   │   (Deflated Sharpe, JKM/Memmel + bootstrap, pure clustering verdict)
   ├──────────────────────────────────────────────────────────────────
   │            allocation/                       backtest/
   │   1/N · cluster-EW · stripped-HRP    walk_forward.py · costs.py · stats.py
   │   (DiversificationResult)            (no-lookahead engine · bps · Sharpe/vol)
   ├──────────────────────────────────────────────────────────────────
   │            clustering/                       stability/
   │   hierarchical · embedding ·         resample · ari · align
   │   kmeans · selection (gap)           (rolling re-fit · adjacent-window ARI)
   ├──────────────────────────────────────────────────────────────────
   │                          correlation/
   │       estimate.py · rmt.py · distance.py (Mantegna · MST · ultrametric)
   ├──────────────────────────────────────────────────────────────────
   │   data.py · data_providers/        foundation (no internal deps)
   │   (yfinance->stooq->synthetic,     _validation · _constants · _typing
   │    risk-free)                       _exceptions · _manifest · _rng
   └──────────────────────────────────────────────────────────────────
```

### Foundation (`_*.py`)

- `_constants.py`: `PERIODS_PER_YEAR = 252` and friends; one source of truth.
- `_validation.py`: input guards (shape, finiteness, sufficient observations).
- `_typing.py` / `_exceptions.py`: shared aliases and the exception taxonomy.
- `_manifest.py` / `_rng.py`: `RunManifest` plus seeded PCG64 substreams. The same
  seed yields byte-identical surrogates, labels, K-means inertia, and bootstrap CI.

### `correlation/`

`estimate.py` turns a price/return panel into a log-return Pearson correlation
(`pct_change(fill_method=None)` then `np.log`, never ffill before differencing).
`rmt.py` re-exports the Marchenko-Pastur eigenvalue clip applied **before**
clustering ([ADR-0001](decisions/0001-rmt-denoise-before-cluster.md)).
`distance.py` builds the Mantegna distance `d_ij = sqrt(2(1 - rho_ij))`, a true
metric ([ADR-0002](decisions/0002-mantegna-metric.md)), plus the MST and the
subdominant ultrametric.

### `clustering/`

`hierarchical.py` wraps SciPy agglomerative linkage (average default; ward/single
ablations) and cuts the tree at `k`, returning a frozen `ClusterResult`.
`embedding.py` embeds assets on the RMT-signal eigenvectors (market mode dropped);
`kmeans.py` runs K-means **on that embedding**, never on the raw distances.
`selection.py` chooses `k` by the gap statistic against a **phase-randomized null**
([ADR-0003](decisions/0003-gap-vs-phase-null.md)) under Tibshirani's **1-SE rule**
([ADR-0005](decisions/0005-k-selection-decision-rule.md)), recording every `k`
candidate as a swept DSR axis.

### `stability/`

`resample.py` re-fits clusters on rolling train windows (no peeking forward).
`ari.py` computes the headline **adjacent-window ARI** (parity to `sklearn`'s
`adjusted_rand_score` at 1e-12). `align.py` does Hungarian / max-Jaccard label
alignment across windows and records cluster births/deaths when `k` changes.

### `allocation/`

Three schemes behind a common interface, all applied with `shift(1)`: `1/N` (the
honest benchmark), cluster-equal-weight, and stripped-HRP (inverse-variance within
cluster, equal across clusters). The frozen `DiversificationResult` carries the
OOS Sharpes, the Memmel-JK `p`-value, the deflated Sharpe, and `n_trials`.

### `evaluation/`

`dsr.py` computes the Deflated/Probabilistic Sharpe with the full-grid `n_trials`
([ADR-0006](decisions/0006-dsr-multiplicity.md)). `comparison.py` holds the
Jobson-Korkie-Memmel test and the Politis-Romano stationary block bootstrap.
`verdict.py` is a **pure function** mapping `(memmel_jk_pvalue, deflated_sharpe,
sharpe_diff)` to a fixed verdict enum
([ADR-0004](decisions/0004-honest-1overN-null.md)).

### Presentation (`metrics.py`, `plots.py`, `cli.py`)

`metrics.py` holds silhouette (parity to `sklearn` 1e-10), cophenetic correlation
(parity to `scipy.cophenet` 1e-10), Newman modularity, and the **post-hoc**
ARI-vs-GICS (GICS never enters the distance or `k`-selection). `plots.py` builds
JSON-able Plotly `{data, layout}` figures with a lazy Plotly import. `cli.py` is a
thin Typer orchestration layer (lazy Typer import) with a `run` command (cluster +
optional `--run-diversification` horse race) and an offline `demo`.

## Data flow through one train / OOS window

```
train window  ──►  log-return correlation  ──►  [RMT clip, ADR-0001]  ──► renormalize
   (prices)            │                                                       │
                       │                                          Mantegna distance (ADR-0002)
                       │                                                       │
                       │          ┌────────────────────────────────┬──────────┘
                       │          ▼                                 ▼
                       │   gap vs phase-null (ADR-0003)       agglomerative linkage / MST
                       │   + 1-SE rule (ADR-0005)  ──► k             │   RMT-signal embedding ─► K-means
                       │                                  └──────────┴──► labels FROZEN
                       ▼
        all clustering + k-selection fit on TRAIN window ONLY
                       │
                       ▼  weights applied with shift(1)
   OOS window  ──►  1/N · cluster-EW · stripped-HRP on the IDENTICAL OOS index
                       │            (post purge + embargo)
                       ▼  (aggregate across windows)
        JKM Sharpe-diff test · block-bootstrap CI · Deflated Sharpe (full n_trials)
                       │
                       ▼
            verdict.py  ──►  clustering verdict (pure-derived enum)
```

The headline comparison is **best cluster-aware vs 1/N** on a single, shared,
post-purge/embargo OOS date index, a property test asserts that index equality
*before* the JKM test runs.

## Key invariants

The compute core guarantees, and tests enforce:

1. **Metric axioms.** The Mantegna distance is non-negative, symmetric, zero on the
   diagonal, and obeys the triangle inequality (Hypothesis property test).
2. **No-lookahead.** In the diversification backtest the correlation, RMT cutoff,
   distance, labels, and (per-window) `k` are deterministic functions of each
   walk-forward TRAIN window; clusters are RE-FIT inside every train window, never
   reused from a full-panel fit. An end-to-end property test future-perturbs
   post-cutoff input rows and asserts the OOS-evaluated train-only labels and the
   pre-cutoff OOS return series are unchanged. (The full-panel display map is
   descriptive only, figures + ARI-vs-GICS, and is never applied out-of-sample.)
3. **Re-fit-then-applied.** Clusters fit on a train window are applied to the next
   OOS window with purge + embargo + `shift(1)`; the next window re-fits afresh.
4. **Identical OOS index.** All three allocators are scored on the same OOS dates;
   a property test asserts index equality before Memmel-JK runs.
5. **Monotonic cut.** `k = N` yields singletons, `k = 1` one cluster, and a higher
   cut never increases the cluster count.
6. **Scale & permutation invariance.** Rescaling returns or relabeling assets leaves
   the cophenetic distances and ARI invariant.
7. **Determinism.** Same seed -> byte-identical phase-null surrogate, K-means
   inertia, bootstrap CI, and selected `k`.
8. **DSR multiplicity.** `n_trials >= product of all swept axes` (clustering
   families × `k` × cluster-aware schemes (2; 1/N is the fixed benchmark) × denoise
   × cost grid); the DSR is non-increasing in `n_trials`.
9. **Verdict safety.** The verdict cannot emit `clusters_beat_1n` while the JKM test
   is insignificant or the deflated Sharpe is non-positive (truth-table unit-tested).
10. **Import purity.** Importing any `src/stockclusters` module triggers no I/O, no
    network, no Plotly/Typer import (subprocess-tested).

## Testing strategy

Tests are partitioned by intent under `tests/` (markers in `pyproject.toml`):

- **`unit/`**: isolated kernels: silhouette/cophenetic/modularity/ARI-vs-GICS, the
  verdict truth table, the plot `{data, layout}` shape, the CLI smoke run.
- **`property/`** (Hypothesis), the invariants above: metric axioms,
  no-lookahead future-perturbation invariance, permutation/scale invariance,
  monotonic cut, seed determinism, identical OOS index.
- **`parity/`**: golden checks against independent references: linkage/cophenetic
  vs `scipy` (1e-10), silhouette (1e-10) and ARI (1e-12) vs `sklearn`, K-means
  **inertia** under fixed init (1e-8), RMT edge vs analytic MP `(1 ± sqrt(q))^2`
  (1e-10), DSR/PSR (1e-8), gap machinery on a fixed surrogate set.
- **`regression/`**: the honest null, locked: on noise the headline never
  over-claims (a `clusters beat 1/N` verdict is structurally impossible) and the
  Sharpe-gap test stays calibrated near its 5% level across many null draws; the
  `k_blocks` recovery guard (ARI vs truth >= threshold); the **RMT denoise-on/off
  ablation** table reported honestly; the DSR trial-count guard; `to_dict()`
  snapshots; valid-Plotly-JSON / explicit-`null` API-shape checks.
- **`integration/`**: end-to-end pipeline runs (may touch data/network).

Seeded fixtures in `conftest.py` (`one_block_correlation`, `k_blocks`,
`k_blocks_truth`, `pure_noise`) give every layer deterministic, adversarial inputs.

## Backend & frontend boundary

The compute core is decoupled from delivery. The backend vendors
`stock-clusters[data]` (not `[all]`, no Streamlit/kaleido) under
`api/lib/stockclusters/` and exposes `POST /tools/stock-clusters/run`, returning
summary scalars plus Plotly `{data, layout}` figures (absent figures are an
explicit `null`). Plotly/Typer are lazy, the sync path caps `n_assets`, `gap_B`,
and the `k` span to stay inside the 512 MB scale-to-zero container, and a
diversification request over a survivor-only universe returns **422**. The frontend
renders the figures, surfaces **ARI-vs-GICS** prominently ("clusters largely
re-discover sectors"), shows a survivorship banner, and reads the pure-derived
verdict, including a "not statistically significant after costs" badge when the
JKM `p > 0.05` or `DSR <= 0`.
