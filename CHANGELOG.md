# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-06-16

### Added

- Initial release of `stock-clusters`: an import-pure, typed, src-layout compute
  library (`import stockclusters`) with zero import-time side effects.
- Reused, import-pure core ported from the HRP tool: `_constants`, `_typing`,
  `_exceptions`, `_validation`, `_manifest` (`RunManifest` with BLAKE2b
  config-hash), `_rng` (seeded PCG64 generator + substream spawning); the RMT
  (`marchenko_pastur_clip`), covariance estimators, walk-forward purge/embargo
  backtest, DSR/PSR, and Jobson-Korkie-Memmel machinery.
- Clustering subpackages:
  - `correlation`: log-return correlation, RMT denoise, Mantegna distance
    `sqrt(2(1 - rho))` (a true metric), MST + subdominant ultrametric.
  - `clustering`: hierarchical (average/ward/single), RMT-signal embedding,
    K-means on the embedding, gap-statistic `k`-selection vs a phase-randomized
    null (Tibshirani 1-SE rule); frozen `ClusterResult` / `GapResult`.
  - `stability`: rolling re-fit, adjacent-window ARI, Hungarian/max-Jaccard label
    alignment; frozen `StabilityResult`.
  - `allocation`: 1/N, cluster-equal-weight, stripped-HRP, plus the no-lookahead
    diversification horse race; frozen `DiversificationResult`.
  - `evaluation`: clustering `ClusteringVerdict` (pure function over Memmel-JK p +
    DSR), so the headline cannot claim "beats 1/N" while the test is insignificant.
  - `metrics`, `plots` (lazy Plotly), `cli` (lazy Typer), `data` /
    `data_providers` (yfinance→Stooq→Polygon→synthetic loader chain).
- **End-to-end pipeline entrypoint** (`pipeline.py`): `run_cluster_analysis(returns,
  params) -> ClusterAnalysis` wires correlation → clustering → (optional) stability
  and diversification into one call with the FULL DSR trial-count, and
  `assemble_figures(analysis)` builds the Plotly figures. The hosted FastAPI router
  is a thin wrapper over these two functions.
- Seeded synthetic test fixtures (`one_block_correlation`, `k_blocks`,
  `pure_noise`) and partitioned `tests/` (unit/parity/property/regression/
  integration), including an end-to-end pipeline integration test and the locked
  honest-null headline regression.

[Unreleased]: https://github.com/FatihHekim0glu/stock-clusters/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/FatihHekim0glu/stock-clusters/releases/tag/v0.1.0
