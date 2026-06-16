# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-06-16

### Added

- Initial package skeleton (src-layout, import name `stockclusters`).
- Reused, import-pure core ported from the HRP tool: `_constants`, `_typing`,
  `_exceptions`, `_validation`, `_manifest` (`RunManifest` with BLAKE2b
  config-hash), `_rng` (seeded PCG64 generator + substream spawning); the RMT
  (`marchenko_pastur_clip`), covariance estimators, walk-forward purge/embargo
  backtest, DSR/PSR, and Jobson-Korkie-Memmel machinery.
- New clustering subpackages with full typed contracts (stubs):
  - `correlation` — log-return correlation, RMT denoise re-export, Mantegna
    distance `sqrt(2(1 - rho))`, MST + subdominant ultrametric.
  - `clustering` — hierarchical (average/ward/single), RMT-signal embedding,
    K-means on the embedding, gap-statistic `k`-selection vs a phase-randomized
    null (Tibshirani 1-SE rule); frozen `ClusterResult` / `GapResult`.
  - `stability` — rolling re-fit, adjacent-window ARI, Hungarian/max-Jaccard label
    alignment; frozen `StabilityResult`.
  - `allocation` — 1/N, cluster-equal-weight, stripped-HRP; frozen
    `DiversificationResult`.
  - `evaluation` — clustering `ClusteringVerdict` (pure function over Memmel-JK p +
    DSR).
  - `metrics`, `plots` (lazy Plotly), `cli` (lazy Typer).
- Seeded synthetic test fixtures (`one_block_correlation`, `k_blocks`,
  `pure_noise`) and partitioned `tests/` (unit/parity/property/regression/
  integration).

[Unreleased]: https://github.com/FatihHekim0glu/stock-clusters/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/FatihHekim0glu/stock-clusters/releases/tag/v0.1.0
