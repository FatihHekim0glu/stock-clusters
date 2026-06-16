# stock-clusters

Cluster an equity universe (e.g. the S&P 500) by its **correlation structure** —
RMT-denoised, under the **Mantegna distance** `d = sqrt(2(1 - rho))` — to map its
diversification skeleton, and **honestly test** whether cluster-aware allocation
beats naive 1/N out-of-sample after costs.

**Live tool:** https://fatihhekimoglu.com/tools/stock-clusters _(hosted demo)_

> **Honest headline (the correct result, not a bug):** correlation clusters mostly
> re-discover GICS sectors (ARI ~0.4–0.7) and cluster-aware allocation does **not**
> beat 1/N out-of-sample after costs (Jobson-Korkie-Memmel insignificant, deflated
> Sharpe ~0). Diagnostic value only — no free alpha.

This is a pure, typed, src-layout compute library (`import stockclusters`) with zero
import-time side effects, so the same functions back a local Typer CLI and a hosted
FastAPI tool unchanged.

## Install

```bash
uv venv
uv pip install -e ".[data,viz,dev]"
uv run python -c "import stockclusters; print(stockclusters.__version__)"
```

Optional extras (install only what you need — the hosted API ships `[data]` only to
keep the scale-to-zero container lean):

| Extra    | Pulls in                                              | For |
| -------- | ----------------------------------------------------- | --- |
| `data`   | yfinance, curl_cffi, pandas-datareader, httpx, pyarrow, diskcache | live price loading (yfinance→Stooq→Polygon→synthetic) |
| `viz`    | plotly, kaleido                                        | the `{data, layout}` figure builders |
| `app`    | streamlit, typer                                       | the local interactive demo + CLI |
| `dev`    | pytest(+cov), hypothesis, syrupy, ruff, mypy, pandas-stubs | the test / lint / type stack |

## Quickstart

The whole pipeline is one call. `run_cluster_analysis` fits the (RMT-denoised)
correlation, builds the Mantegna distance, selects `k`, clusters, and — when asked —
runs the rolling-stability and 1/N-vs-cluster horse race; `assemble_figures` builds
the Plotly figures. This is exactly what the hosted FastAPI router wraps.

```python
import stockclusters as sc

prices, source = sc.get_prices(["AAPL", "MSFT", "XOM", "CVX", "JPM", "BAC"],
                               start=__import__("datetime").date(2018, 1, 1),
                               end=__import__("datetime").date(2023, 12, 31))
returns = sc.compute_returns(prices)

params = sc.ClusterAnalysisParams(method="hierarchical", linkage="average",
                                  n_clusters=None,           # None ⇒ gap-statistic
                                  denoise=True,
                                  run_diversification=True,  # the honest horse race
                                  run_stability=True)
analysis = sc.run_cluster_analysis(returns, params, data_source=source)

print(analysis.n_clusters, analysis.verdict)   # e.g. 5  no_significant_difference
figures = sc.assemble_figures(analysis)         # heatmap/dendrogram/MST/embedding/…
```

`run_cluster_analysis(returns, params) -> ClusterAnalysis` and
`assemble_figures(analysis) -> {figure_name: {data, layout} | None}` are the two
public pipeline entrypoints the backend calls.

## What's inside

| Subpackage              | Responsibility                                                                 |
| ----------------------- | ------------------------------------------------------------------------------ |
| `correlation`           | log-return correlation, RMT denoise, Mantegna distance, MST, ultrametric        |
| `clustering`            | hierarchical / embedding / K-means; gap-statistic `k`-selection (phase null)    |
| `stability`             | rolling re-fit, adjacent-window ARI, Hungarian/max-Jaccard label alignment      |
| `allocation`            | 1/N, cluster-equal-weight, stripped-HRP + the diversification horse race        |
| `evaluation`            | DSR/PSR, Jobson-Korkie-Memmel, the pure-function clustering verdict             |
| `metrics` / `plots`     | silhouette / cophenetic / modularity / ARI-vs-GICS; lazy-Plotly figures         |

## Honest methodology guards

- Correlation, RMT, distance, labels, and `k`-selection fit on the **train window
  only**; clusters are **frozen** then applied to the next OOS window.
- The Mantegna distance is a **true metric** (`sqrt(2(1 - rho))`, not `1 - rho`); the
  metric axioms are property-tested.
- `k` is chosen via the gap statistic vs a **phase-randomized null** (Tibshirani
  1-SE rule); silhouette and MST modularity are reported cross-checks.
- The deflated-Sharpe `n_trials` counts the **full swept grid** (clustering
  families × `k` candidates × cluster-aware weighting schemes (2) × denoise
  settings × cost grid).
- GICS sectors are used **post-hoc only** — never in the distance or `k`-selection.

## Validation table

Every numerical claim is pinned against an independent reference (a paper or a
reference library) at a stated tolerance, in a named test file. Parity oracles use
the strict tolerances below; the property/regression rows pin invariants and the
honest null.

| Claim (paper / reference)                                                   | Tolerance | Test file |
| --------------------------------------------------------------------------- | --------- | --------- |
| Mantegna distance `d = sqrt(2(1 - rho))` is a true metric (Mantegna 1999)   | exact + Hypothesis | `tests/property/test_metric_axioms.py` |
| Agglomerative linkage vs `scipy.cluster.hierarchy.linkage`                  | `1e-10`   | `tests/parity/test_clustering_oracles.py` |
| Cophenetic correlation vs `scipy.cluster.hierarchy.cophenet`                | `1e-10`   | `tests/unit/test_metrics.py`, `tests/parity/test_clustering_oracles.py` |
| Silhouette vs `sklearn.metrics.silhouette_score` (precomputed)              | `1e-10`   | `tests/unit/test_metrics.py` |
| Adjusted Rand Index vs `sklearn.metrics.adjusted_rand_score`                | `1e-12`   | `tests/parity/test_dsr_ari_oracle.py`, `tests/unit/test_metrics.py` |
| K-means **inertia** vs `sklearn.cluster.KMeans` (fixed explicit init)       | `1e-8`    | `tests/parity/test_clustering_oracles.py` |
| RMT edge vs analytic Marchenko-Pastur `(1 ± sqrt(q))^2`                     | `1e-10`   | `tests/parity/test_clustering_oracles.py` |
| Deflated / Probabilistic Sharpe (Bailey & Lopez de Prado 2014)             | `1e-8`    | `tests/parity/test_dsr_ari_oracle.py` |
| Jobson-Korkie-Memmel Sharpe-difference test (Memmel 2003)                  | `1e-8`    | `tests/parity/test_dsr_ari_oracle.py` |
| Gap statistic `W_k` / log-gap / `s_k` on a fixed surrogate (Tibshirani 2001)| `1e-10`   | `tests/parity/test_clustering_oracles.py` |
| Gap-on-uniform-null vs reference (code-path validator, not the selector)    | `1e-6`    | `tests/parity/test_clustering_oracles.py` |
| No-lookahead: post-cutoff perturbation leaves train labels + shifted weights | exact   | `tests/property/test_clustering_properties.py` |
| Monotonic cut (`k=N` singletons, `k=1` one cluster, higher cut ≤ count)     | exact     | `tests/property/test_clustering_properties.py` |
| Seed determinism (phase null, K-means inertia, bootstrap CI)               | exact     | `tests/property/test_clustering_properties.py` |
| Identical OOS date index across 1/N, cluster-EW, stripped-HRP              | exact     | `tests/property/test_stability_allocation.py` |
| Recovery guard: `k_blocks` ARI vs truth ≥ pinned threshold                 | ≥ 0.8     | `tests/unit/test_metrics.py` |
| Honest null: `pure_noise` → Memmel-JK insignificant, `DSR ≤ 0`             | locked    | `tests/regression/test_honest_headline.py`, `tests/regression/test_verdict_honest_null.py` |
| RMT denoise-on/off ablation (ARI-vs-GICS + stability), reported honestly    | locked    | `tests/regression/test_rmt_ablation.py` |
| DSR `n_trials ≥ product of all swept axes` (multiplicity guard)            | locked    | `tests/regression/test_verdict_honest_null.py` |
| Plotly `{data, layout}` JSON: finite scalars, explicit `null` for absent figs | locked  | `tests/unit/test_plots.py` |
| CLI smoke (`--help` + a tiny seeded run, no network)                        | locked    | `tests/unit/test_cli_smoke.py` |

## Honest headline

The expected, literature-consistent result of running this tool on a real S&P 500
window is **not** a trading edge — and the code is built so it cannot pretend
otherwise:

- **Clusters re-discover GICS sectors.** The correlation clusters line up with the
  GICS sector taxonomy at an Adjusted Rand Index of roughly **0.4–0.7** (post-hoc;
  GICS never enters the distance or `k`-selection). The map is real and useful as a
  *diagnostic* of the universe's diversification skeleton.
- **Cluster-aware allocation does NOT beat 1/N out-of-sample after costs.** Across
  the full swept grid (clustering families × `k` × cluster-aware schemes × denoise ×
  cost levels), and with clusters RE-FIT inside each walk-forward train window (no
  look-ahead), the best cluster-aware Sharpe minus the 1/N Sharpe is **not
  statistically
  significant** (Jobson-Korkie-Memmel `p` not significant) and the **deflated Sharpe
  is ~0** once the trial count is honestly accounted for. The headline verdict
  resolves to `no_significant_difference`.

This is the correct finding, not a bug. The verdict is a **pure function** of the
inference outputs (`derive_clustering_verdict`), so it is structurally incapable of
printing "clusters beat 1/N" while the test is insignificant or the deflated Sharpe
is non-positive — the truth table is unit-tested and the `pure_noise` null is a
locked regression. The value here is diagnostic clarity, not free alpha.

## Design decisions

See `docs/decisions/` for the ADRs:

- [ADR-0001](docs/decisions/0001-rmt-denoise-before-cluster.md) — RMT-denoise before clustering
- [ADR-0002](docs/decisions/0002-mantegna-metric.md) — the Mantegna `sqrt(2(1 - rho))` metric
- [ADR-0003](docs/decisions/0003-gap-vs-phase-null.md) — gap statistic vs a phase-randomized null
- [ADR-0004](docs/decisions/0004-honest-1overN-null.md) — the honest 1/N null and pure-function verdict
- [ADR-0005](docs/decisions/0005-k-selection-decision-rule.md) — Tibshirani 1-SE `k`-selection rule
- [ADR-0006](docs/decisions/0006-dsr-multiplicity.md) — DSR `n_trials` counts the full grid

## License

MIT — see [LICENSE](LICENSE).
