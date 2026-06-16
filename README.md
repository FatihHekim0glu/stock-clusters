# stock-clusters

Cluster an equity universe (e.g. the S&P 500) by its **correlation structure** —
RMT-denoised, under the **Mantegna distance** `d = sqrt(2(1 - rho))` — to map its
diversification skeleton, and **honestly test** whether cluster-aware allocation
beats naive 1/N out-of-sample after costs.

> **Honest headline (the correct result, not a bug):** correlation clusters mostly
> re-discover GICS sectors (ARI ~0.4–0.7) and cluster-aware allocation does **not**
> beat 1/N out-of-sample after costs (Jobson-Korkie-Memmel insignificant, deflated
> Sharpe ~0). Diagnostic value only — no free alpha.

This is a pure, typed, src-layout compute library (`import stockclusters`) with zero
import-time side effects, so the same functions back a local CLI/Streamlit demo and
a hosted FastAPI tool unchanged.

## Install

```bash
uv venv
uv pip install -e ".[data,viz,dev]"
uv run python -c "import stockclusters; print(stockclusters.__version__)"
```

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
- The deflated-Sharpe `n_trials` counts the **full swept grid** (linkages × `k`
  candidates × weighting schemes × denoise settings × cost grid).
- GICS sectors are used **post-hoc only** — never in the distance or `k`-selection.

## Validation table

_To be completed by the implementation: paper → tolerance → test file (scipy
linkage/cophenetic 1e-10, sklearn silhouette 1e-10 / ARI 1e-12, K-means inertia
1e-8, RMT edge vs analytic MP 1e-10, DSR/PSR 1e-8)._

## Design decisions

See `docs/decisions/` for the ADRs (RMT-denoise, Mantegna metric, gap-vs-phase-null,
honest 1/N null, `k`-selection rule).

## License

MIT — see [LICENSE](LICENSE).
