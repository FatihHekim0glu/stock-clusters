# Contributing

Thanks for your interest in `stock-clusters`. This project uses
[uv](https://docs.astral.sh/uv/) for environment and dependency management.

## Dev setup

```bash
# 1. Install uv (https://docs.astral.sh/uv/getting-started/installation/)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Create the env and install the project with every extra + dev tooling.
uv sync --all-extras --dev
# (Equivalent: uv pip install -e ".[data,viz,dev]" inside an activated venv.)
```

`uv sync` creates `.venv/` and installs the locked dependency set from
`uv.lock`. Prefix commands with `uv run` to use that env without activating it.

## Quality gates

These are exactly what CI runs (see `.github/workflows/ci.yml`). Run them locally
before opening a pull request:

```bash
uv run ruff check src                                                       # lint
uv run mypy src                                                             # types (strict)
uv run pytest -q --cov=stockclusters --cov-report=term --cov-fail-under=80  # tests + coverage
```

- **Lint** (`ruff`) must pass.
- **Types** (`mypy --strict`) is run on every PR. It is currently non-blocking in
  CI while residual strict-mode issues are burned down, but new code should not
  add type errors.
- **Tests** (`pytest`) must pass with **coverage ≥ 80%** (the gate also lives in
  `[tool.coverage.report] fail_under` in `pyproject.toml`).

CI runs the full matrix on Python 3.11, 3.12, and 3.13.

## Import purity

`src/stockclusters/` must have ZERO import-time side effects: no network, scheduler,
or loop at module load, and optional heavy deps (Plotly, Typer) are imported lazily
inside functions. This is what makes the package safe to vendor into the lean
FastAPI container.

## Commit hygiene

- Use clear, present-tense commit messages.
- Do not add automated co-author or generated-with trailers to commit messages.

## Pull requests

- Branch off `main`; keep PRs focused.
- Make sure the three quality gates above are green locally.
- Update `CHANGELOG.md` (under `[Unreleased]`) when behaviour changes.
