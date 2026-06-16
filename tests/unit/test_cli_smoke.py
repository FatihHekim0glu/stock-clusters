"""CLI smoke tests (Group C).

Exercises the Typer app end-to-end on a deterministic synthetic panel (no network):
``--help`` for the root and both commands, a tiny ``run`` (cluster only), a ``run``
with the diversification horse race, and the ``demo`` command. Also pins the
import-purity contract: importing :mod:`stockclusters.cli` must not import Typer.
"""

from __future__ import annotations

import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    import typer
    from typer.testing import CliRunner


def _runner() -> tuple[CliRunner, typer.Typer]:
    """A Typer ``CliRunner`` bound to the lazily-built app (lazy imports)."""
    from typer.testing import CliRunner

    from stockclusters.cli import build_app

    return CliRunner(), build_app()


@pytest.mark.unit
def test_import_cli_does_not_import_typer() -> None:
    """Importing stockclusters.cli must not pull Typer onto the import path."""
    code = "import sys; import stockclusters.cli; assert 'typer' not in sys.modules; print('pure')"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert out.returncode == 0, out.stderr
    assert "pure" in out.stdout


@pytest.mark.unit
def test_root_help() -> None:
    """The root --help lists the run and demo commands."""
    runner, app = _runner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "run" in result.output
    assert "demo" in result.output


@pytest.mark.unit
@pytest.mark.parametrize("command", ["run", "demo"])
def test_subcommand_help(command: str) -> None:
    """Each subcommand exposes its own --help without executing anything."""
    runner, app = _runner()
    result = runner.invoke(app, [command, "--help"])
    assert result.exit_code == 0


@pytest.mark.unit
def test_run_cluster_only_synthetic() -> None:
    """A tiny cluster-only run on synthetic tickers prints the cluster summary."""
    runner, app = _runner()
    result = runner.invoke(
        app,
        [
            "run",
            "SYN00",
            "SYN01",
            "SYN02",
            "SYN03",
            "SYN04",
            "SYN05",
            "--start",
            "2019-01-01",
            "--end",
            "2021-12-31",
            "--n-clusters",
            "2",
            "--seed",
            "0",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "stock-clusters" in result.output
    assert "clusters (k)" in result.output
    assert "data source        : synthetic" in result.output
    # Diversification block must NOT appear when the flag is off.
    assert "diversification horse race" not in result.output


@pytest.mark.unit
def test_run_with_diversification_synthetic() -> None:
    """A run with --run-diversification prints the honest horse-race verdict."""
    runner, app = _runner()
    result = runner.invoke(
        app,
        [
            "run",
            "SYN00",
            "SYN01",
            "SYN02",
            "SYN03",
            "SYN04",
            "SYN05",
            "--start",
            "2019-01-01",
            "--end",
            "2021-12-31",
            "--n-clusters",
            "2",
            "--run-diversification",
            "--cost-bps",
            "5",
            "--seed",
            "0",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "diversification horse race" in result.output
    assert "Memmel-JK p-value" in result.output
    assert "deflated Sharpe" in result.output
    assert "verdict" in result.output


@pytest.mark.unit
def test_demo_runs_offline() -> None:
    """The demo command runs the full pipeline on a seeded synthetic panel."""
    runner, app = _runner()
    result = runner.invoke(app, ["demo", "--seed", "0"])
    assert result.exit_code == 0, result.output
    assert "data source        : synthetic" in result.output
