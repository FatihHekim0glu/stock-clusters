"""Command-line interface (Typer).

A thin orchestration layer over the compute library: fetch data, cluster the
universe, optionally run the diversification horse race, and print/save the summary
and figures. Typer is built on the standard library, but constructing the app
object is deferred to :func:`build_app` so importing this module has no side effects
(no command registration or I/O at import time). The module-level ``app`` is a
lazily-built singleton consumed by the ``stock-clusters`` console-script entry
point.

Importing this module has no side effects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import typer

__all__ = ["build_app", "main"]


def build_app() -> typer.Typer:
    """Construct and return the Typer application.

    Registers the CLI commands (``run`` and ``demo``) on a fresh ``typer.Typer``
    instance. Typer is imported LAZILY inside this function so that importing
    :mod:`stockclusters.cli` does not import Typer or register any commands.

    Returns
    -------
    typer.Typer
        The configured Typer application.
    """
    # LAZY import: keep Typer off the import path of this pure module.
    import typer

    cli = typer.Typer(
        name="stock-clusters",
        add_completion=False,
        help="Cluster the S&P 500 by correlation structure (RMT-denoised, "
        "Mantegna distance) and honestly test cluster-aware allocation vs 1/N.",
        no_args_is_help=True,
    )

    @cli.command("run")
    def _run_command(
        tickers: list[str] = typer.Argument(  # noqa: B008
            ..., help="Asset symbols to fetch (e.g. AAPL MSFT GOOG)."
        ),
        start: str = typer.Option("2018-01-01", help="Inclusive start date (YYYY-MM-DD)."),
        end: str = typer.Option("2023-12-31", help="Inclusive end date (YYYY-MM-DD)."),
        method: str = typer.Option("both", help="Clustering method (hierarchical|kmeans|both)."),
        linkage: str = typer.Option("average", help="Linkage method (average|ward|single)."),
        n_clusters: int = typer.Option(0, help="Fixed cluster count; 0 ⇒ gap-statistic auto."),
        denoise: bool = typer.Option(True, help="RMT-denoise the correlation before clustering."),
        run_diversification: bool = typer.Option(
            False, help="Run the OOS 1/N vs cluster-aware horse race."
        ),
        cost_bps: float = typer.Option(5.0, help="Per-side transaction cost in basis points."),
        seed: int = typer.Option(0, help="Master RNG seed."),
    ) -> None:
        """Cluster ``tickers`` and (optionally) run the diversification horse race."""
        raise NotImplementedError

    @cli.command("demo")
    def _demo_command(
        seed: int = typer.Option(0, help="Master RNG seed for the synthetic panel."),
    ) -> None:
        """Run the pipeline on a small seeded synthetic block-correlation panel."""
        raise NotImplementedError

    return cli


def main() -> None:
    """Console-script entry point for the ``stock-clusters`` command.

    Builds the Typer app on demand and invokes it. Kept tiny so the heavy Typer
    import stays off the module import path.
    """
    build_app()()


if __name__ == "__main__":
    main()
