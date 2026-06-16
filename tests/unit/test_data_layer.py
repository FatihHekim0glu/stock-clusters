"""Unit tests for the reused data layer: RNG substreams, Polygon parsing, loaders.

These exercise the no-network paths of the data machinery (the synthetic price
fallback, the Polygon payload parser, API-key resolution, and the RNG substream
helpers) so the reused infrastructure is covered without touching the network.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from stockclusters._exceptions import ValidationError
from stockclusters._rng import make_rng, spawn_substreams

# --------------------------------------------------------------------------- #
# _rng                                                                          #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_make_rng_rejects_negative_seed() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        make_rng(-1)


@pytest.mark.unit
def test_spawn_substreams_is_deterministic_and_independent() -> None:
    a = spawn_substreams(7, 3)
    b = spawn_substreams(7, 3)
    assert len(a) == len(b) == 3
    # Same (seed, n) -> byte-identical children.
    for ga, gb in zip(a, b, strict=True):
        assert np.array_equal(ga.standard_normal(5), gb.standard_normal(5))
    # Distinct children are independent (different draws).
    fresh = spawn_substreams(7, 3)
    assert not np.array_equal(fresh[0].standard_normal(5), fresh[1].standard_normal(5))


@pytest.mark.unit
def test_spawn_substreams_validates_inputs() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        spawn_substreams(-1, 2)
    with pytest.raises(ValueError, match="non-negative"):
        spawn_substreams(0, -1)
    assert spawn_substreams(0, 0) == []


# --------------------------------------------------------------------------- #
# data.py synthetic + returns                                                  #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_synthetic_prices_are_deterministic_and_positive() -> None:
    from stockclusters.data import _synthetic_prices

    tickers = ["AAA", "BBB", "CCC"]
    p1 = _synthetic_prices(tickers, date(2020, 1, 1), date(2020, 6, 30))
    p2 = _synthetic_prices(tickers, date(2020, 1, 1), date(2020, 6, 30))
    pd.testing.assert_frame_equal(p1, p2)
    assert list(p1.columns) == tickers
    assert (p1.to_numpy() > 0).all()


@pytest.mark.unit
def test_get_prices_synthetic_fallback_offline() -> None:
    # yfinance/stooq are unreachable in CI; the auto chain falls through to the
    # deterministic synthetic panel rather than raising.
    from stockclusters.data import get_prices

    prices, source = get_prices(
        ["AAA", "BBB"], date(2020, 1, 1), date(2020, 4, 1), source_pref="stooq"
    )
    assert source in {"stooq", "synthetic"}
    assert prices.shape[1] == 2


@pytest.mark.unit
def test_get_prices_validates_inputs() -> None:
    from stockclusters.data import get_prices

    with pytest.raises(ValidationError, match="non-empty"):
        get_prices([], date(2020, 1, 1), date(2020, 2, 1))
    with pytest.raises(ValidationError, match="after start"):
        get_prices(["AAA", "BBB"], date(2020, 2, 1), date(2020, 1, 1))


@pytest.mark.unit
def test_compute_returns_simple_pct_change() -> None:
    from stockclusters.data import compute_returns

    prices = pd.DataFrame(
        {"AAA": [100.0, 110.0, 121.0], "BBB": [50.0, 55.0, 60.5]},
        index=pd.date_range("2020-01-01", periods=3, freq="B"),
    )
    rets = compute_returns(prices)
    assert rets.shape == (2, 2)
    # 110/100 - 1 == 0.10
    assert abs(rets.iloc[0, 0] - 0.10) < 1e-12
    assert abs(rets.iloc[1, 0] - 0.10) < 1e-12


# --------------------------------------------------------------------------- #
# Polygon provider (no network)                                                #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_polygon_resolve_api_key_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    from stockclusters.data_providers.polygon import _resolve_api_key

    # Explicit beats env.
    monkeypatch.setenv("POLYGON_API_KEY", "env-key")
    assert _resolve_api_key("explicit") == "explicit"
    # Env used when no explicit key.
    assert _resolve_api_key(None) == "env-key"


@pytest.mark.unit
def test_polygon_resolve_api_key_missing(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from stockclusters.data_providers import polygon as pg

    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    # Point the .env search at an empty dir so no key is found.
    monkeypatch.setattr(pg, "_load_api_key_from_dotenv", lambda: None)
    with pytest.raises(ValidationError, match="no API key"):
        pg._resolve_api_key(None)


@pytest.mark.unit
def test_polygon_series_from_payload_parses_bars() -> None:
    from stockclusters.data_providers.polygon import PolygonProvider

    payload = {
        "status": "OK",
        "results": [
            {"t": 1577836800000, "c": 100.0},  # 2020-01-01
            {"t": 1577923200000, "c": 101.5},  # 2020-01-02
        ],
    }
    series = PolygonProvider._series_from_payload(payload, "AAA")
    assert list(series.to_numpy()) == [100.0, 101.5]
    assert series.name == "AAA"


@pytest.mark.unit
def test_polygon_series_from_payload_empty_raises() -> None:
    from stockclusters.data_providers.polygon import PolygonProvider

    with pytest.raises(ValueError, match="no results"):
        PolygonProvider._series_from_payload({"status": "OK", "results": []}, "AAA")


@pytest.mark.unit
def test_polygon_fetch_validates_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    from stockclusters.data_providers.polygon import PolygonProvider

    monkeypatch.setenv("POLYGON_API_KEY", "test-key")
    provider = PolygonProvider()
    with pytest.raises(ValidationError, match="non-empty"):
        provider.fetch([], date(2020, 1, 1), date(2020, 2, 1))
    with pytest.raises(ValidationError, match="after start"):
        provider.fetch(["AAA"], date(2020, 2, 1), date(2020, 1, 1))


@pytest.mark.unit
def test_polygon_url_includes_key_and_dates(monkeypatch: pytest.MonkeyPatch) -> None:
    from stockclusters.data_providers.polygon import PolygonProvider

    monkeypatch.setenv("POLYGON_API_KEY", "secret123")
    provider = PolygonProvider()
    url = provider._url("AAPL", date(2020, 1, 1), date(2020, 12, 31))
    assert "AAPL" in url
    assert "2020-01-01" in url
    assert "2020-12-31" in url
    assert "secret123" in url


@pytest.mark.unit
def test_polygon_fetch_inner_joins(monkeypatch: pytest.MonkeyPatch) -> None:
    """fetch() inner-joins per-ticker series and preserves request order."""
    from stockclusters.data_providers import polygon as pg

    monkeypatch.setenv("POLYGON_API_KEY", "test-key")
    provider = pg.PolygonProvider()

    def _fake_get_json(ticker: str, start: date, end: date) -> dict:
        # AAA has 3 days, BBB only 2 (the first two) -> inner-join keeps 2.
        base = [
            {"t": 1577836800000, "c": 10.0},
            {"t": 1577923200000, "c": 11.0},
        ]
        if ticker == "AAA":
            base = [*base, {"t": 1578009600000, "c": 12.0}]
        return {"status": "OK", "results": base}

    monkeypatch.setattr(provider, "_get_json", _fake_get_json)
    frame = provider.fetch(["AAA", "BBB"], date(2020, 1, 1), date(2020, 1, 5))
    assert list(frame.columns) == ["AAA", "BBB"]
    assert len(frame) == 2  # inner-joined to the shared dates
    assert frame.index.is_monotonic_increasing
