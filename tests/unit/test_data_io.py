"""Unit tests for the data I/O layer (:mod:`stockclusters.data`).

These exercise the network-touching machinery of the data loader *without any
real network*. Every upstream library (yfinance, pandas-datareader, curl_cffi,
the Polygon provider) is imported lazily *inside* the functions under test, so we
monkeypatch the lazily-imported modules (or the ``_fetch_*`` helpers) to return
canned frames or raise, then assert on the observable behaviour:

- the ``get_prices`` fallback chain order (polygon -> yfinance -> stooq ->
  synthetic) and the ``data_source`` label / return shape at every rung;
- ``_extract_close`` over the messy real-provider shapes (single vs multi-ticker,
  MultiIndex columns, missing/partial tickers, empty frames);
- ``compute_returns`` correctness (pct_change with ``fill_method=None``, NaN
  handling, single column);
- ``get_risk_free`` / ``_fetch_risk_free_annual`` success via a mocked
  pandas-datareader *and* the synthetic-fallback branch, plus index alignment and
  deannualization.

This file complements ``test_data_layer.py`` (which covers the synthetic panel,
input validation, and the Polygon payload parser) by driving the *fetcher* and
*fallback orchestration* paths those tests deliberately leave to the network.
"""

from __future__ import annotations

import sys
import types
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
import pytest

import stockclusters.data as data_mod
from stockclusters._exceptions import ValidationError
from stockclusters.data import (
    _business_days,
    _extract_close,
    _fetch_polygon,
    _fetch_risk_free_annual,
    _fetch_stooq,
    _fetch_yfinance,
    _synthetic_prices,
    compute_returns,
    get_prices,
    get_risk_free,
)

START = date(2020, 1, 1)
END = date(2020, 4, 1)


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def _wide_panel(tickers: list[str], n: int = 5, base: float = 100.0) -> pd.DataFrame:
    """A clean wide ``date x ticker`` panel of strictly increasing closes."""
    index = pd.date_range("2020-01-01", periods=n, freq="B")
    data = {t: base + i + np.arange(n, dtype="float64") for i, t in enumerate(tickers)}
    return pd.DataFrame(data, index=index)


def _install_fake_module(
    monkeypatch: pytest.MonkeyPatch, name: str, **attrs: Any
) -> types.ModuleType:
    """Register a throwaway module under ``name`` so ``import name`` finds it.

    Used to intercept the lazy ``import yfinance`` / ``from pandas_datareader
    import data as pdr`` / ``from curl_cffi import requests`` statements without a
    real network round-trip.
    """
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    monkeypatch.setitem(sys.modules, name, mod)
    return mod


# =========================================================================== #
# get_prices: input validation (lines 213-216)                                #
# =========================================================================== #
@pytest.mark.unit
def test_get_prices_rejects_empty_tickers() -> None:
    with pytest.raises(ValidationError, match="non-empty"):
        get_prices([], START, END)


@pytest.mark.unit
def test_get_prices_rejects_end_not_after_start() -> None:
    with pytest.raises(ValidationError, match="after start"):
        get_prices(["AAA"], END, START)  # end <= start


# =========================================================================== #
# get_prices: fallback chain order + per-rung data_source / shape             #
# =========================================================================== #
@pytest.mark.unit
def test_get_prices_polygon_rung_used_first() -> None:
    """source_pref='polygon' returns polygon data when the provider succeeds."""
    tickers = ["AAA", "BBB"]
    panel = _wide_panel(tickers)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(data_mod, "_fetch_polygon", lambda t, s, e: panel)
        # The remaining rungs must NOT be consulted; make them explode if reached.
        mp.setattr(data_mod, "_fetch_yfinance", lambda t, s, e: pytest.fail("yfinance reached"))
        mp.setattr(data_mod, "_fetch_stooq", lambda t, s, e: pytest.fail("stooq reached"))
        frame, source = get_prices(tickers, START, END, source_pref="polygon")
    assert source == "polygon"
    assert list(frame.columns) == tickers
    assert frame.shape == panel.shape
    assert frame.to_numpy().dtype == np.float64


@pytest.mark.unit
def test_get_prices_polygon_falls_through_to_yfinance() -> None:
    """polygon raising -> the chain advances to yfinance (line 222/238-242 path)."""
    tickers = ["AAA", "BBB"]
    panel = _wide_panel(tickers)
    calls: list[str] = []

    def _polygon_boom(t: list[str], s: date, e: date) -> pd.DataFrame:
        calls.append("polygon")
        raise ValueError("polygon down")

    def _yf_ok(t: list[str], s: date, e: date) -> pd.DataFrame:
        calls.append("yfinance")
        return panel

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(data_mod, "_fetch_polygon", _polygon_boom)
        mp.setattr(data_mod, "_fetch_yfinance", _yf_ok)
        mp.setattr(data_mod, "_fetch_stooq", lambda t, s, e: pytest.fail("stooq reached"))
        frame, source = get_prices(tickers, START, END, source_pref="polygon")
    assert calls == ["polygon", "yfinance"]  # order: polygon tried, then yfinance
    assert source == "yfinance"
    assert list(frame.columns) == tickers


@pytest.mark.unit
def test_get_prices_yfinance_falls_through_to_stooq_in_auto() -> None:
    """auto chain: yfinance failing -> stooq is used (and labelled 'stooq')."""
    tickers = ["AAA", "BBB"]
    panel = _wide_panel(tickers)
    calls: list[str] = []

    def _yf_boom(t: list[str], s: date, e: date) -> pd.DataFrame:
        calls.append("yfinance")
        raise RuntimeError("yfinance down")

    def _stooq_ok(t: list[str], s: date, e: date) -> pd.DataFrame:
        calls.append("stooq")
        return panel

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(data_mod, "_fetch_yfinance", _yf_boom)
        mp.setattr(data_mod, "_fetch_stooq", _stooq_ok)
        frame, source = get_prices(tickers, START, END, source_pref="auto")
    assert calls == ["yfinance", "stooq"]
    assert source == "stooq"
    assert frame.shape == panel.shape


@pytest.mark.unit
def test_get_prices_all_real_sources_fail_synthetic_fallback() -> None:
    """Every real rung failing -> deterministic synthetic panel (lines 246-248)."""
    tickers = ["AAA", "BBB", "CCC"]

    def _boom(t: list[str], s: date, e: date) -> pd.DataFrame:
        raise ConnectionError("offline")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(data_mod, "_fetch_polygon", _boom)
        mp.setattr(data_mod, "_fetch_yfinance", _boom)
        mp.setattr(data_mod, "_fetch_stooq", _boom)
        frame, source = get_prices(tickers, START, END, source_pref="polygon")
    assert source == "synthetic"
    assert list(frame.columns) == tickers
    # Synthetic panel is deterministic and strictly positive (matches _synthetic_prices).
    expected = _synthetic_prices(tickers, START, END)
    pd.testing.assert_frame_equal(frame, expected)
    assert (frame.to_numpy() > 0).all()


@pytest.mark.unit
def test_get_prices_empty_frame_advances_chain() -> None:
    """A fetcher returning an EMPTY (non-raising) frame is skipped (line 243 guard)."""
    tickers = ["AAA", "BBB"]
    panel = _wide_panel(tickers)
    calls: list[str] = []

    def _yf_empty(t: list[str], s: date, e: date) -> pd.DataFrame:
        calls.append("yfinance")
        return pd.DataFrame()  # empty -> ``not frame.empty`` is False -> advance

    def _stooq_ok(t: list[str], s: date, e: date) -> pd.DataFrame:
        calls.append("stooq")
        return panel

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(data_mod, "_fetch_yfinance", _yf_empty)
        mp.setattr(data_mod, "_fetch_stooq", _stooq_ok)
        frame, source = get_prices(tickers, START, END, source_pref="auto")
    assert calls == ["yfinance", "stooq"]
    assert source == "stooq"
    assert not frame.empty


@pytest.mark.unit
def test_get_prices_yfinance_pref_single_rung_then_synthetic() -> None:
    """source_pref='yfinance' is a single-rung chain (line 228) -> synthetic on fail."""
    tickers = ["AAA"]
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(data_mod, "_fetch_yfinance", lambda t, s, e: (_ for _ in ()).throw(OSError()))
        # stooq must never be consulted for the 'yfinance' preference.
        mp.setattr(data_mod, "_fetch_stooq", lambda t, s, e: pytest.fail("stooq reached"))
        frame, source = get_prices(tickers, START, END, source_pref="yfinance")
    assert source == "synthetic"
    assert list(frame.columns) == tickers


@pytest.mark.unit
def test_get_prices_stooq_pref_success_returns_stooq() -> None:
    """source_pref='stooq' single-rung chain (line 230) returns stooq when it works."""
    tickers = ["AAA", "BBB"]
    panel = _wide_panel(tickers)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(data_mod, "_fetch_stooq", lambda t, s, e: panel)
        mp.setattr(data_mod, "_fetch_yfinance", lambda t, s, e: pytest.fail("yfinance reached"))
        frame, source = get_prices(tickers, START, END, source_pref="stooq")
    assert source == "stooq"
    pd.testing.assert_frame_equal(frame, panel.astype("float64"))


@pytest.mark.unit
def test_get_prices_use_cache_flag_is_accepted_noop() -> None:
    """use_cache is accepted for API parity (line 236 ``del use_cache``)."""
    tickers = ["AAA"]
    panel = _wide_panel(tickers)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(data_mod, "_fetch_yfinance", lambda t, s, e: panel)
        frame_a, src_a = get_prices(tickers, START, END, source_pref="yfinance", use_cache=True)
        frame_b, src_b = get_prices(tickers, START, END, source_pref="yfinance", use_cache=False)
    assert src_a == src_b == "yfinance"
    pd.testing.assert_frame_equal(frame_a, frame_b)


# =========================================================================== #
# _fetch_polygon: lazy PolygonProvider import + empty guard (lines 87-92)      #
# =========================================================================== #
@pytest.mark.unit
def test_fetch_polygon_returns_provider_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    """_fetch_polygon imports PolygonProvider lazily and returns its frame."""
    import stockclusters.data_providers.polygon as pg

    tickers = ["AAA", "BBB"]
    panel = _wide_panel(tickers)
    captured: dict[str, Any] = {}

    class _FakeProvider:
        def fetch(self, t: list[str], s: date, e: date) -> pd.DataFrame:
            captured["args"] = (t, s, e)
            return panel

    # Patch the symbol that ``from ...polygon import PolygonProvider`` resolves to.
    monkeypatch.setattr(pg, "PolygonProvider", _FakeProvider)
    frame = _fetch_polygon(tickers, START, END)
    assert captured["args"] == (tickers, START, END)
    pd.testing.assert_frame_equal(frame, panel)


@pytest.mark.unit
def test_fetch_polygon_empty_frame_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty Polygon frame -> ValueError so get_prices falls through (lines 90-91)."""
    import stockclusters.data_providers.polygon as pg

    class _EmptyProvider:
        def fetch(self, t: list[str], s: date, e: date) -> pd.DataFrame:
            return pd.DataFrame()

    monkeypatch.setattr(pg, "PolygonProvider", _EmptyProvider)
    with pytest.raises(ValueError, match="no usable price data"):
        _fetch_polygon(["AAA"], START, END)


@pytest.mark.unit
def test_fetch_polygon_all_nan_frame_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """An all-NaN (non-empty) Polygon frame is also rejected (line 90 ``isna().all``)."""
    import stockclusters.data_providers.polygon as pg

    index = pd.date_range("2020-01-01", periods=2, freq="B")
    nan_frame = pd.DataFrame({"AAA": [np.nan, np.nan]}, index=index)

    class _NanProvider:
        def fetch(self, t: list[str], s: date, e: date) -> pd.DataFrame:
            return nan_frame

    monkeypatch.setattr(pg, "PolygonProvider", _NanProvider)
    with pytest.raises(ValueError, match="no usable price data"):
        _fetch_polygon(["AAA"], START, END)


# =========================================================================== #
# _fetch_yfinance: lazy yfinance + curl_cffi, with _extract_close             #
# =========================================================================== #
@pytest.mark.unit
def test_fetch_yfinance_uses_curl_cffi_session_and_parses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Happy path: curl_cffi session is built and passed to yf.download (lines 97-123)."""
    tickers = ["AAA", "BBB"]
    captured: dict[str, Any] = {}

    # A multi-ticker yfinance frame: MultiIndex columns (field, ticker) w/ 'Close'.
    index = pd.date_range("2020-01-01", periods=4, freq="B")
    cols = pd.MultiIndex.from_product([["Close", "Volume"], tickers])
    values = np.arange(4 * 4, dtype="float64").reshape(4, 4) + 10.0
    raw = pd.DataFrame(values, index=index, columns=cols)

    def _fake_download(passed_tickers: Any, **kwargs: Any) -> pd.DataFrame:
        captured["tickers"] = passed_tickers
        captured["kwargs"] = kwargs
        return raw

    class _FakeSession:
        def __init__(self, impersonate: str | None = None) -> None:
            captured["impersonate"] = impersonate

    fake_requests = types.SimpleNamespace(Session=_FakeSession)
    _install_fake_module(monkeypatch, "yfinance", download=_fake_download)
    _install_fake_module(monkeypatch, "curl_cffi", requests=fake_requests)
    monkeypatch.setitem(sys.modules, "curl_cffi.requests", fake_requests)

    frame = _fetch_yfinance(tickers, START, END)

    # curl_cffi Chrome impersonation session was created and threaded through.
    assert captured["impersonate"] == "chrome"
    assert "session" in captured["kwargs"]
    assert isinstance(captured["kwargs"]["session"], _FakeSession)
    # ``end`` is bumped by one day (yfinance end is exclusive) and start is ISO.
    assert captured["kwargs"]["start"] == START.isoformat()
    assert captured["kwargs"]["end"] == (END + timedelta(days=1)).isoformat()
    assert captured["kwargs"]["auto_adjust"] is True
    # Parsed to a clean wide panel of the 'Close' level in request order.
    assert list(frame.columns) == tickers
    assert frame.shape == (4, 2)


@pytest.mark.unit
def test_fetch_yfinance_without_curl_cffi_omits_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """curl_cffi import failing -> session=None, no ``session`` kwarg (lines 104-117)."""
    tickers = ["AAA"]
    captured: dict[str, Any] = {}

    index = pd.date_range("2020-01-01", periods=3, freq="B")
    # Single-ticker frame with flat OHLCV columns (yfinance single-symbol shape).
    raw = pd.DataFrame(
        {"Open": [1.0, 2.0, 3.0], "Close": [10.0, 11.0, 12.0]},
        index=index,
    )

    def _fake_download(passed_tickers: Any, **kwargs: Any) -> pd.DataFrame:
        captured["kwargs"] = kwargs
        return raw

    # curl_cffi.requests.Session raises -> the except branch sets session=None.
    class _BoomSession:
        def __init__(self, impersonate: str | None = None) -> None:
            raise RuntimeError("no curl_cffi backend")

    fake_requests = types.SimpleNamespace(Session=_BoomSession)
    _install_fake_module(monkeypatch, "yfinance", download=_fake_download)
    _install_fake_module(monkeypatch, "curl_cffi", requests=fake_requests)
    monkeypatch.setitem(sys.modules, "curl_cffi.requests", fake_requests)

    frame = _fetch_yfinance(tickers, START, END)
    assert "session" not in captured["kwargs"]
    # Single-ticker OHLCV -> 'Close' renamed to the requested ticker.
    assert list(frame.columns) == tickers
    assert list(frame.iloc[:, 0]) == [10.0, 11.0, 12.0]


@pytest.mark.unit
def test_fetch_yfinance_empty_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """yfinance returning an all-NaN/empty frame -> ValueError (lines 121-122)."""
    tickers = ["AAA"]

    def _fake_download(passed_tickers: Any, **kwargs: Any) -> pd.DataFrame:
        return pd.DataFrame()  # no columns -> _extract_close yields empty frame

    _install_fake_module(monkeypatch, "yfinance", download=_fake_download)
    # Force the curl_cffi branch to the except path (session=None).
    monkeypatch.setitem(sys.modules, "curl_cffi", types.ModuleType("curl_cffi"))
    monkeypatch.delitem(sys.modules, "curl_cffi.requests", raising=False)

    with pytest.raises(ValueError, match="no usable price data"):
        _fetch_yfinance(tickers, START, END)


# =========================================================================== #
# _fetch_stooq: lazy pandas_datareader                                        #
# =========================================================================== #
@pytest.mark.unit
def test_fetch_stooq_sorts_ascending_and_parses(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stooq returns DESCENDING dates -> _fetch_stooq sorts ascending (lines 130-135)."""
    tickers = ["AAA", "BBB"]
    captured: dict[str, Any] = {}

    # Stooq multi-ticker shape: MultiIndex (field, ticker), dates DESCENDING.
    index = pd.DatetimeIndex(["2020-01-03", "2020-01-02", "2020-01-01"])
    cols = pd.MultiIndex.from_product([["Close", "Open"], tickers])
    values = np.arange(3 * 4, dtype="float64").reshape(3, 4) + 1.0
    raw = pd.DataFrame(values, index=index, columns=cols)

    def _fake_datareader(passed_tickers: Any, source: str, **kwargs: Any) -> pd.DataFrame:
        captured["source"] = source
        captured["tickers"] = passed_tickers
        return raw

    fake_pdr = types.SimpleNamespace(DataReader=_fake_datareader)
    fake_module = _install_fake_module(monkeypatch, "pandas_datareader")
    # ``from pandas_datareader import data as pdr`` -> need the ``data`` attribute.
    fake_module.data = fake_pdr  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pandas_datareader.data", fake_pdr)

    frame = _fetch_stooq(tickers, START, END)
    assert captured["source"] == "stooq"
    assert list(frame.columns) == tickers
    # Sorted ascending: index is monotonic increasing after the call.
    assert frame.index.is_monotonic_increasing
    assert list(frame.index) == sorted(frame.index)


@pytest.mark.unit
def test_fetch_stooq_empty_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stooq all-NaN frame -> ValueError (line 132-133 guard)."""
    tickers = ["AAA"]
    index = pd.date_range("2020-01-01", periods=2, freq="B")
    raw = pd.DataFrame({"Close": [np.nan, np.nan]}, index=index)

    fake_pdr = types.SimpleNamespace(DataReader=lambda t, source, **k: raw)
    fake_module = _install_fake_module(monkeypatch, "pandas_datareader")
    fake_module.data = fake_pdr  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pandas_datareader.data", fake_pdr)

    with pytest.raises(ValueError, match="no usable price data"):
        _fetch_stooq(tickers, START, END)


# =========================================================================== #
# _extract_close: messy provider shapes (lines 141-165)                       #
# =========================================================================== #
@pytest.mark.unit
def test_extract_close_multiindex_prefers_adj_close() -> None:
    """MultiIndex columns: 'Adj Close' level is preferred over 'Close' (line 145-147)."""
    tickers = ["AAA", "BBB"]
    index = pd.date_range("2020-01-01", periods=3, freq="B")
    cols = pd.MultiIndex.from_product([["Adj Close", "Close"], tickers])
    values = np.arange(3 * 4, dtype="float64").reshape(3, 4)
    raw = pd.DataFrame(values, index=index, columns=cols)

    frame = _extract_close(raw, tickers)
    assert list(frame.columns) == tickers
    # 'Adj Close' block is columns 0,1 of the value grid (not the 'Close' block).
    expected = pd.DataFrame(raw["Adj Close"])[tickers].astype("float64")
    pd.testing.assert_frame_equal(frame, expected)


@pytest.mark.unit
def test_extract_close_multiindex_falls_back_to_close() -> None:
    """MultiIndex with only 'Close' (no 'Adj Close') selects 'Close' (line 145-148)."""
    tickers = ["AAA", "BBB"]
    index = pd.date_range("2020-01-01", periods=2, freq="B")
    cols = pd.MultiIndex.from_product([["Close", "Volume"], tickers])
    values = np.arange(2 * 4, dtype="float64").reshape(2, 4) + 5.0
    raw = pd.DataFrame(values, index=index, columns=cols)

    frame = _extract_close(raw, tickers)
    expected = pd.DataFrame(raw["Close"])[tickers].astype("float64")
    pd.testing.assert_frame_equal(frame, expected)


@pytest.mark.unit
def test_extract_close_multiindex_no_price_level_uses_first_level() -> None:
    """MultiIndex lacking Adj Close/Close -> xs of the first level (lines 149-150)."""
    tickers = ["AAA", "BBB"]
    index = pd.date_range("2020-01-01", periods=2, freq="B")
    # No 'Close'/'Adj Close' field present at all -> the ``else`` (xs) branch.
    cols = pd.MultiIndex.from_product([["Mid", "Bid"], tickers])
    values = np.arange(2 * 4, dtype="float64").reshape(2, 4) + 7.0
    raw = pd.DataFrame(values, index=index, columns=cols)

    frame = _extract_close(raw, tickers)
    # The else branch uses ``raw.columns.levels[0][0]`` which is the *sorted*-first
    # level label. For {'Mid','Bid'} that is 'Bid', not 'Mid'.
    multi_cols = raw.columns
    assert isinstance(multi_cols, pd.MultiIndex)
    assert multi_cols.levels[0][0] == "Bid"
    expected = pd.DataFrame(raw.xs("Bid", axis=1, level=0))[tickers].astype("float64")
    pd.testing.assert_frame_equal(frame, expected)


@pytest.mark.unit
def test_extract_close_single_ticker_ohlcv_renames_close() -> None:
    """Flat single-ticker OHLCV: 'Close' is selected + renamed (lines 151-154)."""
    tickers = ["AAA"]
    index = pd.date_range("2020-01-01", periods=3, freq="B")
    raw = pd.DataFrame(
        {"Open": [1.0, 2.0, 3.0], "High": [2.0, 3.0, 4.0], "Close": [9.0, 8.0, 7.0]},
        index=index,
    )
    frame = _extract_close(raw, tickers)
    assert list(frame.columns) == ["AAA"]
    assert list(frame["AAA"]) == [9.0, 8.0, 7.0]


@pytest.mark.unit
def test_extract_close_already_wide_passthrough() -> None:
    """A frame that is already wide (no 'Close' col) passes through (lines 156-157)."""
    tickers = ["AAA", "BBB"]
    index = pd.date_range("2020-01-01", periods=3, freq="B")
    raw = pd.DataFrame({"AAA": [1.0, 2.0, 3.0], "BBB": [4.0, 5.0, 6.0]}, index=index)
    frame = _extract_close(raw, tickers)
    assert list(frame.columns) == tickers
    pd.testing.assert_frame_equal(frame, raw.astype("float64"))


@pytest.mark.unit
def test_extract_close_keeps_only_present_tickers_in_request_order() -> None:
    """Missing tickers are dropped; present ones kept in request order (lines 160-163)."""
    requested = ["AAA", "MISSING", "BBB"]
    index = pd.date_range("2020-01-01", periods=2, freq="B")
    # Frame has BBB before AAA and an extra unrequested column.
    raw = pd.DataFrame(
        {"BBB": [4.0, 5.0], "AAA": [1.0, 2.0], "EXTRA": [9.0, 9.0]},
        index=index,
    )
    frame = _extract_close(raw, requested)
    # Only AAA, BBB survive and they follow request order (AAA before BBB).
    assert list(frame.columns) == ["AAA", "BBB"]


@pytest.mark.unit
def test_extract_close_no_present_tickers_keeps_frame() -> None:
    """When none of the requested tickers are present, ``present`` is empty (line 162)."""
    requested = ["XXX", "YYY"]
    index = pd.date_range("2020-01-01", periods=2, freq="B")
    raw = pd.DataFrame({"AAA": [1.0, 2.0], "BBB": [3.0, 4.0]}, index=index)
    frame = _extract_close(raw, requested)
    # No filtering applied (the ``if present`` guard is False) -> columns unchanged.
    assert list(frame.columns) == ["AAA", "BBB"]


@pytest.mark.unit
def test_extract_close_coerces_index_to_datetime() -> None:
    """The output index is coerced to a DatetimeIndex (line 164)."""
    tickers = ["AAA"]
    raw = pd.DataFrame({"AAA": [1.0, 2.0]}, index=["2020-01-01", "2020-01-02"])
    frame = _extract_close(raw, tickers)
    assert isinstance(frame.index, pd.DatetimeIndex)


# =========================================================================== #
# _synthetic_prices: empty-span branch (line 58-59)                           #
# =========================================================================== #
@pytest.mark.unit
def test_synthetic_prices_empty_span_returns_empty_typed_frame() -> None:
    """A span with no business days -> an empty, correctly-columned frame (line 59)."""
    # 2021-01-02 is a Saturday, 2021-01-03 a Sunday: zero business days inclusive.
    tickers = ["AAA", "BBB"]
    frame = _synthetic_prices(tickers, date(2021, 1, 2), date(2021, 1, 3))
    assert frame.empty
    assert len(frame.index) == 0
    assert list(frame.columns) == tickers
    assert frame.dtypes.apply(lambda d: d == np.float64).all()


# =========================================================================== #
# compute_returns: pct_change(fill_method=None) + log/NaN correctness         #
# =========================================================================== #
@pytest.mark.unit
def test_compute_returns_drops_leading_nan_row_and_values() -> None:
    """First all-NaN row is dropped; simple pct returns are exact."""
    prices = pd.DataFrame(
        {"AAA": [100.0, 110.0, 99.0], "BBB": [50.0, 50.0, 75.0]},
        index=pd.date_range("2020-01-01", periods=3, freq="B"),
    )
    rets = compute_returns(prices)
    assert rets.shape == (2, 2)
    assert abs(rets.iloc[0]["AAA"] - 0.10) < 1e-12  # 110/100 - 1
    assert abs(rets.iloc[1]["AAA"] - (-0.10)) < 1e-12  # 99/110 - 1
    assert rets.iloc[0]["BBB"] == 0.0  # 50/50 - 1
    assert abs(rets.iloc[1]["BBB"] - 0.5) < 1e-12  # 75/50 - 1
    assert rets.to_numpy().dtype == np.float64


@pytest.mark.unit
def test_compute_returns_no_lookahead_does_not_fill_gaps() -> None:
    """With an internal NaN, ``fill_method=None`` does NOT manufacture a 0 return.

    If prices were forward-filled before differencing, the row at the gap would read
    0.0 and the row after it 0.21 — but ffill would also have leaked the stale price
    forward. With ``fill_method=None`` BOTH the gap row and the row immediately after
    stay NaN (no spurious 0 return is invented across the missing observation), which
    is the no-lookahead guarantee the docstring promises. The unaffected columns are
    differenced cleanly.
    """
    prices = pd.DataFrame(
        {"AAA": [100.0, np.nan, 121.0], "BBB": [10.0, 11.0, 22.0]},
        index=pd.date_range("2020-01-01", periods=3, freq="B"),
    )
    rets = compute_returns(prices)
    # Gap row and the row after it are NaN for AAA (no manufactured 0 return).
    assert np.isnan(rets.iloc[0]["AAA"])
    assert np.isnan(rets.iloc[1]["AAA"])
    # A column without gaps is differenced normally (sanity that this is real).
    assert abs(rets.iloc[0]["BBB"] - 0.10) < 1e-12  # 11/10 - 1
    assert abs(rets.iloc[1]["BBB"] - 1.0) < 1e-12  # 22/11 - 1

    # Contrast: ffill-then-diff WOULD have manufactured a spurious 0.0 at the gap.
    ffilled = prices.ffill().pct_change(fill_method=None).iloc[1:]
    assert ffilled.iloc[0]["AAA"] == 0.0  # the spurious zero we deliberately avoid


@pytest.mark.unit
def test_compute_returns_single_column() -> None:
    """A single-column panel returns a single-column frame, leading row dropped."""
    prices = pd.DataFrame(
        {"AAA": [10.0, 12.0, 6.0]},
        index=pd.date_range("2020-01-01", periods=3, freq="B"),
    )
    rets = compute_returns(prices)
    assert rets.shape == (2, 1)
    assert abs(rets.iloc[0]["AAA"] - 0.2) < 1e-12  # 12/10 - 1
    assert abs(rets.iloc[1]["AAA"] - (-0.5)) < 1e-12  # 6/12 - 1


# =========================================================================== #
# get_risk_free / _fetch_risk_free_annual                                     #
# =========================================================================== #
@pytest.mark.unit
def test_get_risk_free_validates_end_after_start() -> None:
    with pytest.raises(ValidationError, match="after start"):
        get_risk_free(date(2020, 2, 1), date(2020, 1, 1))


@pytest.mark.unit
def test_fetch_risk_free_annual_success_via_mocked_pdr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mocked FRED (pandas_datareader) success: percent -> fraction, aligned to index."""
    index = _business_days(START, date(2020, 1, 10))
    # FRED quotes percent; DGS3MO ~ 1.5% on a sparse calendar that is ffill/bfill-ed.
    fred_index = pd.DatetimeIndex(["2020-01-02", "2020-01-06"])
    raw = pd.DataFrame({"DGS3MO": [1.5, 3.0]}, index=fred_index)

    captured: dict[str, Any] = {}

    def _fake_datareader(name: str, source: str, **kwargs: Any) -> pd.DataFrame:
        captured["name"] = name
        captured["source"] = source
        return raw

    fake_pdr = types.SimpleNamespace(DataReader=_fake_datareader)
    fake_module = _install_fake_module(monkeypatch, "pandas_datareader")
    fake_module.data = fake_pdr  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pandas_datareader.data", fake_pdr)

    annual = _fetch_risk_free_annual(START, date(2020, 1, 10), index)
    assert captured["name"] == "DGS3MO"
    assert captured["source"] == "fred"
    # Result is aligned exactly to the supplied business-day index.
    assert list(annual.index) == list(index)
    # Percent converted to fraction; first value reflects the 1.5% reading.
    assert abs(annual.iloc[0] - 0.015) < 1e-12
    # No NaNs left after ffill/bfill across the grid.
    assert not annual.isna().any()


@pytest.mark.unit
def test_fetch_risk_free_annual_empty_series_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An all-NaN FRED series raises internally and falls back to flat 2% (lines 347-352)."""
    index = _business_days(START, date(2020, 1, 10))
    # All-NaN -> after ffill/bfill still all-NaN -> ``raise ValueError`` -> except.
    raw = pd.DataFrame(
        {"DGS3MO": [np.nan, np.nan]},
        index=pd.DatetimeIndex(
            ["2019-06-01", "2019-06-02"]  # outside the requested window too
        ),
    )
    fake_pdr = types.SimpleNamespace(DataReader=lambda n, source, **k: raw)
    fake_module = _install_fake_module(monkeypatch, "pandas_datareader")
    fake_module.data = fake_pdr  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pandas_datareader.data", fake_pdr)

    annual = _fetch_risk_free_annual(START, date(2020, 1, 10), index)
    # Flat 2% synthetic fallback, aligned to the requested index.
    assert list(annual.index) == list(index)
    assert (annual.to_numpy() == 0.02).all()


@pytest.mark.unit
def test_fetch_risk_free_annual_import_error_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pandas_datareader unimportable -> flat 2% synthetic fallback (except branch)."""
    index = _business_days(START, date(2020, 1, 10))

    # Force ``from pandas_datareader import data as pdr`` to raise ImportError.
    real_import = __import__

    def _blocking_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "pandas_datareader" or name.startswith("pandas_datareader."):
            raise ImportError("pandas_datareader not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "pandas_datareader", raising=False)
    monkeypatch.delitem(sys.modules, "pandas_datareader.data", raising=False)
    monkeypatch.setattr("builtins.__import__", _blocking_import)

    annual = _fetch_risk_free_annual(START, date(2020, 1, 10), index)
    assert (annual.to_numpy() == 0.02).all()
    assert list(annual.index) == list(index)


@pytest.mark.unit
def test_get_risk_free_deannualizes_and_aligns(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_risk_free deannualizes the annual rate to a per-period rate (lines 320-327)."""

    # Patch the (internal) annual fetcher so the test is network-free and exact.
    def _flat_annual(start: date, end: date, index: pd.DatetimeIndex) -> pd.Series:
        return pd.Series(0.04, index=index, dtype="float64")

    monkeypatch.setattr(data_mod, "_fetch_risk_free_annual", _flat_annual)

    series = get_risk_free(START, END, periods_per_year=252)
    expected_index = _business_days(START, END)
    assert list(series.index) == list(expected_index)
    assert series.name == "risk_free"
    # Per-period = (1 + 0.04)^(1/252) - 1, applied elementwise.
    expected_per_period = (1.0 + 0.04) ** (1.0 / 252) - 1.0
    assert np.allclose(series.to_numpy(), expected_per_period)
    # Sanity: deannualized daily rate is far below the annual rate.
    assert series.iloc[0] < 0.04


@pytest.mark.unit
def test_get_risk_free_respects_periods_per_year(monkeypatch: pytest.MonkeyPatch) -> None:
    """A different periods_per_year changes the per-period magnitude (line 325)."""

    def _flat_annual(start: date, end: date, index: pd.DatetimeIndex) -> pd.Series:
        return pd.Series(0.06, index=index, dtype="float64")

    monkeypatch.setattr(data_mod, "_fetch_risk_free_annual", _flat_annual)

    monthly = get_risk_free(START, END, periods_per_year=12)
    daily = get_risk_free(START, END, periods_per_year=252)
    # Compounding 12x should give a larger per-period rate than 252x.
    assert monthly.iloc[0] > daily.iloc[0]
    expected_monthly = (1.06) ** (1.0 / 12) - 1.0
    assert abs(monthly.iloc[0] - expected_monthly) < 1e-15
