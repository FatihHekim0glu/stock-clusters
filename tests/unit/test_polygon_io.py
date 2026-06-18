"""Network-free unit tests for the Polygon.io EOD provider I/O paths.

These tests exercise the parts of ``stockclusters.data_providers.polygon`` that
the existing ``test_data_layer.py`` does not reach: the ``.env`` parser, the
httpx and urllib HTTP transports (both lazily imported inside the functions and
fully monkeypatched), and the end-to-end ``fetch`` assembly. Every HTTP call is
faked, so the suite is deterministic and touches no network.

The lazily imported network libraries (``httpx`` and ``urllib.request``) are
imported *inside* the methods under test, so we monkeypatch the modules in
``sys.modules`` / on the real module objects rather than at import time.
"""

from __future__ import annotations

import sys
import time
import types
import urllib.error
import urllib.request
from datetime import date
from typing import Any

import pandas as pd
import pytest

from stockclusters._exceptions import ValidationError
from stockclusters.data_providers import polygon as pg
from stockclusters.data_providers.polygon import (
    PolygonProvider,
    _load_api_key_from_dotenv,
    _resolve_api_key,
)

# --------------------------------------------------------------------------- #
# Helpers                                                                       #
# --------------------------------------------------------------------------- #

# Two real epoch-ms timestamps (UTC midnight) used across the HTTP tests.
_T0 = 1577836800000  # 2020-01-01
_T1 = 1577923200000  # 2020-01-02


def _ok_payload() -> dict[str, Any]:
    """A well-formed two-bar Polygon aggregates payload."""
    return {
        "status": "OK",
        "results": [
            {"t": _T0, "c": 100.0},
            {"t": _T1, "c": 101.5},
        ],
    }


class _FakeHTTPXResponse:
    """Minimal stand-in for ``httpx.Response`` used by the fake client."""

    def __init__(
        self,
        status_code: int,
        payload: dict[str, Any] | None = None,
        *,
        raise_for_status_exc: Exception | None = None,
    ) -> None:
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self._raise_for_status_exc = raise_for_status_exc

    def json(self) -> dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        if self._raise_for_status_exc is not None:
            raise self._raise_for_status_exc


class _FakeHTTPXClient:
    """Context-manager client that returns queued responses for each GET.

    ``responses`` is consumed left-to-right; the last entry is reused if more
    GETs occur than responses provided. ``calls`` records every URL requested.
    """

    def __init__(self, responses: list[_FakeHTTPXResponse]) -> None:
        self._responses = responses
        self._idx = 0
        self.calls: list[str] = []

    def __enter__(self) -> _FakeHTTPXClient:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def get(self, url: str) -> _FakeHTTPXResponse:
        self.calls.append(url)
        resp = self._responses[min(self._idx, len(self._responses) - 1)]
        self._idx += 1
        return resp


def _install_fake_httpx(
    monkeypatch: pytest.MonkeyPatch,
    client: _FakeHTTPXClient,
) -> types.ModuleType:
    """Install a fake ``httpx`` module exposing ``Client`` and ``HTTPStatusError``.

    ``_get_json`` does ``import httpx`` lazily, so placing a module object in
    ``sys.modules['httpx']`` makes that import resolve to our fake.
    """
    fake = types.ModuleType("httpx")
    fake.Client = lambda *args, **kwargs: client  # type: ignore[attr-defined]

    class _HTTPStatusError(Exception):
        pass

    fake.HTTPStatusError = _HTTPStatusError  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "httpx", fake)
    return fake


# --------------------------------------------------------------------------- #
# _load_api_key_from_dotenv  (lines 52-70)                                       #
# --------------------------------------------------------------------------- #


def _point_dotenv_search_at(monkeypatch: pytest.MonkeyPatch, root: Any) -> None:
    """Make ``_load_api_key_from_dotenv`` walk ``root`` instead of the repo tree.

    The function uses ``Path(__file__).resolve().parents``; we monkeypatch the
    module's ``Path`` so ``Path(__file__)`` yields a fake whose single parent is
    the supplied ``root`` directory.
    """
    from pathlib import Path as _RealPath

    class _FakePath:
        def __init__(self, *_: object) -> None:
            pass

        def resolve(self) -> _FakePath:
            return self

        @property
        def parents(self) -> list[Any]:
            return [_RealPath(str(root))]

    monkeypatch.setattr(pg, "Path", _FakePath, raising=True)


@pytest.mark.unit
def test_load_dotenv_parses_present_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    (tmp_path / ".env").write_text(
        "# a comment\n\nOTHER=ignored\nexport POLYGON_API_KEY='dotenv-secret'\n",
        encoding="utf-8",
    )
    _point_dotenv_search_at(monkeypatch, tmp_path)
    assert _load_api_key_from_dotenv() == "dotenv-secret"


@pytest.mark.unit
def test_load_dotenv_strips_quotes_and_no_export(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    # No 'export' prefix and double quotes -> still parsed, quotes stripped.
    (tmp_path / ".env").write_text('POLYGON_API_KEY="quoted-key"\n', encoding="utf-8")
    _point_dotenv_search_at(monkeypatch, tmp_path)
    assert _load_api_key_from_dotenv() == "quoted-key"


@pytest.mark.unit
def test_load_dotenv_present_file_missing_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    # A .env exists but has no POLYGON_API_KEY: stop at the first .env, return None.
    (tmp_path / ".env").write_text(
        "# only junk\nFOO=bar\nmalformed-line-without-equals\n", encoding="utf-8"
    )
    _point_dotenv_search_at(monkeypatch, tmp_path)
    assert _load_api_key_from_dotenv() is None


@pytest.mark.unit
def test_load_dotenv_no_file_anywhere(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    # No .env in any searched parent -> None (the final return).
    _point_dotenv_search_at(monkeypatch, tmp_path / "empty_subdir")
    assert _load_api_key_from_dotenv() is None


@pytest.mark.unit
def test_load_dotenv_unreadable_returns_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    # An OSError while reading the .env is swallowed and yields None (lines 66-67).
    (tmp_path / ".env").write_text("POLYGON_API_KEY=whatever\n", encoding="utf-8")
    _point_dotenv_search_at(monkeypatch, tmp_path)

    def _boom(self: Any, *args: object, **kwargs: object) -> str:
        raise OSError("permission denied")

    # The fake parent resolves to a real pathlib.Path, so patching read_text on
    # the real class makes the candidate .env read raise OSError.
    from pathlib import Path as _RealPath

    monkeypatch.setattr(_RealPath, "read_text", _boom, raising=True)
    assert _load_api_key_from_dotenv() is None


# --------------------------------------------------------------------------- #
# _resolve_api_key  (explicit / env / dotenv branch at line 82, missing-key)     #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_resolve_api_key_explicit_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLYGON_API_KEY", "env-key")
    monkeypatch.setattr(pg, "_load_api_key_from_dotenv", lambda: "dotenv-key")
    assert _resolve_api_key("explicit-key") == "explicit-key"


@pytest.mark.unit
def test_resolve_api_key_env_used(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLYGON_API_KEY", "env-key")
    monkeypatch.setattr(pg, "_load_api_key_from_dotenv", lambda: "dotenv-key")
    # Env beats .env when no explicit key.
    assert _resolve_api_key(None) == "env-key"


@pytest.mark.unit
def test_resolve_api_key_falls_through_to_dotenv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # No explicit, no env -> the dotenv branch (line 82) is taken and returned.
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    monkeypatch.setattr(pg, "_load_api_key_from_dotenv", lambda: "from-dotenv")
    assert _resolve_api_key(None) == "from-dotenv"


@pytest.mark.unit
def test_resolve_api_key_missing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    monkeypatch.setattr(pg, "_load_api_key_from_dotenv", lambda: None)
    with pytest.raises(ValidationError, match="no API key"):
        _resolve_api_key(None)


# --------------------------------------------------------------------------- #
# _series_from_payload  (well-formed / empty / missing / error)                  #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_series_from_payload_sorts_and_dedupes() -> None:
    # Out-of-order bars plus a duplicate timestamp; parser must produce a clean,
    # date-indexed float Series keeping the *last* bar for the duplicated day.
    payload = {
        "status": "OK",
        "results": [
            {"t": _T1, "c": 101.5},  # 2020-01-02
            {"t": _T0, "c": 100.0},  # 2020-01-01
            {"t": _T1, "c": 999.0},  # 2020-01-02 duplicate -> keep this one
        ],
    }
    series = PolygonProvider._series_from_payload(payload, "AAA")
    assert series.name == "AAA"
    assert str(series.dtype) == "float64"
    # Duplicate day collapses to the last occurrence (999.0); two unique days.
    assert series.loc[pd.Timestamp("2020-01-02")] == 999.0
    assert series.loc[pd.Timestamp("2020-01-01")] == 100.0
    assert len(series) == 2
    # Index entries are normalized to midnight.
    assert all(ts == ts.normalize() for ts in series.index)


@pytest.mark.unit
def test_series_from_payload_missing_results_key_raises() -> None:
    # No "results" key at all -> `payload.get("results") or []` -> empty -> raises.
    with pytest.raises(ValueError, match="no results"):
        PolygonProvider._series_from_payload({"status": "OK"}, "BBB")


@pytest.mark.unit
def test_series_from_payload_error_status_raises_with_status() -> None:
    # An error payload (results None, status != OK): the raised message surfaces
    # the status for debuggability.
    payload = {"status": "ERROR", "error": "bad key", "results": None}
    with pytest.raises(ValueError, match="ERROR"):
        PolygonProvider._series_from_payload(payload, "CCC")


@pytest.mark.unit
def test_series_from_payload_empty_results_raises() -> None:
    with pytest.raises(ValueError, match="no results"):
        PolygonProvider._series_from_payload({"status": "OK", "results": []}, "DDD")


# --------------------------------------------------------------------------- #
# _get_json via httpx (lines 188-210)                                            #
# --------------------------------------------------------------------------- #


def _make_provider(monkeypatch: pytest.MonkeyPatch, **kwargs: Any) -> PolygonProvider:
    monkeypatch.setenv("POLYGON_API_KEY", "test-key")
    return PolygonProvider(**kwargs)


@pytest.mark.unit
def test_get_json_httpx_success(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _make_provider(monkeypatch)
    client = _FakeHTTPXClient([_FakeHTTPXResponse(200, _ok_payload())])
    _install_fake_httpx(monkeypatch, client)

    data = provider._get_json("AAPL", date(2020, 1, 1), date(2020, 1, 5))
    assert data["status"] == "OK"
    assert len(data["results"]) == 2
    # The request URL carried the resolved key and the ticker.
    assert client.calls and "AAPL" in client.calls[0]
    assert "test-key" in client.calls[0]


@pytest.mark.unit
def test_get_json_httpx_429_retries_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # First response is 429, second is a 200 success: the retry loop must sleep,
    # retry, and return the eventual payload.
    provider = _make_provider(monkeypatch, max_retries=2, backoff_base=0.01)
    client = _FakeHTTPXClient(
        [
            _FakeHTTPXResponse(429),
            _FakeHTTPXResponse(200, _ok_payload()),
        ]
    )
    _install_fake_httpx(monkeypatch, client)

    slept: list[float] = []
    fake_time = types.ModuleType("time")
    fake_time.sleep = lambda s: slept.append(s)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "time", fake_time)

    data = provider._get_json("MSFT", date(2020, 1, 1), date(2020, 1, 5))
    assert data["status"] == "OK"
    assert len(client.calls) == 2  # one 429 + one success
    assert slept == [0.01]  # backoff_base * 2**0 on the first retry


@pytest.mark.unit
def test_get_json_httpx_429_exhausts_retries_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Always 429: after max_retries the loop breaks and a RuntimeError is raised
    # carrying the last status (lines 202-206, 210-212).
    provider = _make_provider(monkeypatch, max_retries=2, backoff_base=0.01)
    client = _FakeHTTPXClient([_FakeHTTPXResponse(429)])
    _install_fake_httpx(monkeypatch, client)

    fake_time = types.ModuleType("time")
    fake_time.sleep = lambda s: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "time", fake_time)

    with pytest.raises(RuntimeError, match="HTTP 429"):
        provider._get_json("TSLA", date(2020, 1, 1), date(2020, 1, 5))
    # max_retries + 1 total attempts were made.
    assert len(client.calls) == 3


@pytest.mark.unit
def test_get_json_httpx_non_429_error_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A 500 (non-retryable) must propagate via raise_for_status immediately.
    boom = RuntimeError("server error 500")
    provider = _make_provider(monkeypatch)
    client = _FakeHTTPXClient([_FakeHTTPXResponse(500, raise_for_status_exc=boom)])
    _install_fake_httpx(monkeypatch, client)

    with pytest.raises(RuntimeError, match="server error 500"):
        provider._get_json("NVDA", date(2020, 1, 1), date(2020, 1, 5))
    assert len(client.calls) == 1  # no retry on a non-429 error


# --------------------------------------------------------------------------- #
# _get_json_urllib fallback (lines 216-233) - httpx forced ABSENT               #
# --------------------------------------------------------------------------- #


class _FakeURLResponse:
    """Context-manager stand-in for the object returned by ``urlopen``."""

    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self) -> _FakeURLResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body


def _force_httpx_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``import httpx`` raise ImportError so the urllib fallback runs."""
    monkeypatch.setitem(sys.modules, "httpx", None)


@pytest.mark.unit
def test_get_json_falls_back_to_urllib_when_httpx_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _make_provider(monkeypatch)
    _force_httpx_import_error(monkeypatch)

    body = b'{"status": "OK", "results": [{"t": 1577836800000, "c": 42.0}]}'

    captured: dict[str, Any] = {}

    def _fake_urlopen(url: str, timeout: float | None = None) -> _FakeURLResponse:
        captured["url"] = url
        captured["timeout"] = timeout
        return _FakeURLResponse(body)

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)

    data = provider._get_json("AAPL", date(2020, 1, 1), date(2020, 1, 5))
    assert data == {"status": "OK", "results": [{"t": 1577836800000, "c": 42.0}]}
    assert "AAPL" in captured["url"]
    assert captured["timeout"] == provider.timeout


@pytest.mark.unit
def test_get_json_urllib_429_retries_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # urllib path: first urlopen raises HTTP 429, second succeeds.
    provider = _make_provider(monkeypatch, max_retries=2, backoff_base=0.01)

    calls = {"n": 0}
    body = b'{"status": "OK", "results": [{"t": 1577836800000, "c": 7.0}]}'

    def _fake_urlopen(url: str, timeout: float | None = None) -> _FakeURLResponse:
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.HTTPError(url, 429, "Too Many Requests", {}, None)  # type: ignore[arg-type]
        return _FakeURLResponse(body)

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)

    slept: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda s: slept.append(s), raising=True)

    data = provider._get_json_urllib(provider._url("AAPL", date(2020, 1, 1), date(2020, 1, 5)))
    assert data["status"] == "OK"
    assert calls["n"] == 2
    assert slept == [0.01]


@pytest.mark.unit
def test_get_json_urllib_non_429_error_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A non-429 HTTPError (e.g. 500) re-raises immediately without retrying.
    provider = _make_provider(monkeypatch, max_retries=2, backoff_base=0.01)

    calls = {"n": 0}

    def _fake_urlopen(url: str, timeout: float | None = None) -> _FakeURLResponse:
        calls["n"] += 1
        raise urllib.error.HTTPError(url, 500, "Server Error", {}, None)  # type: ignore[arg-type]

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)

    with pytest.raises(urllib.error.HTTPError):
        provider._get_json_urllib(provider._url("AAPL", date(2020, 1, 1), date(2020, 1, 5)))
    assert calls["n"] == 1  # no retry on a non-429 error


@pytest.mark.unit
def test_get_json_urllib_429_exhausts_retries_raises_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Persistent 429s: after the retry budget the final HTTPError propagates
    # (the loop re-raises on the last attempt rather than reaching line 233).
    provider = _make_provider(monkeypatch, max_retries=1, backoff_base=0.01)

    def _fake_urlopen(url: str, timeout: float | None = None) -> _FakeURLResponse:
        raise urllib.error.HTTPError(url, 429, "Too Many Requests", {}, None)  # type: ignore[arg-type]

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    monkeypatch.setattr(time, "sleep", lambda s: None, raising=True)

    with pytest.raises(urllib.error.HTTPError):
        provider._get_json_urllib(provider._url("AAPL", date(2020, 1, 1), date(2020, 1, 5)))


# --------------------------------------------------------------------------- #
# fetch end-to-end (with _get_json monkeypatched)                               #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_fetch_assembles_multi_ticker_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _make_provider(monkeypatch)

    def _fake_get_json(ticker: str, start: date, end: date) -> dict[str, Any]:
        prices = {"AAA": [10.0, 11.0, 12.0], "BBB": [20.0, 21.0, 22.0]}
        results = [
            {"t": t, "c": c} for t, c in zip([_T0, _T1, 1578009600000], prices[ticker], strict=True)
        ]
        return {"status": "OK", "results": results}

    monkeypatch.setattr(provider, "_get_json", _fake_get_json)
    frame = provider.fetch(["AAA", "BBB"], date(2020, 1, 1), date(2020, 1, 5))

    assert list(frame.columns) == ["AAA", "BBB"]
    assert len(frame) == 3
    assert str(frame.dtypes["AAA"]) == "float64"
    assert frame.index.is_monotonic_increasing
    # Values land in the right cells.
    assert frame.iloc[0]["AAA"] == 10.0
    assert frame.iloc[-1]["BBB"] == 22.0


@pytest.mark.unit
def test_fetch_inner_join_drops_unshared_dates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # AAA has 3 days, BBB only the first 2 -> inner-join keeps the 2 shared days.
    provider = _make_provider(monkeypatch)

    def _fake_get_json(ticker: str, start: date, end: date) -> dict[str, Any]:
        base = [{"t": _T0, "c": 1.0}, {"t": _T1, "c": 2.0}]
        if ticker == "AAA":
            base = [*base, {"t": 1578009600000, "c": 3.0}]
        return {"status": "OK", "results": base}

    monkeypatch.setattr(provider, "_get_json", _fake_get_json)
    frame = provider.fetch(["AAA", "BBB"], date(2020, 1, 1), date(2020, 1, 5))
    assert len(frame) == 2
    assert list(frame.columns) == ["AAA", "BBB"]
    assert frame.notna().all().all()  # no NaNs survive the inner join


@pytest.mark.unit
def test_fetch_propagates_empty_ticker_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # If any ticker yields an empty payload, _series_from_payload raises and the
    # error propagates out of fetch (the empty ticker is not silently dropped).
    provider = _make_provider(monkeypatch)

    def _fake_get_json(ticker: str, start: date, end: date) -> dict[str, Any]:
        if ticker == "BBB":
            return {"status": "OK", "results": []}
        return {"status": "OK", "results": [{"t": _T0, "c": 5.0}]}

    monkeypatch.setattr(provider, "_get_json", _fake_get_json)
    with pytest.raises(ValueError, match="no results"):
        provider.fetch(["AAA", "BBB"], date(2020, 1, 1), date(2020, 1, 5))
