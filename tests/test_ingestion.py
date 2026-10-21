import pandas as pd
import pytest
import requests
from entsoe.exceptions import NoMatchingDataError

from nl_energy_pipeline.db import models
from nl_energy_pipeline.db.repository import read_frame
from nl_energy_pipeline.ingestion import entsoe, open_meteo
from nl_energy_pipeline.ingestion.run import ingest, month_chunks
from nl_energy_pipeline.ingestion.storage import save_raw
from tests.conftest import hourly_index

AMS = "Europe/Amsterdam"


class FakeClient:
    def __init__(self, prices=None, load=None, error=None):
        self._prices, self._load, self._error = prices, load, error

    def query_day_ahead_prices(self, area, start, end):
        if self._error:
            raise self._error
        return self._prices

    def query_load(self, area, start, end):
        if self._error:
            raise self._error
        return self._load


def test_fetch_prices_returns_utc_frame_with_stable_column_name():
    idx = pd.date_range("2024-01-01", periods=3, freq="1h", tz=AMS)
    client = FakeClient(prices=pd.Series([1.0, 2.0, 3.0], index=idx))

    df = entsoe.fetch_day_ahead_prices(client, idx[0], idx[-1])

    assert list(df.columns) == ["price_eur_mwh"]
    assert str(df.index.tz) == "UTC"
    assert df.index[0] == pd.Timestamp("2023-12-31 23:00", tz="UTC")


def test_fetch_load_renames_entsoe_column():
    idx = pd.date_range("2024-01-01", periods=2, freq="15min", tz=AMS)
    client = FakeClient(load=pd.DataFrame({"Actual Load": [11000.0, 11100.0]}, index=idx))

    df = entsoe.fetch_load(client, idx[0], idx[-1])

    assert list(df.columns) == ["load_mw"]
    assert str(df.index.tz) == "UTC"


def test_no_matching_data_gives_empty_frame_but_http_errors_propagate():
    ts = pd.Timestamp("2030-01-01", tz="UTC")
    empty = entsoe.fetch_day_ahead_prices(FakeClient(error=NoMatchingDataError()), ts, ts)
    assert empty.empty and list(empty.columns) == ["price_eur_mwh"]

    with pytest.raises(requests.HTTPError):
        entsoe.fetch_load(FakeClient(error=requests.HTTPError("401")), ts, ts)


def test_get_client_requires_a_token(monkeypatch):
    monkeypatch.setattr(entsoe.settings, "entsoe_api_token", None)
    with pytest.raises(RuntimeError, match="ENTSOE_API_TOKEN"):
        entsoe.get_client()


def test_fetch_weather_builds_utc_indexed_frame(monkeypatch):
    class Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "hourly": {
                    "time": ["2024-01-01T00:00", "2024-01-01T01:00"],
                    "temperature_2m": [7.6, 7.2],
                    "wind_speed_10m": [32.2, None],
                }
            }

    class Session:
        def get(self, url, params, timeout):
            assert params["start_date"] == "2024-01-01" and timeout
            return Resp()

    monkeypatch.setattr(open_meteo, "_session", lambda: Session())

    df = open_meteo.fetch_weather(
        pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-01-02", tz="UTC")
    )

    assert str(df.index.tz) == "UTC"
    assert df["wind_speed_10m"].isna().tolist() == [False, True]


def test_month_chunks_split_on_month_boundaries():
    chunks = month_chunks(
        pd.Timestamp("2024-01-15", tz="UTC"), pd.Timestamp("2024-03-10", tz="UTC")
    )

    assert [(s.strftime("%m-%d"), e.strftime("%m-%d")) for s, e in chunks] == [
        ("01-15", "02-01"),
        ("02-01", "03-01"),
        ("03-01", "03-10"),
    ]


def test_save_raw_names_file_by_range_and_skips_empty(tmp_path):
    df = pd.DataFrame({"x": [1.0, 2.0]}, index=hourly_index("2024-01-01", 2))

    path = save_raw(df, "demo", tmp_path / "nested")

    assert path.name == "demo_2024-01-01_2024-01-01.parquet"
    assert pd.read_parquet(path).equals(df)
    assert save_raw(df.iloc[0:0], "demo", tmp_path) is None


def test_ingest_loads_every_source_and_rerunning_changes_nothing(db, monkeypatch, tmp_path):
    idx = hourly_index("2024-01-01", 48)
    monkeypatch.setattr(
        entsoe,
        "fetch_day_ahead_prices",
        lambda c, s, e: pd.DataFrame({"price_eur_mwh": 1.0}, index=idx),
    )
    monkeypatch.setattr(
        entsoe, "fetch_load", lambda c, s, e: pd.DataFrame({"load_mw": 2.0}, index=idx)
    )
    monkeypatch.setattr(
        open_meteo,
        "fetch_weather",
        lambda s, e: pd.DataFrame({"temperature_2m": 3.0, "wind_speed_10m": 4.0}, index=idx),
    )
    monkeypatch.setattr(entsoe, "RAW_DIR", tmp_path / "entsoe")
    monkeypatch.setattr(open_meteo, "RAW_DIR", tmp_path / "weather")

    start, end = pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-01-03", tz="UTC")
    first = ingest(start, end, db, client=object())
    second = ingest(start, end, db, client=object())

    assert first == second == {"day_ahead_prices": 48, "load": 48, "weather": 48}
    assert len(read_frame(db, models.DayAheadPrice)) == 48
    assert len(read_frame(db, models.Weather)) == 48
    assert list((tmp_path / "entsoe").glob("*.parquet"))


def test_ingest_skips_sources_without_data(db, monkeypatch):
    empty = pd.DataFrame(columns=["price_eur_mwh"])
    monkeypatch.setattr(entsoe, "fetch_day_ahead_prices", lambda c, s, e: empty)
    monkeypatch.setattr(entsoe, "fetch_load", lambda c, s, e: pd.DataFrame(columns=["load_mw"]))
    monkeypatch.setattr(
        open_meteo,
        "fetch_weather",
        lambda s, e: pd.DataFrame(columns=["temperature_2m", "wind_speed_10m"]),
    )

    totals = ingest(
        pd.Timestamp("2024-01-01", tz="UTC"),
        pd.Timestamp("2024-01-02", tz="UTC"),
        db,
        client=object(),
        save_parquet=False,
    )

    assert totals == {"day_ahead_prices": 0, "load": 0, "weather": 0}


def http_error(status: int) -> requests.HTTPError:
    response = requests.Response()
    response.status_code = status
    return requests.HTTPError(
        f"{status} Error for url: https://x/api?securityToken=SECRET123&a=1", response=response
    )


def test_redact_hides_token_only():
    from nl_energy_pipeline.ingestion.resilience import redact

    assert redact("u?a=1&securityToken=SECRET123&b=2") == "u?a=1&securityToken=***&b=2"


def test_transient_errors_are_retried_with_backoff_then_succeed():
    from nl_energy_pipeline.ingestion.resilience import call_with_retries

    calls, delays = [], []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise http_error(599)
        return "ok"

    assert call_with_retries(flaky, sleep=delays.append) == "ok"
    assert delays == [5.0, 10.0]


def test_client_errors_fail_immediately_and_never_leak_the_token():
    from nl_energy_pipeline.ingestion.resilience import call_with_retries

    calls = []

    def unauthorized():
        calls.append(1)
        raise http_error(401)

    with pytest.raises(requests.HTTPError) as exc:
        call_with_retries(unauthorized, sleep=lambda s: None)

    assert len(calls) == 1
    assert "SECRET123" not in str(exc.value)
    assert exc.value.response.status_code == 401


def test_exhausted_retries_raise_a_redacted_error():
    from nl_energy_pipeline.ingestion.resilience import call_with_retries

    calls = []

    def always_down():
        calls.append(1)
        raise http_error(503)

    with pytest.raises(requests.HTTPError) as exc:
        call_with_retries(always_down, attempts=3, sleep=lambda s: None)

    assert len(calls) == 3
    assert "SECRET123" not in str(exc.value)
