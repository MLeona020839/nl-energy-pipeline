import numpy as np
import pandas as pd
import pytest

from nl_energy_pipeline.features.transform import (
    FEATURE_COLUMNS,
    TARGET,
    build_features,
    clean_load,
    clean_weather,
    to_hourly,
)
from tests.conftest import synthetic_market


def test_to_hourly_averages_quarter_hours():
    idx = pd.date_range("2025-11-01", periods=8, freq="15min", tz="UTC")
    df = pd.DataFrame({"price_eur_mwh": [10, 20, 30, 40, 1, 1, 1, 1]}, index=idx)

    hourly = to_hourly(df)

    assert list(hourly["price_eur_mwh"]) == [25.0, 1.0]
    assert hourly.index.freqstr == "h"


def test_to_hourly_makes_gaps_explicit_and_drops_duplicates():
    idx = pd.DatetimeIndex(["2024-01-01 00:00", "2024-01-01 00:00", "2024-01-01 03:00"], tz="UTC")
    df = pd.DataFrame({"x": [1.0, 5.0, 9.0]}, index=idx)  # duplicate hour 0, nothing for 01-02

    hourly = to_hourly(df)

    assert len(hourly) == 4
    assert hourly["x"].iloc[0] == 5.0  # last duplicate wins
    assert hourly["x"].iloc[1:3].isna().all()


def test_clean_load_and_weather_null_out_impossible_values():
    load = pd.DataFrame({"load_mw": [12000.0, 0.0, -5.0]})
    assert clean_load(load)["load_mw"].isna().tolist() == [False, True, True]

    weather = pd.DataFrame({"temperature_2m": [10.0, 99.0], "wind_speed_10m": [5.0, -3.0]})
    cleaned = clean_weather(weather)
    assert cleaned["temperature_2m"].isna().tolist() == [False, True]
    assert cleaned["wind_speed_10m"].isna().tolist() == [False, True]


def test_lags_look_back_the_stated_number_of_hours():
    prices, load, weather = synthetic_market(days=20)
    frame = build_features(prices, load, weather)

    t = frame.index[300]
    assert frame.loc[t, "price_lag_24"] == prices.loc[t - pd.Timedelta(hours=24), TARGET]
    assert frame.loc[t, "price_lag_168"] == prices.loc[t - pd.Timedelta(hours=168), TARGET]
    assert frame.loc[t, "load_lag_48"] == load.loc[t - pd.Timedelta(hours=48), "load_mw"]
    assert frame.loc[t, "price_roll_mean_24"] == pytest.approx(
        prices[TARGET].loc[t - pd.Timedelta(hours=47) : t - pd.Timedelta(hours=24)].mean()
    )


def test_features_never_use_the_target_hour_or_anything_after_it():
    prices, load, weather = synthetic_market(days=20)
    baseline = build_features(prices, load, weather)

    t = baseline.index[300]
    tampered_prices = prices.copy()
    tampered_prices.loc[tampered_prices.index >= t, TARGET] += 1000
    tampered_load = load.copy()
    tampered_load.loc[tampered_load.index >= t - pd.Timedelta(hours=47), "load_mw"] += 1000

    tampered = build_features(tampered_prices, tampered_load, weather)

    pd.testing.assert_series_equal(
        baseline.loc[t, FEATURE_COLUMNS], tampered.loc[t, FEATURE_COLUMNS], check_names=False
    )


def test_calendar_features_use_local_dst_aware_time():
    idx = pd.DatetimeIndex(
        ["2024-01-01 23:00", "2024-07-01 12:00", "2024-12-25 12:00"], tz="UTC", name="ts"
    )
    prices = pd.DataFrame({TARGET: [1.0, 2.0, 3.0]}, index=idx)
    empty_load = pd.DataFrame({"load_mw": [np.nan] * 3}, index=idx)
    empty_weather = pd.DataFrame(
        {"temperature_2m": [np.nan] * 3, "wind_speed_10m": [np.nan] * 3}, index=idx
    )

    frame = build_features(prices, empty_load, empty_weather).loc[
        idx
    ]  # grid is continuous; pick our rows

    assert frame["hour"].tolist() == [0, 14, 13]  # CET, CEST, CET
    assert frame["dayofweek"].iloc[0] == 1  # Tuesday 2 Jan in Amsterdam, still Monday in UTC
    assert frame["is_holiday"].tolist() == [0, 0, 1]  # Christmas Day
