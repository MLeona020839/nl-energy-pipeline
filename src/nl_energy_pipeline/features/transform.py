"""Cleaning and feature engineering as pure functions (DataFrame in, DataFrame out).

Feature availability contract: the model forecasts all 24 hours of day D at ~10:00 UTC on
D-1. Everything used as an input must be known by then:
  * prices: the whole of day D-1 is already published (day-ahead), so lags >= 24h are fine;
  * actual load: only known up to the forecast time, so lags must be >= 48h;
  * weather at the target hour: assumed to be available as a weather *forecast*
    (backtests use observed weather as a stand-in for a perfect forecast).
"""

import holidays
import pandas as pd

LOCAL_TZ = "Europe/Amsterdam"
TARGET = "price_eur_mwh"

FEATURE_COLUMNS = [
    "price_lag_24",
    "price_lag_48",
    "price_lag_168",
    "price_roll_mean_24",
    "load_lag_48",
    "load_lag_168",
    "temperature_2m",
    "wind_speed_10m",
    "hour",
    "dayofweek",
    "month",
    "is_weekend",
    "is_holiday",
]


def to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Sort, drop duplicate timestamps, and average to a complete hourly UTC grid.

    ENTSO-E mixes 15-minute and hourly resolution over time; reindexing to a full grid
    turns gaps into explicit NaN so that `shift(24)` really means "24 hours earlier".
    """
    if df.empty:
        return df
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    hourly = df.resample("1h").mean()
    full_index = pd.date_range(hourly.index.min(), hourly.index.max(), freq="1h")
    return hourly.reindex(full_index).rename_axis("ts")


def clean_load(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.loc[out["load_mw"] <= 0, "load_mw"] = pd.NA  # a non-positive national load is a data error
    return out.astype("float64")


def clean_weather(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.loc[~out["temperature_2m"].between(-40, 50), "temperature_2m"] = pd.NA
    out.loc[~out["wind_speed_10m"].between(0, 200), "wind_speed_10m"] = pd.NA
    return out.astype("float64")


def build_features(prices: pd.DataFrame, load: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Hourly feature matrix + target, indexed by UTC timestamp. Rows with any NaN are kept
    (callers drop them) so that the grid stays regular."""
    price = to_hourly(prices)[TARGET]
    load_mw = clean_load(to_hourly(load))["load_mw"]
    weather_h = clean_weather(to_hourly(weather))

    frame = pd.DataFrame({TARGET: price})
    frame["price_lag_24"] = price.shift(24)
    frame["price_lag_48"] = price.shift(48)
    frame["price_lag_168"] = price.shift(168)
    frame["price_roll_mean_24"] = price.shift(24).rolling(24).mean()
    frame["load_lag_48"] = load_mw.reindex(frame.index).shift(48)
    frame["load_lag_168"] = load_mw.reindex(frame.index).shift(168)
    frame["temperature_2m"] = weather_h["temperature_2m"].reindex(frame.index)
    frame["wind_speed_10m"] = weather_h["wind_speed_10m"].reindex(frame.index)

    # Calendar effects follow local (DST-aware) time, not UTC.
    local = frame.index.tz_convert(LOCAL_TZ)
    nl_holidays = holidays.country_holidays(
        "NL", years=range(local.year.min(), local.year.max() + 1)
    )
    frame["hour"] = local.hour
    frame["dayofweek"] = local.dayofweek
    frame["month"] = local.month
    frame["is_weekend"] = (local.dayofweek >= 5).astype(int)
    frame["is_holiday"] = [int(d in nl_holidays) for d in local.date]
    return frame
