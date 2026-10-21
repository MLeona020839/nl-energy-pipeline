from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

LATITUDE = 52.37
LONGITUDE = 4.89
WEATHER_VARS = "temperature_2m,wind_speed_10m"
BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
RAW_DIR = Path("data/raw/open_meteo")


def _session() -> requests.Session:
    # Retry only transient failures (rate limit / 5xx); a 4xx caused by our params still raises.
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def fetch_weather(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Hourly temperature and wind speed for Amsterdam, UTC-indexed.

    The archive lags real time by a few days; the missing tail comes back as NaN and is
    overwritten on a later run (writes are upserts).
    """
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
        "hourly": WEATHER_VARS,
        "timezone": "UTC",
    }

    resp = _session().get(BASE_URL, params=params, timeout=30)
    resp.raise_for_status()

    weather = pd.DataFrame(resp.json()["hourly"])
    weather["time"] = pd.to_datetime(weather["time"]).dt.tz_localize("UTC")
    return weather.set_index("time")
