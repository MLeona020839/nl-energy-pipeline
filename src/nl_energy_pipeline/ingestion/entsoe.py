from pathlib import Path

import pandas as pd
from entsoe import EntsoePandasClient
from entsoe.exceptions import NoMatchingDataError

from nl_energy_pipeline.config import settings

AREA_CODE = "NL"
RAW_DIR = Path("data/raw/entsoe")


def get_client() -> EntsoePandasClient:
    if not settings.entsoe_api_token:
        raise RuntimeError("ENTSOE_API_TOKEN is not set (put it in .env or the environment)")
    return EntsoePandasClient(api_key=settings.entsoe_api_token)


def fetch_day_ahead_prices(client, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Day-ahead prices (EUR/MWh) as a one-column frame `price_eur_mwh`, UTC index.

    NoMatchingDataError means "no data for this window" and yields an empty frame;
    HTTP/auth errors are deliberately not caught so a broken run fails loudly.
    """
    try:
        prices = client.query_day_ahead_prices(AREA_CODE, start=start, end=end)
    except NoMatchingDataError:
        return pd.DataFrame(columns=["price_eur_mwh"])

    df = prices.to_frame(name="price_eur_mwh")
    df.index = df.index.tz_convert("UTC")
    return df


def fetch_load(client, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Actual total load (MW, 15-minute resolution) as a one-column frame `load_mw`, UTC index."""
    try:
        actual_load = client.query_load(AREA_CODE, start=start, end=end)
    except NoMatchingDataError:
        return pd.DataFrame(columns=["load_mw"])

    df = actual_load.iloc[:, 0].rename("load_mw").to_frame()
    df.index = df.index.tz_convert("UTC")
    return df
