import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from alembic.config import Config
from sqlalchemy import Engine, text

from alembic import command
from nl_energy_pipeline.db.session import get_engine

ROOT = Path(__file__).resolve().parents[1]
TABLES = ["day_ahead_prices", "load", "weather", "price_predictions", "model_runs"]


@pytest.fixture(scope="session")
def engine() -> Engine:
    """A real Postgres with the schema built by the Alembic migrations.

    CI provides TEST_DATABASE_URL (service container); locally pgserver starts a private instance.
    """
    url = os.environ.get("TEST_DATABASE_URL")
    server = None
    if not url:
        import pgserver

        server = pgserver.get_server(tempfile.mkdtemp(prefix="nlep-test-"), cleanup_mode="delete")
        host = server.get_uri().split("host=")[1]
        url = f"postgresql+psycopg://postgres:@/postgres?host={host}"

    engine = get_engine(url)
    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "alembic"))
    with engine.begin() as conn:
        for table in [*TABLES, "alembic_version"]:
            conn.execute(text(f"DROP TABLE IF EXISTS {table} CASCADE"))
        cfg.attributes["connection"] = conn
        command.upgrade(cfg, "head")

    yield engine
    engine.dispose()
    if server is not None:
        server.cleanup()


@pytest.fixture
def db(engine: Engine) -> Engine:
    """Engine with all tables emptied, so tests do not see each other's rows."""
    with engine.begin() as conn:
        conn.execute(text(f"TRUNCATE {', '.join(TABLES)}"))
    return engine


def hourly_index(start: str, periods: int) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=periods, freq="1h", tz="UTC", name="ts")


def synthetic_market(
    days: int = 400, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Prices with daily + weekly seasonality driven by load and wind, plus noise."""
    rng = np.random.default_rng(seed)
    idx = hourly_index("2024-01-01", days * 24)
    hour = np.asarray(idx.hour)
    weekend = np.asarray(idx.dayofweek >= 5)
    wind = np.clip(20 + 10 * np.sin(np.arange(len(idx)) / 90) + rng.normal(0, 4, len(idx)), 0, None)
    temp = 10 + 8 * np.sin(np.arange(len(idx)) / (24 * 58)) + rng.normal(0, 1, len(idx))
    load = (
        12000
        + 2500 * np.sin((hour - 6) / 24 * 2 * np.pi)
        - 1500 * weekend
        + rng.normal(0, 150, len(idx))
    )
    price = 60 + 0.004 * (load - 12000) - 1.2 * wind + 8 * weekend + rng.normal(0, 3, len(idx))
    prices = pd.DataFrame({"price_eur_mwh": price}, index=idx)
    loads = pd.DataFrame({"load_mw": load}, index=idx)
    weather = pd.DataFrame({"temperature_2m": temp, "wind_speed_10m": wind}, index=idx)
    return prices, loads, weather
