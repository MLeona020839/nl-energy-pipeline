import numpy as np
import pandas as pd
import pytest
from sqlalchemy import inspect, text

from nl_energy_pipeline.db import models
from nl_energy_pipeline.db.repository import read_frame, upsert_frame
from nl_energy_pipeline.db.session import session_scope
from tests.conftest import hourly_index


def prices(values, start="2024-01-01") -> pd.DataFrame:
    return pd.DataFrame({"price_eur_mwh": values}, index=hourly_index(start, len(values)))


def count(engine, table: str) -> int:
    with engine.connect() as conn:
        return conn.execute(text(f"SELECT count(*) FROM {table}")).scalar()


def test_migrations_create_expected_schema(engine):
    inspector = inspect(engine)
    assert {"day_ahead_prices", "load", "weather", "price_predictions", "model_runs"} <= set(
        inspector.get_table_names()
    )
    assert "ix_price_predictions_anomalies" in [
        i["name"] for i in inspector.get_indexes("price_predictions")
    ]


def test_upsert_is_idempotent(db):
    df = prices([10.0, 20.0, 30.0])
    for _ in range(3):
        with session_scope(db) as session:
            upsert_frame(session, models.DayAheadPrice, df)

    assert count(db, "day_ahead_prices") == 3


def test_upsert_overwrites_changed_values_and_fills_nulls(db):
    weather = pd.DataFrame(
        {"temperature_2m": [5.0, np.nan], "wind_speed_10m": [3.0, np.nan]},
        index=hourly_index("2024-01-01", 2),
    )
    with session_scope(db) as session:
        upsert_frame(session, models.Weather, weather)
    assert read_frame(db, models.Weather)["temperature_2m"].isna().tolist() == [False, True]

    revised = weather.copy()
    revised.loc[:, :] = [[6.0, 3.5], [7.0, 4.5]]  # archive caught up with the missing hour
    with session_scope(db) as session:
        upsert_frame(session, models.Weather, revised)

    stored = read_frame(db, models.Weather)
    assert stored["temperature_2m"].tolist() == [6.0, 7.0]
    assert count(db, "weather") == 2


def test_duplicate_timestamps_in_one_batch_do_not_crash(db):
    df = prices([1.0, 2.0, 3.0])
    df.index = df.index[[0, 0, 1]]  # two rows for the same hour (e.g. a DST fold)

    with session_scope(db) as session:
        upsert_frame(session, models.DayAheadPrice, df)

    assert read_frame(db, models.DayAheadPrice)["price_eur_mwh"].tolist() == [2.0, 3.0]


def test_empty_frame_is_a_noop_and_unknown_columns_are_rejected(db):
    with session_scope(db) as session:
        assert upsert_frame(session, models.DayAheadPrice, prices([])) == 0
        with pytest.raises(ValueError, match="not on day_ahead_prices"):
            upsert_frame(
                session,
                models.DayAheadPrice,
                prices([1.0]).rename(columns={"price_eur_mwh": "oops"}),
            )


def test_read_frame_filters_range_and_returns_utc_index(db):
    with session_scope(db) as session:
        upsert_frame(session, models.DayAheadPrice, prices(range(10)))

    out = read_frame(
        db,
        models.DayAheadPrice,
        pd.Timestamp("2024-01-01 03:00", tz="UTC"),
        pd.Timestamp("2024-01-01 06:00", tz="UTC"),
    )

    assert out["price_eur_mwh"].tolist() == [3.0, 4.0, 5.0]
    assert str(out.index.tz) == "UTC"
    assert "ingested_at" not in out.columns
