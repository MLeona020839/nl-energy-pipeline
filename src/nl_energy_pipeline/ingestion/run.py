import logging
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import pandas as pd
from sqlalchemy import Engine

from nl_energy_pipeline.db import models
from nl_energy_pipeline.db.repository import upsert_frame
from nl_energy_pipeline.db.session import session_scope
from nl_energy_pipeline.ingestion import entsoe, open_meteo
from nl_energy_pipeline.ingestion.resilience import call_with_retries
from nl_energy_pipeline.ingestion.storage import save_raw

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Source:
    name: str
    fetch: Callable[[pd.Timestamp, pd.Timestamp], pd.DataFrame]
    model: type[models.Base]
    raw_dir: Path


def month_chunks(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Split [start, end) on calendar-month boundaries (UTC)."""
    chunks = []
    cursor = start
    while cursor < end:
        next_month = (cursor + pd.offsets.MonthBegin(1)).normalize()
        chunk_end = min(next_month, end)
        chunks.append((cursor, chunk_end))
        cursor = chunk_end
    return chunks


def ingest(
    start: pd.Timestamp,
    end: pd.Timestamp,
    engine: Engine,
    *,
    client=None,
    save_parquet: bool = True,
) -> dict[str, int]:
    """Fetch prices, load and weather for [start, end) month by month and upsert them.

    Each month is committed on its own and every write is an upsert, so a crash halfway
    leaves earlier months intact and simply re-running the same command is safe.
    """
    end = min(end, pd.Timestamp.now(tz="UTC").floor("h"))
    client = client or entsoe.get_client()

    sources = [
        Source(
            "day_ahead_prices",
            lambda s, e: entsoe.fetch_day_ahead_prices(client, s, e),
            models.DayAheadPrice,
            entsoe.RAW_DIR,
        ),
        Source(
            "load",
            lambda s, e: entsoe.fetch_load(client, s, e),
            models.Load,
            entsoe.RAW_DIR,
        ),
        Source("weather", open_meteo.fetch_weather, models.Weather, open_meteo.RAW_DIR),
    ]

    totals = {source.name: 0 for source in sources}
    for chunk_start, chunk_end in month_chunks(start, end):
        for source in sources:
            df = call_with_retries(partial(source.fetch, chunk_start, chunk_end))
            if df.empty:
                logger.warning("%s: no data for %s..%s", source.name, chunk_start, chunk_end)
                continue
            if save_parquet:
                save_raw(df, source.name, source.raw_dir)
            with session_scope(engine) as session:
                written = upsert_frame(session, source.model, df)
            totals[source.name] += written
            logger.info("%s %s: upserted %d rows", source.name, chunk_start.date(), written)
    return totals
