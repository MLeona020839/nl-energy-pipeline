import pandas as pd
from sqlalchemy import Engine, func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from nl_energy_pipeline.db.models import Base

BATCH_SIZE = 5000  # rows * columns must stay below Postgres' 65535 bind-parameter limit


def upsert_frame(session: Session, model: type[Base], df: pd.DataFrame) -> int:
    """Idempotent write: insert new rows, overwrite rows whose primary key already exists.

    `df` is indexed by a tz-aware timestamp (becomes the `ts` column); every other
    column must exist on the model. Returns the number of rows sent to the database.
    """
    if df.empty:
        return 0

    table = model.__table__
    pk_cols = [c.name for c in table.primary_key.columns]

    frame = df[~df.index.duplicated(keep="last")].rename_axis("ts").reset_index()
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True)
    unknown = set(frame.columns) - set(table.columns.keys())
    if unknown:
        raise ValueError(f"Columns not on {table.name}: {sorted(unknown)}")

    # NaN -> NULL
    frame = frame.astype(object).where(frame.notna(), None)
    update_cols = [c for c in frame.columns if c not in pk_cols]

    written = 0
    for i in range(0, len(frame), BATCH_SIZE):
        batch = frame.iloc[i : i + BATCH_SIZE].to_dict("records")
        stmt = insert(table).values(batch)
        set_ = {c: stmt.excluded[c] for c in update_cols}
        if "ingested_at" in table.columns:
            set_["ingested_at"] = func.now()
        session.execute(stmt.on_conflict_do_update(index_elements=pk_cols, set_=set_))
        written += len(batch)
    return written


def read_frame(
    engine: Engine,
    model: type[Base],
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Read rows with start <= ts < end as a UTC-indexed DataFrame (metadata columns dropped)."""
    table = model.__table__
    query = select(table)
    if start is not None:
        query = query.where(table.c.ts >= start.to_pydatetime())
    if end is not None:
        query = query.where(table.c.ts < end.to_pydatetime())
    query = query.order_by(table.c.ts)

    df = pd.read_sql(query, engine)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.set_index("ts")
    return df.drop(columns=[c for c in ("ingested_at", "created_at") if c in df.columns])
