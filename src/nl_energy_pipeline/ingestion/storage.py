from pathlib import Path

import pandas as pd


def save_raw(df: pd.DataFrame, name: str, out_dir: Path) -> Path | None:
    """Land a fetched frame as parquet; the file name carries the covered date range.

    Returns the written path, or None when there was nothing to save.
    """
    if df.empty:
        return None

    filename = f"{name}_{df.index.min().date()}_{df.index.max().date()}.parquet"
    filepath = out_dir / filename

    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(filepath)

    return filepath
