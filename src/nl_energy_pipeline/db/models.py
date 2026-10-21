from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, Index, Integer, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


def _ts_pk() -> Mapped[datetime]:
    # timestamptz, always stored/returned in UTC; the timestamp itself is the natural key.
    return mapped_column(DateTime(timezone=True), primary_key=True)


class DayAheadPrice(Base):
    __tablename__ = "day_ahead_prices"

    ts: Mapped[datetime] = _ts_pk()
    price_eur_mwh: Mapped[float] = mapped_column(Float, nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class Load(Base):
    __tablename__ = "load"

    ts: Mapped[datetime] = _ts_pk()
    load_mw: Mapped[float] = mapped_column(Float, nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class Weather(Base):
    __tablename__ = "weather"

    ts: Mapped[datetime] = _ts_pk()
    temperature_2m: Mapped[float | None] = mapped_column(Float)
    wind_speed_10m: Mapped[float | None] = mapped_column(Float)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class PricePrediction(Base):
    """Out-of-sample forecast per hour, plus the anomaly flag derived from its residual."""

    __tablename__ = "price_predictions"
    __table_args__ = (Index("ix_price_predictions_anomalies", "model_version", "is_anomaly", "ts"),)

    ts: Mapped[datetime] = _ts_pk()
    model_version: Mapped[str] = mapped_column(String(64), primary_key=True)
    actual_price: Mapped[float] = mapped_column(Float, nullable=False)
    predicted_price: Mapped[float] = mapped_column(Float, nullable=False)
    residual: Mapped[float] = mapped_column(Float, nullable=False)
    is_anomaly: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ModelRun(Base):
    """One row per training run: which data it saw and how it scored out-of-sample."""

    __tablename__ = "model_runs"

    model_version: Mapped[str] = mapped_column(String(64), primary_key=True)
    trained_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    n_train_rows: Mapped[int] = mapped_column(Integer, nullable=False)
    metrics: Mapped[dict] = mapped_column(JSONB, nullable=False)
