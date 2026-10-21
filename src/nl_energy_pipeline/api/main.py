from collections.abc import Iterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from pydantic import BaseModel
from sqlalchemy import Engine, select, text
from sqlalchemy.orm import Session

from nl_energy_pipeline.db import models
from nl_energy_pipeline.db.session import get_engine

MAX_ROWS = 5000


class PricePoint(BaseModel):
    ts: datetime
    price_eur_mwh: float


class PredictionPoint(BaseModel):
    ts: datetime
    model_version: str
    actual_price: float
    predicted_price: float
    residual: float
    is_anomaly: bool


class ModelInfo(BaseModel):
    model_version: str
    trained_at: datetime
    n_train_rows: int
    metrics: dict


def _as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _window(start: datetime | None, end: datetime | None) -> tuple[datetime, datetime]:
    """Default to the last 7 days; naive timestamps are read as UTC."""
    end_utc = _as_utc(end) if end else datetime.now(UTC)
    start_utc = _as_utc(start) if start else end_utc - timedelta(days=7)
    if start_utc >= end_utc:
        raise HTTPException(status_code=422, detail="start must be before end")
    return start_utc, end_utc


def create_app(engine: Engine | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.engine = engine or get_engine()
        yield
        app.state.engine.dispose()

    app = FastAPI(
        title="NL Energy Pipeline",
        description="Dutch day-ahead prices, out-of-sample forecasts and price anomalies.",
        lifespan=lifespan,
    )

    def db(request: Request) -> Iterator[Session]:
        with Session(request.app.state.engine) as session:
            yield session

    @app.get("/health")
    def health(session: Session = Depends(db)) -> dict:
        try:
            session.execute(text("SELECT 1"))
        except Exception as exc:
            raise HTTPException(status_code=503, detail="database unavailable") from exc
        return {"status": "ok"}

    @app.get("/prices", response_model=list[PricePoint])
    def prices(
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = Query(MAX_ROWS, ge=1, le=MAX_ROWS),
        session: Session = Depends(db),
    ):
        start_utc, end_utc = _window(start, end)
        t = models.DayAheadPrice
        rows = session.execute(
            select(t.ts, t.price_eur_mwh)
            .where(t.ts >= start_utc, t.ts < end_utc)
            .order_by(t.ts)
            .limit(limit)
        ).all()
        return [PricePoint(ts=r.ts, price_eur_mwh=r.price_eur_mwh) for r in rows]

    def latest_version(session: Session) -> str | None:
        return session.scalar(
            select(models.ModelRun.model_version)
            .order_by(models.ModelRun.trained_at.desc())
            .limit(1)
        )

    def predictions_query(
        session: Session,
        start: datetime | None,
        end: datetime | None,
        limit: int,
        *,
        only_anomalies: bool,
    ) -> list[PredictionPoint]:
        version = latest_version(session)
        if version is None:
            raise HTTPException(
                status_code=404, detail="no trained model yet; run `nl-energy-pipeline train`"
            )
        # Predictions only exist for the backtest period: default to its most recent 30 days.
        if start is None and end is None:
            last_ts = session.scalar(
                select(models.PricePrediction.ts)
                .where(models.PricePrediction.model_version == version)
                .order_by(models.PricePrediction.ts.desc())
                .limit(1)
            )
            end = (last_ts + timedelta(hours=1)) if last_ts else None
            start = (end - timedelta(days=30)) if end else None
        start_utc, end_utc = _window(start, end)
        t = models.PricePrediction
        query = select(t).where(t.model_version == version, t.ts >= start_utc, t.ts < end_utc)
        if only_anomalies:
            query = query.where(t.is_anomaly.is_(True))
        rows = session.scalars(query.order_by(t.ts).limit(limit)).all()
        return [
            PredictionPoint(
                ts=r.ts,
                model_version=r.model_version,
                actual_price=r.actual_price,
                predicted_price=r.predicted_price,
                residual=r.residual,
                is_anomaly=r.is_anomaly,
            )
            for r in rows
        ]

    @app.get("/predictions", response_model=list[PredictionPoint])
    def predictions(
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = Query(MAX_ROWS, ge=1, le=MAX_ROWS),
        session: Session = Depends(db),
    ):
        return predictions_query(session, start, end, limit, only_anomalies=False)

    @app.get("/anomalies", response_model=list[PredictionPoint])
    def anomalies(
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = Query(MAX_ROWS, ge=1, le=MAX_ROWS),
        session: Session = Depends(db),
    ):
        return predictions_query(session, start, end, limit, only_anomalies=True)

    @app.get("/model", response_model=ModelInfo)
    def model_info(session: Session = Depends(db)):
        run = session.scalars(
            select(models.ModelRun).order_by(models.ModelRun.trained_at.desc()).limit(1)
        ).first()
        if run is None:
            raise HTTPException(status_code=404, detail="no trained model yet")
        return ModelInfo(
            model_version=run.model_version,
            trained_at=run.trained_at,
            n_train_rows=run.n_train_rows,
            metrics=run.metrics,
        )

    return app


app = create_app()
