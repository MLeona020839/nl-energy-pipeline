import pandas as pd
import pytest
from fastapi.testclient import TestClient

from nl_energy_pipeline.api.main import create_app
from nl_energy_pipeline.db import models
from nl_energy_pipeline.db.repository import upsert_frame
from nl_energy_pipeline.db.session import session_scope
from tests.conftest import hourly_index


@pytest.fixture
def client(db):
    with TestClient(create_app(db)) as c:
        yield c


def seed_predictions(db, version="lgbm-20240110"):
    idx = hourly_index("2024-01-01", 4)
    df = pd.DataFrame(
        {
            "model_version": version,
            "actual_price": [10.0, 11.0, 90.0, 12.0],
            "predicted_price": [10.5, 10.5, 12.0, 12.5],
            "residual": [-0.5, 0.5, 78.0, -0.5],
            "is_anomaly": [False, False, True, False],
        },
        index=idx,
    )
    with session_scope(db) as session:
        upsert_frame(session, models.PricePrediction, df)
        session.merge(
            models.ModelRun(model_version=version, n_train_rows=100, metrics={"mae_predicted": 1.5})
        )


def test_health_reports_database_status(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_prices_returns_requested_window_in_order(db, client):
    df = pd.DataFrame({"price_eur_mwh": [1.0, 2.0, 3.0, 4.0]}, index=hourly_index("2024-01-01", 4))
    with session_scope(db) as session:
        upsert_frame(session, models.DayAheadPrice, df)

    body = client.get(
        "/prices", params={"start": "2024-01-01T01:00:00", "end": "2024-01-01T03:00:00"}
    ).json()

    assert [p["price_eur_mwh"] for p in body] == [2.0, 3.0]
    assert body[0]["ts"].startswith("2024-01-01T01:00:00")


def test_prices_rejects_inverted_window(client):
    response = client.get(
        "/prices", params={"start": "2024-02-01T00:00:00", "end": "2024-01-01T00:00:00"}
    )

    assert response.status_code == 422


def test_anomalies_only_returns_flagged_hours(db, client):
    seed_predictions(db)

    body = client.get("/anomalies").json()

    assert [(a["actual_price"], a["is_anomaly"]) for a in body] == [(90.0, True)]


def test_predictions_return_every_hour_for_latest_model(db, client):
    seed_predictions(db)

    assert len(client.get("/predictions").json()) == 4


def test_model_endpoint_exposes_metrics_and_404s_before_training(db, client):
    assert client.get("/model").status_code == 404
    assert client.get("/anomalies").status_code == 404

    seed_predictions(db)
    body = client.get("/model").json()

    assert body["model_version"] == "lgbm-20240110"
    assert body["metrics"] == {"mae_predicted": 1.5}
