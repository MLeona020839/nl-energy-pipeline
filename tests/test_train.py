import pandas as pd

from nl_energy_pipeline.db import models
from nl_energy_pipeline.db.repository import read_frame, upsert_frame
from nl_energy_pipeline.db.session import session_scope
from nl_energy_pipeline.models.train import train
from tests.conftest import synthetic_market


def test_training_beats_naive_baselines_and_persists_results(db, tmp_path, monkeypatch):
    prices, load, weather = synthetic_market(days=400)
    spike_ts = prices.index[-300]
    prices.loc[spike_ts, "price_eur_mwh"] += 400  # an obvious outlier in the backtest period

    with session_scope(db) as session:
        upsert_frame(session, models.DayAheadPrice, prices)
        upsert_frame(session, models.Load, load)
        upsert_frame(session, models.Weather, weather)
    monkeypatch.setattr("nl_energy_pipeline.models.train.settings.artifact_dir", tmp_path)

    result = train(db)

    assert result["mae_predicted"] < result["mae_naive_168"]
    assert result["skill_vs_naive_168"] > 0
    assert result["feature_importance_gain_pct"]
    assert (tmp_path / f"{result['model_version']}.txt").exists()

    predictions = read_frame(db, models.PricePrediction)
    assert len(predictions) == result["oof_rows"]
    assert predictions.loc[spike_ts, "is_anomaly"]
    assert predictions["is_anomaly"].mean() < 0.02

    with db.connect() as conn:
        stored = conn.exec_driver_sql("SELECT metrics->>'mae_predicted' FROM model_runs").scalar()
    assert float(stored) == result["mae_predicted"]


def test_retraining_the_same_version_replaces_its_predictions(db, tmp_path, monkeypatch):
    prices, load, weather = synthetic_market(days=300)
    with session_scope(db) as session:
        upsert_frame(session, models.DayAheadPrice, prices)
        upsert_frame(session, models.Load, load)
        upsert_frame(session, models.Weather, weather)
    monkeypatch.setattr("nl_energy_pipeline.models.train.settings.artifact_dir", tmp_path)

    first = train(db)
    stale = pd.DataFrame(
        {
            "model_version": first["model_version"],
            "actual_price": 1.0,
            "predicted_price": 1.0,
            "residual": 0.0,
            "is_anomaly": False,
        },
        index=pd.DatetimeIndex(["2020-01-01"], tz="UTC", name="ts"),
    )
    with session_scope(db) as session:
        upsert_frame(session, models.PricePrediction, stale)

    second = train(db)

    kept = read_frame(db, models.PricePrediction)
    assert pd.Timestamp("2020-01-01", tz="UTC") not in kept.index
    assert len(kept) == second["oof_rows"]
