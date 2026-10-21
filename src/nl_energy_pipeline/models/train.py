import logging

import pandas as pd
from sqlalchemy import Engine

from nl_energy_pipeline.config import settings
from nl_energy_pipeline.db import models
from nl_energy_pipeline.db.repository import read_frame, upsert_frame
from nl_energy_pipeline.db.session import session_scope
from nl_energy_pipeline.features.transform import FEATURE_COLUMNS, TARGET, build_features
from nl_energy_pipeline.models.anomaly import flag_anomalies
from nl_energy_pipeline.models.forecast import complete_rows, cross_validate, make_model, score

logger = logging.getLogger(__name__)


def load_features(engine: Engine) -> pd.DataFrame:
    return build_features(
        read_frame(engine, models.DayAheadPrice),
        read_frame(engine, models.Load),
        read_frame(engine, models.Weather),
    )


def train(engine: Engine, *, save_booster: bool = True) -> dict:
    """Backtest, flag anomalies, fit the final model, and persist everything to Postgres."""
    frame = load_features(engine)
    data = complete_rows(frame)
    logger.info(
        "training on %d complete hourly rows (%s .. %s)",
        len(data),
        data.index.min(),
        data.index.max(),
    )

    oof = cross_validate(frame)
    oof["residual"] = oof["actual"] - oof["predicted"]
    oof["is_anomaly"] = flag_anomalies(oof["residual"])

    metrics = score(oof)
    metrics["oof_rows"] = int(len(oof))
    metrics["anomaly_count"] = int(oof["is_anomaly"].sum())
    metrics["anomaly_rate"] = round(float(oof["is_anomaly"].mean()), 4)

    final = make_model().fit(data[FEATURE_COLUMNS], data[TARGET])
    gain = final.booster_.feature_importance("gain")
    metrics["feature_importance_gain_pct"] = {
        name: round(100 * float(g) / float(gain.sum()), 2)
        for name, g in sorted(zip(FEATURE_COLUMNS, gain, strict=True), key=lambda x: -x[1])
    }

    version = f"lgbm-{data.index.max():%Y%m%d}"
    predictions = oof[["actual", "predicted", "residual", "is_anomaly"]].rename(
        columns={"actual": "actual_price", "predicted": "predicted_price"}
    )
    predictions.insert(0, "model_version", version)

    with session_scope(engine) as session:
        upsert_frame(session, models.PricePrediction, predictions)
        session.merge(
            models.ModelRun(model_version=version, n_train_rows=len(data), metrics=metrics)
        )

    if save_booster:
        settings.artifact_dir.mkdir(parents=True, exist_ok=True)
        final.booster_.save_model(str(settings.artifact_dir / f"{version}.txt"))

    logger.info(
        "model %s: %s",
        version,
        {k: v for k, v in metrics.items() if k != "feature_importance_gain_pct"},
    )
    return {"model_version": version, **metrics}
