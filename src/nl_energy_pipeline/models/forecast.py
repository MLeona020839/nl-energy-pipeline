import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from nl_energy_pipeline.features.transform import FEATURE_COLUMNS, TARGET

PARAMS = {
    "n_estimators": 400,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": 1,
    "verbose": -1,
}


def make_model() -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(**PARAMS)


def complete_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.dropna(subset=[*FEATURE_COLUMNS, TARGET])


def cross_validate(frame: pd.DataFrame, n_splits: int = 5, gap: int = 24) -> pd.DataFrame:
    """Expanding-window backtest: each fold trains on the past only and predicts the next block.

    `gap` hours are skipped between train and test so rolling features cannot straddle the split.
    Returns one out-of-sample prediction per tested hour (plus the naive baselines for scoring).
    """
    data = complete_rows(frame)
    folds = []
    for fold, (train_idx, test_idx) in enumerate(TimeSeriesSplit(n_splits, gap=gap).split(data)):
        train, test = data.iloc[train_idx], data.iloc[test_idx]
        model = make_model().fit(train[FEATURE_COLUMNS], train[TARGET])
        folds.append(
            pd.DataFrame(
                {
                    "actual": test[TARGET],
                    "predicted": model.predict(test[FEATURE_COLUMNS]),
                    "naive_24": test["price_lag_24"],
                    "naive_168": test["price_lag_168"],
                    "fold": fold,
                }
            )
        )
    return pd.concat(folds)


def score(oof: pd.DataFrame) -> dict[str, float]:
    """Model vs. 'same hour yesterday' and 'same hour last week' on identical rows."""
    result: dict[str, float] = {}
    for name in ("predicted", "naive_24", "naive_168"):
        result[f"mae_{name}"] = float(mean_absolute_error(oof["actual"], oof[name]))
        result[f"rmse_{name}"] = float(root_mean_squared_error(oof["actual"], oof[name]))
    result["skill_vs_naive_24"] = 1 - result["mae_predicted"] / result["mae_naive_24"]
    result["skill_vs_naive_168"] = 1 - result["mae_predicted"] / result["mae_naive_168"]
    return {key: round(value, 4) for key, value in result.items()}
