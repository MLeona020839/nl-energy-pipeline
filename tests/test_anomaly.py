import numpy as np
import pandas as pd

from nl_energy_pipeline.models.anomaly import flag_anomalies


def residuals(n: int = 2000, seed: int = 1) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(
        rng.normal(0, 5, n), index=pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    )


def test_spike_is_flagged_and_normal_noise_mostly_is_not():
    r = residuals()
    r.iloc[1500] = 120.0

    flags = flag_anomalies(r)

    assert flags.iloc[1500]
    assert flags.mean() < 0.01


def test_hours_without_enough_history_are_never_flagged():
    r = residuals()
    r.iloc[10] = 500.0

    assert not flag_anomalies(r).iloc[:168].any()


def test_a_spike_cannot_hide_by_inflating_its_own_threshold():
    r = residuals()
    r.iloc[1500:1503] = [100.0, 100.0, 100.0]

    flags = flag_anomalies(r)

    assert flags.iloc[1500]  # judged against the past only
