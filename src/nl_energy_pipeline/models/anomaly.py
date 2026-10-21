import pandas as pd


def flag_anomalies(
    residuals: pd.Series, window: int = 720, min_periods: int = 168, k: float = 4.0
) -> pd.Series:
    """Flag hours whose forecast residual is extreme relative to the *recent past*.

    The scale is a robust sigma (1.4826 * MAD) over the trailing `window` hours, computed
    from residuals strictly before the hour being judged (`shift(1)`), so there is no
    look-ahead and a spike cannot inflate its own threshold. Early hours without enough
    history are never flagged.
    """
    past = residuals.shift(1)
    center = past.rolling(window, min_periods=min_periods).median()
    mad = (past - center).abs().rolling(window, min_periods=min_periods).median()
    sigma = 1.4826 * mad
    return ((residuals - center).abs() > k * sigma) & (sigma > 0)
