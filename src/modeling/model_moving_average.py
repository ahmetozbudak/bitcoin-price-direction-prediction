import pandas as pd
import numpy as np


def run(close_prices, y_test, split_idx):
    """
    SMA crossover stratejisi:
    SMA_short (10) > SMA_long (50) ise yukari (1), degilse asagi (0) tahmin eder.
    """
    close_series = pd.Series(close_prices)
    sma_short = close_series.rolling(window=10).mean()
    sma_long = close_series.rolling(window=50).mean()

    signals = (sma_short > sma_long).astype(int)
    y_pred = signals.iloc[split_idx:].values

    min_len = min(len(y_pred), len(y_test))
    y_pred = y_pred[:min_len]
    y_test_trimmed = y_test[:min_len]

    return y_pred, y_test_trimmed
