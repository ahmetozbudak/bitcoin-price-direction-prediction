import numpy as np
import warnings
from statsmodels.tsa.arima.model import ARIMA


def run(close_prices, y_test, split_idx):
    """
    ARIMA bazli yon tahmini:
    Egitim setindeki kapanis fiyatlarina ARIMA uydurur,
    test seti icin yon tahmini yapar.
    """
    warnings.filterwarnings("ignore")

    train_close = close_prices[:split_idx]
    test_close = close_prices[split_idx:]
    n_test = len(y_test)

    # Son 5000 gozlemi kullan (ARIMA buyuk veride yavas)
    max_train = 5000
    if len(train_close) > max_train:
        train_close = train_close[-max_train:]

    model = ARIMA(train_close, order=(5, 1, 0))
    fitted = model.fit()

    forecast = fitted.forecast(steps=n_test)

    # Yon tahmini: tahmin edilen fiyat > onceki gercek fiyat ise yukari
    prev_prices = np.concatenate([[train_close[-1]], test_close[:-1]])
    prev_prices = prev_prices[:n_test]
    y_pred = (forecast > prev_prices).astype(int)

    min_len = min(len(y_pred), len(y_test))
    y_pred = y_pred[:min_len]
    y_test_trimmed = y_test[:min_len]

    warnings.filterwarnings("default")
    return y_pred, y_test_trimmed
