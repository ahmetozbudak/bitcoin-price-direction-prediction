import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)
from sklearn.base import clone

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = str(PROJECT_ROOT / "data" / "processed")

NON_FEATURE_COLS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_asset_volume", "number_of_trades",
    "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume",
    "target",
]


def load_data(filepath):
    df = pd.read_csv(filepath, parse_dates=["open_time"])
    df = df.sort_values("open_time").reset_index(drop=True)
    return df


def prepare_data(
    df,
    target_mode="next_bar",
    *,
    threshold_pct=0.002,
    horizon=3,
):
    """
    Hedef degiskeni olusturur, feature/target ayirir.

    target_mode:
      - "next_bar"       : bir sonraki mumda yukari mi (klasik)
      - "threshold_next": sonraki getiri mutlak degeri threshold_pct'ten kucuk
                          satirlar elenir; kalanlarda yon 0/1
      - "horizon_k"      : k mum sonra close, simdikinden yukari mi

    threshold_pct: oran (or. 0.002 = %0.2) — sadece threshold_next icin
    horizon: tam sayi — sadece horizon_k icin
    """
    from .target_definitions import build_target

    df, target_meta = build_target(
        df,
        target_mode,
        threshold_pct=float(threshold_pct),
        horizon=int(horizon),
    )

    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = df[feature_cols].values
    y = df["target"].values
    close_prices = df["close"].values

    return X, y, feature_cols, close_prices, target_meta


def split_data(X, y, train_ratio=0.8):
    """Kronolojik 80/20 bolme (shuffle yok)."""
    split_idx = int(len(X) * train_ratio)
    return (
        X[:split_idx], X[split_idx:],
        y[:split_idx], y[split_idx:],
        split_idx,
    )


def probability_threshold_predict(
    model,
    X_train,
    y_train,
    X_test,
    *,
    val_ratio=0.15,
    min_val_samples=200,
    n_thresholds=91,
):
    """
    Egitim verisinin son val_ratio diliminde (kronolojik) olasiliklari kullanarak
    pozitif sinif esigini F1 (weighted) ile arar; modeli tum egitimde yeniden egitir,
    secilen esigi yalnizca test olasiliklarina uygular (test etiketi esik seciminde kullanilmaz).

    Donus: (y_pred_test, meta_dict, m_full)  — m_full: tum egitimde fit edilmis model
    """
    n = len(X_train)
    n_val = max(int(n * val_ratio), min_val_samples)
    if n_val >= n - 50:
        n_val = max(n // 5, min(100, n // 2))

    X_fit = X_train[:-n_val]
    y_fit = y_train[:-n_val]
    X_val = X_train[-n_val:]
    y_val = y_train[-n_val:]

    m = clone(model)
    m.fit(X_fit, y_fit)
    proba_val = m.predict_proba(X_val)
    if proba_val.shape[1] < 2:
        raise ValueError("predict_proba ikinci sinif uretmiyor (tek sinif egitim?)")
    p_val = proba_val[:, 1]

    thresholds = np.linspace(0.05, 0.95, n_thresholds)
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        y_hat = (p_val >= t).astype(int)
        f1 = f1_score(y_val, y_hat, average="weighted", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)

    m_full = clone(model)
    m_full.fit(X_train, y_train)
    p_test = m_full.predict_proba(X_test)[:, 1]
    y_pred = (p_test >= best_t).astype(int)

    meta = {
        "threshold": best_t,
        "val_f1_tuned": best_f1,
        "val_samples": n_val,
    }
    return y_pred, meta, m_full


def scale_data_fit(X_train, X_test):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_test_s, scaler


def scale_data(X_train, X_test):
    X_train_s, X_test_s, _ = scale_data_fit(X_train, X_test)
    return X_train_s, X_test_s


def calc_metrics(y_true, y_pred, model_name):
    return {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "cm": confusion_matrix(y_true, y_pred),
        "report": classification_report(y_true, y_pred, target_names=["Asagi (0)", "Yukari (1)"], zero_division=0),
    }
