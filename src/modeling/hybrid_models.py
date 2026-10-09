"""
Hybrid Models for BTC Trend Prediction

Three hybrid approaches:
  1. Feature Augmentation  - Trend signals as extra ML features
  2. Weighted Voting       - Validation-weighted ensemble
  3. Stacking              - Meta-learner over base model predictions
"""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score as compute_f1
from .utils import scale_data, calc_metrics


# ──────────────────────────────────────────────────────────────
#  Shared helpers
# ──────────────────────────────────────────────────────────────

def _train_ml(name, X_train, y_train, X_test):
    """Train a single ML model and return test predictions."""
    if name == "XGBoost":
        from xgboost import XGBClassifier
        m = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=42, eval_metric="logloss", verbosity=0)
        m.fit(X_train, y_train)
        return m.predict(X_test)

    if name == "Random Forest":
        from sklearn.ensemble import RandomForestClassifier
        m = RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
        m.fit(X_train, y_train)
        return m.predict(X_test)

    if name == "MLP":
        from sklearn.neural_network import MLPClassifier
        Xtr, Xte = scale_data(X_train, X_test)
        m = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32), random_state=42,
            max_iter=300, early_stopping=True, verbose=False)
        m.fit(Xtr, y_train)
        return m.predict(Xte)

    if name == "SVM":
        from sklearn.svm import SVC
        Xtr, Xte = scale_data(X_train, X_test)
        m = SVC(kernel="rbf", C=1.0, gamma="scale",
                random_state=42, max_iter=5000)
        m.fit(Xtr, y_train)
        return m.predict(Xte)

    raise ValueError(f"Bilinmeyen model: {name}")


def _ma_signal(close_prices, start, length, short=10, long=50):
    """Moving-average crossover signal for a slice of close prices."""
    s = pd.Series(close_prices)
    sma_s = s.rolling(window=short, min_periods=1).mean()
    sma_l = s.rolling(window=long, min_periods=1).mean()
    sig = (sma_s > sma_l).astype(int).values
    return sig[start: start + length]


def _arima_signal(close_prices, train_end, n_forecast):
    """ARIMA direction forecast."""
    import warnings
    from statsmodels.tsa.arima.model import ARIMA

    warnings.filterwarnings("ignore")
    train = close_prices[:train_end]
    if len(train) > 5000:
        train = train[-5000:]

    model = ARIMA(train, order=(5, 1, 0))
    fitted = model.fit()
    fc = fitted.forecast(steps=n_forecast)

    test = close_prices[train_end: train_end + n_forecast]
    prev = np.concatenate([[train[-1]], test[:-1]])[:n_forecast]
    sig = (fc[:n_forecast] > prev).astype(int)
    warnings.filterwarnings("default")
    return sig


# ──────────────────────────────────────────────────────────────
#  1.  Feature Augmentation
# ──────────────────────────────────────────────────────────────

def _trend_features(close_prices, sma_short=10, sma_long=50):
    """
    Compute three binary trend signals (no future data used):
      - MA crossover  (SMA_short > SMA_long)
      - SMA slope     (SMA_short rising over 3 periods)
      - Price > SMA   (close above SMA_long)
    """
    c = pd.Series(close_prices)
    sma_s = c.rolling(window=sma_short, min_periods=1).mean()
    sma_l = c.rolling(window=sma_long, min_periods=1).mean()

    sig_cross = (sma_s > sma_l).astype(float).values
    sig_slope = (sma_s.diff(3) > 0).astype(float).fillna(0).values
    sig_pos = (c > sma_l).astype(float).values

    return np.column_stack([sig_cross, sig_slope, sig_pos])


def run_feature_augmentation(data, params):
    sidx = data["split_idx"]
    n_te = len(data["X_test"])

    trend = _trend_features(
        data["close_prices"],
        params.get("sma_short", 10),
        params.get("sma_long", 50),
    )

    X_tr = np.column_stack([data["X_train"], trend[:sidx]])
    X_te = np.column_stack([data["X_test"], trend[sidx: sidx + n_te]])
    y_train = data["y_train"]
    y_test = data["y_test"]

    ml = params.get("ml_model", "XGBoost")
    needs_scale = ml in ("MLP",)
    if needs_scale:
        X_tr, X_te = scale_data(X_tr, X_te)

    if ml == "XGBoost":
        from xgboost import XGBClassifier
        model = XGBClassifier(
            n_estimators=params.get("n_estimators", 200),
            max_depth=params.get("max_depth", 6),
            learning_rate=params.get("learning_rate", 0.1),
            random_state=42, eval_metric="logloss", verbosity=0)
    elif ml == "Random Forest":
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(
            n_estimators=params.get("n_estimators", 200),
            max_depth=params.get("max_depth", 10),
            random_state=42, n_jobs=-1)
    elif ml == "MLP":
        from sklearn.neural_network import MLPClassifier
        model = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32), random_state=42,
            max_iter=params.get("max_iter", 300),
            early_stopping=True, verbose=False)
    else:
        raise ValueError(f"Desteklenmeyen ML modeli: {ml}")

    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)

    result = calc_metrics(y_test, y_pred, f"Hybrid FA ({ml})")

    extra = (
        f"\n\nTrend Feature Bilgisi:\n"
        f"  SMA Kisa Periyot : {params.get('sma_short', 10)}\n"
        f"  SMA Uzun Periyot : {params.get('sma_long', 50)}\n"
        f"  Eklenen feature  : signal_ma_cross, signal_sma_slope, signal_price_pos\n"
        f"  Toplam feature   : {X_tr.shape[1]}\n"
    )
    result["report"] += extra
    return result


# ──────────────────────────────────────────────────────────────
#  2.  Weighted Voting
# ──────────────────────────────────────────────────────────────

def run_weighted_voting(data, params):
    X_train = data["X_train"]
    X_test  = data["X_test"]
    y_train = data["y_train"]
    y_test  = data["y_test"]
    close   = data["close_prices"]
    sidx    = data["split_idx"]
    n_te    = len(X_test)

    ml_names = []
    if params.get("use_xgboost", True):  ml_names.append("XGBoost")
    if params.get("use_rf", True):       ml_names.append("Random Forest")
    if params.get("use_mlp", True):      ml_names.append("MLP")
    if params.get("use_svm", False):     ml_names.append("SVM")
    use_ma    = params.get("use_ma", True)
    use_arima = params.get("use_arima", False)

    all_names = ml_names[:]
    if use_ma:    all_names.append("MA")
    if use_arima: all_names.append("ARIMA")

    if not all_names:
        raise ValueError("En az bir model secilmeli!")

    # ── validation split for weight calculation ──
    val_ratio = params.get("val_ratio", 0.15)
    val_size  = max(int(len(X_train) * val_ratio), 100)
    X_tr, X_val = X_train[:-val_size], X_train[-val_size:]
    y_tr, y_val = y_train[:-val_size], y_train[-val_size:]

    weights = {}
    for name in ml_names:
        pred = _train_ml(name, X_tr, y_tr, X_val)
        f1 = compute_f1(y_val, pred, average="weighted", zero_division=0)
        weights[name] = max(f1, 0.01)

    if use_ma:
        val_start = sidx - val_size
        pred = _ma_signal(close, val_start, val_size)
        f1 = compute_f1(y_val, pred[:len(y_val)],
                        average="weighted", zero_division=0)
        weights["MA"] = max(f1, 0.01)

    if use_arima:
        val_train_end = sidx - val_size
        pred = _arima_signal(close, val_train_end, val_size)
        f1 = compute_f1(y_val, pred[:len(y_val)],
                        average="weighted", zero_division=0)
        weights["ARIMA"] = max(f1, 0.01)

    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}

    # ── retrain on full training data ──
    test_preds = {}
    for name in ml_names:
        test_preds[name] = _train_ml(name, X_train, y_train, X_test)
    if use_ma:
        test_preds["MA"] = _ma_signal(close, sidx, n_te)
    if use_arima:
        test_preds["ARIMA"] = _arima_signal(close, sidx, n_te)

    # ── weighted vote ──
    w_sum = np.zeros(n_te)
    for name, pred in test_preds.items():
        p = pred[:n_te]
        w_sum[:len(p)] += weights[name] * p

    y_pred = (w_sum > 0.5).astype(int)

    result = calc_metrics(y_test, y_pred, "Hybrid Voting")

    wt_lines = "\n\nModel Agirliklari (Validation F1 bazli):\n"
    for name in all_names:
        wt_lines += f"  {name:<20s}  w = {weights[name]:.4f}\n"
    result["report"] += wt_lines
    result["weights"] = weights
    return result


# ──────────────────────────────────────────────────────────────
#  3.  Stacking  (Meta-Learner)
# ──────────────────────────────────────────────────────────────

def run_stacking(data, params):
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.linear_model import LogisticRegression

    X_train = data["X_train"]
    X_test  = data["X_test"]
    y_train = data["y_train"]
    y_test  = data["y_test"]
    close   = data["close_prices"]
    sidx    = data["split_idx"]
    n_te    = len(X_test)

    n_folds = params.get("n_folds", 5)

    base = []
    if params.get("use_xgboost", True):  base.append("XGBoost")
    if params.get("use_rf", True):       base.append("Random Forest")
    if params.get("use_mlp", True):      base.append("MLP")
    use_ma = params.get("use_ma", True)

    all_names = base[:]
    if use_ma:
        all_names.append("MA")

    if not all_names:
        raise ValueError("En az bir model secilmeli!")

    n_tr = len(X_train)
    tscv = TimeSeriesSplit(n_splits=n_folds)

    oof = {name: np.full(n_tr, np.nan) for name in all_names}

    for tr_idx, val_idx in tscv.split(X_train):
        Xf_tr, Xf_val = X_train[tr_idx], X_train[val_idx]
        yf_tr = y_train[tr_idx]

        for name in base:
            oof[name][val_idx] = _train_ml(name, Xf_tr, yf_tr, Xf_val)

        if use_ma:
            oof["MA"][val_idx] = _ma_signal(
                close, val_idx[0], len(val_idx))[:len(val_idx)]

    # meta-train matrix (rows with predictions only)
    meta_full = np.column_stack([oof[n] for n in all_names])
    valid = ~np.isnan(meta_full).any(axis=1)
    meta_train = meta_full[valid]
    y_meta = y_train[valid]

    # test predictions from full-training models
    test_cols = []
    for name in base:
        test_cols.append(_train_ml(name, X_train, y_train, X_test))
    if use_ma:
        test_cols.append(_ma_signal(close, sidx, n_te)[:n_te])
    meta_test = np.column_stack(test_cols)

    # meta-learner
    ml_type = params.get("meta_learner", "Logistic Regression")
    if ml_type == "Ridge Classifier":
        from sklearn.linear_model import RidgeClassifier
        meta = RidgeClassifier(random_state=42)
    else:
        meta = LogisticRegression(random_state=42, max_iter=1000)

    meta.fit(meta_train, y_meta)
    y_pred = meta.predict(meta_test)

    result = calc_metrics(y_test, y_pred, "Hybrid Stacking")

    extra = (
        f"\n\nStacking Bilgisi:\n"
        f"  Meta-Learner  : {ml_type}\n"
        f"  CV Fold       : {n_folds}\n"
        f"  Base Modeller : {', '.join(all_names)}\n"
        f"  Meta egitim   : {meta_train.shape[0]} satir\n"
    )
    result["report"] += extra
    return result
