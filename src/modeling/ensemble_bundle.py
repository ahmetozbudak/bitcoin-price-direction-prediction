"""
Oylama (weighted voting) ve stacking icin egitim + joblib paketi + tahmin.

Egitim uygulamasindan bagimsiz tahmin programi bu modulu yukler.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score as compute_f1

from .utils import load_data, scale_data_fit


def _fit_ml(name, X_train, y_train):
    """ML bileseni egit; (model, scaler|None) doner."""
    if name == "XGBoost":
        from xgboost import XGBClassifier
        m = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=42, eval_metric="logloss", verbosity=0)
        m.fit(X_train, y_train)
        return m, None

    if name == "Random Forest":
        from sklearn.ensemble import RandomForestClassifier
        m = RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
        m.fit(X_train, y_train)
        return m, None

    if name == "MLP":
        from sklearn.neural_network import MLPClassifier
        Xs, _, sc = scale_data_fit(X_train, X_train)
        m = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32), random_state=42,
            max_iter=300, early_stopping=True, verbose=False)
        m.fit(Xs, y_train)
        return m, sc

    if name == "SVM":
        from sklearn.svm import SVC
        Xs, _, sc = scale_data_fit(X_train, X_train)
        m = SVC(
            kernel="rbf", C=1.0, gamma="scale",
            random_state=42, max_iter=5000, probability=True)
        m.fit(Xs, y_train)
        return m, sc

    raise ValueError(f"Bilinmeyen ML: {name}")


def _ml_predict_row(name, model, scaler, X_row_feats):
    """X_row_feats: (1, n_feat) ham ozellik."""
    if scaler is not None:
        Xp = scaler.transform(X_row_feats)
    else:
        Xp = X_row_feats
    return int(model.predict(Xp)[0])


def _ml_predict_proba_up(name, model, scaler, X_row_feats):
    if scaler is not None:
        Xp = scaler.transform(X_row_feats)
    else:
        Xp = X_row_feats
    pr = model.predict_proba(Xp)[0]
    return float(pr[1]) if len(pr) > 1 else float(pr[0])


def _ma_signal_full(close: np.ndarray, short: int, long: int) -> np.ndarray:
    s = pd.Series(close)
    sma_s = s.rolling(window=short, min_periods=1).mean()
    sma_l = s.rolling(window=long, min_periods=1).mean()
    return (sma_s > sma_l).astype(int).values


def _arima_signal(close_prices, train_end, n_forecast):
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


def _arima_direction_next(close: np.ndarray, max_train: int = 5000) -> int:
    """Son gozlem icin bir adim ileri yon (1=yukari)."""
    import warnings
    from statsmodels.tsa.arima.model import ARIMA

    if len(close) < 20:
        return 0
    warnings.filterwarnings("ignore")
    train = close.copy()
    if len(train) > max_train:
        train = train[-max_train:]
    last = float(train[-1])
    model = ARIMA(train, order=(5, 1, 0))
    fitted = model.fit()
    fc = float(fitted.forecast(steps=1)[0])
    warnings.filterwarnings("default")
    return int(fc > last)


# ── Voting bundle ────────────────────────────────────────────

def fit_voting_bundle(data: dict, params: dict) -> dict:
    """
    data: X_train, X_test, y_train, y_test, close_prices, split_idx, feature_cols
    """
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    close = np.asarray(data["close_prices"], dtype=float)
    sidx = int(data["split_idx"])
    n_te = len(X_test)
    feats = list(data["feature_cols"])

    ml_names = []
    if params.get("use_xgboost", True):
        ml_names.append("XGBoost")
    if params.get("use_rf", True):
        ml_names.append("Random Forest")
    if params.get("use_mlp", True):
        ml_names.append("MLP")
    if params.get("use_svm", False):
        ml_names.append("SVM")
    use_ma = params.get("use_ma", True)
    use_arima = params.get("use_arima", False)

    all_names = ml_names[:]
    if use_ma:
        all_names.append("MA")
    if use_arima:
        all_names.append("ARIMA")
    if not all_names:
        raise ValueError("En az bir bilesen secin")

    val_ratio = float(params.get("val_ratio", 0.15))
    val_size = max(int(len(X_train) * val_ratio), 100)
    X_tr, X_val = X_train[:-val_size], X_train[-val_size:]
    y_tr, y_val = y_train[:-val_size], y_train[-val_size:]

    weights = {}
    for name in ml_names:
        m, sc = _fit_ml(name, X_tr, y_tr)
        if sc is None:
            pv = m.predict(X_val)
        else:
            pv = m.predict(sc.transform(X_val))
        weights[name] = max(compute_f1(y_val, pv, average="weighted", zero_division=0), 0.01)

    if use_ma:
        val_start = sidx - val_size
        pred = _ma_signal_full(close, params.get("sma_short", 10), params.get("sma_long", 50))
        pred = pred[val_start: val_start + len(y_val)]
        weights["MA"] = max(
            compute_f1(y_val, pred[: len(y_val)], average="weighted", zero_division=0), 0.01)

    if use_arima:
        val_train_end = sidx - val_size
        pred_ar = _arima_signal(close, val_train_end, val_size)
        weights["ARIMA"] = max(
            compute_f1(y_val, pred_ar[: len(y_val)], average="weighted", zero_division=0), 0.01)

    tot = sum(weights.values())
    weights = {k: v / tot for k, v in weights.items()}

    components = {}
    for name in ml_names:
        m, sc = _fit_ml(name, X_train, y_train)
        components[name] = {"model": m, "scaler": sc, "type": "sklearn"}

    if use_ma:
        components["MA"] = {
            "type": "ma",
            "short": int(params.get("sma_short", 10)),
            "long": int(params.get("sma_long", 50)),
        }
    if use_arima:
        components["ARIMA"] = {"type": "arima", "order": (5, 1, 0), "max_train": 5000}

    bundle = {
        "kind": "voting",
        "feature_cols": feats,
        "weights": weights,
        "components": components,
        "target_meta": dict(data.get("target_meta") or {}),
    }
    return bundle


# ── Stacking bundle ───────────────────────────────────────────

def fit_stacking_bundle(data: dict, params: dict) -> dict:
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.linear_model import LogisticRegression, RidgeClassifier

    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    close = np.asarray(data["close_prices"], dtype=float)
    sidx = int(data["split_idx"])
    n_te = len(X_test)
    n_folds = int(params.get("n_folds", 5))
    feats = list(data["feature_cols"])
    sma_s = int(params.get("sma_short", 10))
    sma_l = int(params.get("sma_long", 50))

    base = []
    if params.get("use_xgboost", True):
        base.append("XGBoost")
    if params.get("use_rf", True):
        base.append("Random Forest")
    if params.get("use_mlp", True):
        base.append("MLP")
    use_ma = params.get("use_ma", True)
    all_names = base[:]
    if use_ma:
        all_names.append("MA")
    if not all_names:
        raise ValueError("En az bir temel model secin")

    n_tr = len(X_train)
    tscv = TimeSeriesSplit(n_splits=n_folds)
    oof = {name: np.full(n_tr, np.nan) for name in all_names}

    for tr_idx, val_idx in tscv.split(X_train):
        Xf_tr, Xf_val = X_train[tr_idx], X_train[val_idx]
        yf_tr = y_train[tr_idx]
        for name in base:
            m, sc = _fit_ml(name, Xf_tr, yf_tr)
            if sc is None:
                oof[name][val_idx] = m.predict(Xf_val)
            else:
                oof[name][val_idx] = m.predict(sc.transform(Xf_val))
        if use_ma:
            sig = _ma_signal_full(close, sma_s, sma_l)
            oof["MA"][val_idx] = sig[val_idx]

    meta_full = np.column_stack([oof[n] for n in all_names])
    valid = ~np.isnan(meta_full).any(axis=1)
    meta_train = meta_full[valid]
    y_meta = y_train[valid]

    ml_type = params.get("meta_learner", "Logistic Regression")
    if ml_type == "Ridge Classifier":
        meta = RidgeClassifier(random_state=42)
    else:
        meta = LogisticRegression(random_state=42, max_iter=1000)
    meta.fit(meta_train, y_meta)

    components = {}
    for name in base:
        m, sc = _fit_ml(name, X_train, y_train)
        components[name] = {"model": m, "scaler": sc, "type": "sklearn"}
    if use_ma:
        components["MA"] = {"type": "ma", "short": sma_s, "long": sma_l}

    bundle = {
        "kind": "stacking",
        "feature_cols": feats,
        "base_names": all_names,
        "components": components,
        "meta": meta,
        "meta_type": ml_type,
        "target_meta": dict(data.get("target_meta") or {}),
    }
    return bundle


def _row_feats(df, feats):
    last = df.iloc[[-1]]
    if last[feats].isnull().any().any():
        return None, "Son satirda NaN var"
    return last[feats].values.astype(np.float64), None


def predict_voting_bundle(bundle: dict, csv_path: str) -> dict:
    df = load_data(csv_path)
    feats = bundle["feature_cols"]
    miss = set(feats) - set(df.columns)
    if miss:
        return {"ok": False, "error": f"Eksik sutunlar: {sorted(miss)[:8]}"}
    close = df["close"].astype(float).values
    Xr, err = _row_feats(df, feats)
    if Xr is None:
        return {"ok": False, "error": err}
    last_i = len(df) - 1
    # w_vote: agirlikli sinif oylari (0-1); hepsi 1 ise 1.0 olur — bu P(yukari) degil.
    w_vote = 0.0
    # w_proba: agirlikli ortalama P(yukari); arayuzde gosterilecek deger
    w_proba = 0.0
    details = []

    for name, w in bundle["weights"].items():
        comp = bundle["components"][name]
        if comp["type"] == "sklearn":
            p = _ml_predict_row(name, comp["model"], comp["scaler"], Xr)
            pu = _ml_predict_proba_up(name, comp["model"], comp["scaler"], Xr)
        elif comp["type"] == "ma":
            sig = _ma_signal_full(close, comp["short"], comp["long"])
            p = int(sig[last_i])
            pu = float(p)
        elif comp["type"] == "arima":
            p = _arima_direction_next(close, comp.get("max_train", 5000))
            pu = float(p)
        else:
            continue
        w_vote += w * p
        w_proba += w * pu
        details.append(f"  {name}: oyu={p}  P(yukari)~{pu:.3f}  agirlik={w:.3f}")

    pred = int(w_vote > 0.5)
    return {
        "ok": True,
        "open_time": df["open_time"].iloc[-1],
        "close": float(df["close"].iloc[-1]),
        "prediction": pred,
        "p_yukari": float(w_proba),
        "vote_fraction": float(w_vote),
        "label": "Yukarı" if pred == 1 else "Aşağı",
        "kind": "voting",
        "detail_lines": "\n".join(details),
    }


def predict_stacking_bundle(bundle: dict, csv_path: str) -> dict:
    df = load_data(csv_path)
    feats = bundle["feature_cols"]
    miss = set(feats) - set(df.columns)
    if miss:
        return {"ok": False, "error": f"Eksik sutunlar: {sorted(miss)[:8]}"}
    close = df["close"].astype(float).values
    Xr, err = _row_feats(df, feats)
    if Xr is None:
        return {"ok": False, "error": err}
    last_i = len(df) - 1
    cols = []
    details = []
    for name in bundle["base_names"]:
        comp = bundle["components"][name]
        if comp["type"] == "sklearn":
            p = _ml_predict_row(name, comp["model"], comp["scaler"], Xr)
            pu = _ml_predict_proba_up(name, comp["model"], comp["scaler"], Xr)
        else:
            sig = _ma_signal_full(close, comp["short"], comp["long"])
            p = int(sig[last_i])
            pu = float(p)
        cols.append(p)
        details.append(f"  {name}: sinif={p}  P(yukari)~{pu:.3f}")

    meta = bundle["meta"]
    row = np.array([cols], dtype=float)
    pred = int(meta.predict(row)[0])
    proba = None
    if hasattr(meta, "predict_proba"):
        pr = meta.predict_proba(row)[0]
        proba = float(pr[1]) if len(pr) > 1 else float(pr[0])
    else:
        proba = float(pred)

    return {
        "ok": True,
        "open_time": df["open_time"].iloc[-1],
        "close": float(df["close"].iloc[-1]),
        "prediction": pred,
        "p_yukari": proba,
        "label": "Yukarı" if pred == 1 else "Aşağı",
        "kind": "stacking",
        "meta_type": bundle.get("meta_type", ""),
        "detail_lines": "\n".join(details),
    }


def predict_any_bundle(bundle: dict, csv_path: str) -> dict:
    k = bundle.get("kind", "single")
    if k == "voting":
        return predict_voting_bundle(bundle, csv_path)
    if k == "stacking":
        return predict_stacking_bundle(bundle, csv_path)
    from .model_bundle import predict_last_row
    return predict_last_row(bundle, csv_path)
