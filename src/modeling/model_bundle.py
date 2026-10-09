"""
Egitilmis modeli disa aktarma ve CSV uzerinden guncel tahmin.
"""
import os
import numpy as np
import joblib

from .utils import load_data


def save_bundle(path, bundle):
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    joblib.dump(bundle, path)


def load_bundle(path):
    return joblib.load(path)


def predict_last_row(bundle, csv_path):
    """
    csv_path: ozellik sutunlari iceren veri (egitimle ayni format).
    Son mumun ozellikleriyle bir sonraki donem yon tahmini.
    """
    try:
        from sklearn.pipeline import Pipeline
    except ImportError:
        Pipeline = ()

    df = load_data(csv_path)
    feats = bundle.get("feature_cols") or []
    missing = set(feats) - set(df.columns)
    if missing:
        return {"ok": False, "error": f"CSV'de eksik sutunlar: {sorted(missing)[:12]}"}

    last = df.iloc[[-1]]
    if last[feats].isnull().any().any():
        return {"ok": False, "error": "Son satirda NaN var; veriyi guncelleyin."}

    model = bundle["model"]
    scaler = bundle.get("scaler")
    prob_thr = bundle.get("prob_threshold")

    if isinstance(model, Pipeline):
        X_for_pred = last[feats]
    elif scaler is not None:
        X_for_pred = scaler.transform(last[feats].values.astype(np.float64))
    else:
        X_for_pred = last[feats].values.astype(np.float64)

    proba = model.predict_proba(X_for_pred)[0]
    p_up = float(proba[1]) if len(proba) > 1 else float(proba[0])

    if prob_thr is not None:
        pred = int(p_up >= prob_thr)
    else:
        pred = int(model.predict(X_for_pred)[0])

    ts = last["open_time"].iloc[0] if "open_time" in last.columns else None
    close_val = float(last["close"].iloc[0]) if "close" in last.columns else None

    return {
        "ok": True,
        "open_time": ts,
        "close": close_val,
        "prediction": pred,
        "p_yukari": p_up,
        "label": "Yukarı" if pred == 1 else "Aşağı",
        "prob_threshold": prob_thr,
    }
