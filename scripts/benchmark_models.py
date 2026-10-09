# -*- coding: utf-8 -*-
"""Reproduce the main 4H v2 benchmark and save metrics to results/."""
from __future__ import annotations

from pathlib import Path
import sys
import csv
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from src.modeling.utils import load_data, prepare_data, split_data, calc_metrics, scale_data
from src.modeling.hybrid_models import run_feature_augmentation, run_weighted_voting, run_stacking

DATA_PATH = ROOT / "data" / "processed" / "btc_4h_features.csv"
RESULT_PATH = ROOT / "results" / "model_comparison.csv"


def main() -> None:
    df = load_data(DATA_PATH)
    X, y, feature_cols, close_prices, target_meta = prepare_data(df)
    X_train, X_test, y_train, y_test, split_idx = split_data(X, y)
    data = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "close_prices": close_prices,
        "split_idx": split_idx,
        "feature_cols": feature_cols,
        "target_meta": target_meta,
    }

    rows = []
    def record(name, pred):
        m = calc_metrics(y_test, pred, name)
        rows.append({
            "model": name,
            "accuracy": m["accuracy"],
            "precision_weighted": m["precision"],
            "recall_weighted": m["recall"],
            "f1_weighted": m["f1"],
        })

    xgb = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        random_state=42, eval_metric="logloss", verbosity=0,
    )
    xgb.fit(X_train, y_train)
    record("XGBoost", xgb.predict(X_test))

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=42, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    record("Random Forest", rf.predict(X_test))

    Xtr_s, Xte_s = scale_data(X_train, X_test)
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32), random_state=42,
        max_iter=300, early_stopping=True, verbose=False,
    )
    mlp.fit(Xtr_s, y_train)
    record("MLP", mlp.predict(Xte_s))

    hybrid_runs = [
        ("Hybrid Feature Augmentation", run_feature_augmentation, {
            "ml_model": "XGBoost", "sma_short": 10, "sma_long": 50,
            "n_estimators": 200, "max_depth": 6, "learning_rate": 0.1,
        }),
        ("Weighted Voting", run_weighted_voting, {
            "use_xgboost": True, "use_rf": True, "use_mlp": True,
            "use_svm": False, "use_ma": True, "use_arima": False,
            "val_ratio": 0.15, "sma_short": 10, "sma_long": 50,
        }),
        ("Stacking", run_stacking, {
            "use_xgboost": True, "use_rf": True, "use_mlp": True,
            "use_ma": True, "n_folds": 5,
            "meta_learner": "Logistic Regression",
        }),
    ]
    for name, fn, params in hybrid_runs:
        m = fn(data, params)
        rows.append({
            "model": name,
            "accuracy": m["accuracy"],
            "precision_weighted": m["precision"],
            "recall_weighted": m["recall"],
            "f1_weighted": m["f1"],
        })

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)

    print(f"Dataset rows: {len(df):,}")
    print(f"Features: {len(feature_cols)}")
    print(f"Test up-class ratio: {np.mean(y_test):.4f}")
    for r in rows:
        print(f"{r['model']:<28} Acc={r['accuracy']:.4f}  F1={r['f1_weighted']:.4f}")
    print(f"Saved: {RESULT_PATH}")


if __name__ == "__main__":
    main()
