"""
Oylama veya stacking paketini (.joblib) uretir.
Ornek:
  python scripts/export_ensemble.py voting --csv data/processed/btc_4h_features.csv --out artifacts/voting.joblib
  python scripts/export_ensemble.py stacking --csv data/processed/btc_4h_features.csv --out artifacts/stacking.joblib
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.modeling.utils import load_data, prepare_data, split_data
from src.modeling.model_bundle import save_bundle
from src.modeling.ensemble_bundle import fit_voting_bundle, fit_stacking_bundle


def main():
    p = argparse.ArgumentParser(description="Ensemble joblib uret")
    p.add_argument("mode", choices=["voting", "stacking"])
    p.add_argument("--csv", required=True, help="Ozellikli veri CSV (v1/v2)")
    p.add_argument("--out", required=True, help="Cikti .joblib yolu")
    p.add_argument("--target", default="next_bar", choices=["next_bar", "threshold_next", "horizon_k"])
    p.add_argument("--thr", type=float, default=0.002, help="threshold_next icin oran")
    p.add_argument("--horizon", type=int, default=3)
    args = p.parse_args()

    df = load_data(args.csv)
    X, y, feat_cols, close, tmeta = prepare_data(
        df,
        target_mode=args.target,
        threshold_pct=args.thr,
        horizon=args.horizon,
    )
    Xtr, Xte, ytr, yte, sidx = split_data(X, y)
    data = {
        "X_train": Xtr,
        "X_test": Xte,
        "y_train": ytr,
        "y_test": yte,
        "close_prices": close,
        "split_idx": sidx,
        "feature_cols": feat_cols,
        "target_meta": tmeta,
    }
    params = {
        "use_xgboost": True,
        "use_rf": True,
        "use_mlp": True,
        "use_svm": False,
        "use_ma": True,
        "use_arima": False,
        "val_ratio": 0.15,
        "sma_short": 10,
        "sma_long": 50,
        "n_folds": 5,
        "meta_learner": "Logistic Regression",
    }
    if args.mode == "voting":
        bundle = fit_voting_bundle(data, params)
    else:
        bundle = fit_stacking_bundle(data, params)

    save_bundle(args.out, bundle)
    print(f"Kaydedildi: {args.out}")
    print(f"Tur: {bundle['kind']}  Ozellik: {len(bundle['feature_cols'])}")


if __name__ == "__main__":
    main()
