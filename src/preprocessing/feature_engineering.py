"""
Feature Engineering V2 - Enhanced Feature Set

Key improvements over v1:
  - All price-dependent features normalized (scale-free)
  - MACD histogram for momentum shift detection
  - Bollinger Band %B for volatility-relative position
  - Stochastic Oscillator for overbought/oversold
  - Volume-based features (ratio, change)
  - Lag features for temporal context (returns, directions)
  - RSI dynamics (rate of change)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "cleaned" / "btc_4h_cleaned.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "btc_4h_features.csv"

def _rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def add_features(df):
    c = df["close"].astype(float)
    h = df["high"].astype(float)
    lo = df["low"].astype(float)
    o = df["open"].astype(float)
    v = df["volume"].astype(float)

    # ────────────────────────────────────────────
    # 1. RSI (0-100, already normalized)
    # ────────────────────────────────────────────
    df["rsi_14"] = _rsi(c, 14)
    df["rsi_change"] = df["rsi_14"].diff()

    # ────────────────────────────────────────────
    # 2. MACD Histogram (normalized by close)
    # ────────────────────────────────────────────
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    df["macd_hist"] = (macd_line - macd_signal) / c

    # ────────────────────────────────────────────
    # 3. Bollinger Band %B (0-1 range)
    # ────────────────────────────────────────────
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    bb_width = bb_upper - bb_lower
    df["bb_position"] = (c - bb_lower) / bb_width.replace(0, np.nan)

    # ────────────────────────────────────────────
    # 4. Stochastic Oscillator (0-100)
    # ────────────────────────────────────────────
    low14 = lo.rolling(14).min()
    high14 = h.rolling(14).max()
    k_denom = (high14 - low14).replace(0, np.nan)
    df["stoch_k"] = (c - low14) / k_denom * 100
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    # ────────────────────────────────────────────
    # 5. ATR (normalized by close)
    # ────────────────────────────────────────────
    tr = pd.concat([
        h - lo,
        (h - c.shift(1)).abs(),
        (lo - c.shift(1)).abs(),
    ], axis=1).max(axis=1)
    df["atr_pct"] = tr.rolling(14).mean() / c

    # ────────────────────────────────────────────
    # 6. Volatility measures
    # ────────────────────────────────────────────
    log_ret = np.log(c / c.shift(1))
    df["volatility_20"] = log_ret.rolling(20).std()
    df["high_low_range"] = (h - lo) / c
    df["vol_regime"] = (
        df["volatility_20"] / df["volatility_20"].rolling(20).mean()
    ).clip(0, 5)

    # ────────────────────────────────────────────
    # 7. Momentum (all percentage-based)
    # ────────────────────────────────────────────
    df["momentum_pct"] = (c - c.shift(10)) / c.shift(10)
    df["roc_5"] = (c - c.shift(5)) / c.shift(5)

    # ────────────────────────────────────────────
    # 8. Volume features (normalized)
    # ────────────────────────────────────────────
    vol_ma = v.rolling(20).mean().replace(0, np.nan)
    df["volume_ratio"] = (v / vol_ma).clip(0, 10)
    df["volume_change"] = v.pct_change(5).clip(-5, 5)

    # ────────────────────────────────────────────
    # 9. Candlestick features (normalized)
    # ────────────────────────────────────────────
    df["candle_body_pct"] = (c - o) / c
    total_range = (h - lo).replace(0, np.nan)
    df["candle_body_ratio"] = (c - o).abs() / total_range
    oc_max = pd.concat([o, c], axis=1).max(axis=1)
    oc_min = pd.concat([o, c], axis=1).min(axis=1)
    df["upper_shadow_ratio"] = (h - oc_max) / total_range
    df["lower_shadow_ratio"] = (oc_min - lo) / total_range
    for col in ["candle_body_ratio", "upper_shadow_ratio", "lower_shadow_ratio"]:
        df[col] = df[col].fillna(0)

    # ────────────────────────────────────────────
    # 10. Lag features (temporal context)
    # ────────────────────────────────────────────
    for lag in [1, 2, 3]:
        df[f"return_{lag}"] = c.pct_change(lag)
        df[f"direction_{lag}"] = (c > c.shift(lag)).astype(float)

    return df


def process_file(label, input_path, output_path):
    print(f"\n[{label}] Yukleniyor: {input_path}")
    df = pd.read_csv(input_path, parse_dates=["open_time"])
    df = df.sort_values("open_time").reset_index(drop=True)

    rows_before = len(df)
    cols_before = len(df.columns)

    print(f"[{label}] v2 Feature engineering uygulaniyor...")
    df = add_features(df)

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna().reset_index(drop=True)

    rows_after = len(df)
    cols_after = len(df.columns)

    non_feature = {
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume",
    }
    feature_cols = [c for c in df.columns if c not in non_feature]

    tmin, tmax = df["open_time"].min(), df["open_time"].max()
    df.to_csv(output_path, index=False)

    print(f"[{label}] Satirlar : {rows_before} -> {rows_after}"
          f" ({rows_before - rows_after} silindi)")
    print(f"[{label}] Feature  : {len(feature_cols)}")
    print(f"[{label}] open_time: {tmin}  ->  {tmax}")
    print(f"[{label}] Kaydedildi: {output_path}")

    return {
        "rows_before": rows_before,
        "rows_after": rows_after,
        "cols_before": cols_before,
        "cols_after": cols_after,
        "features": feature_cols,
        "open_time_min": tmin,
        "open_time_max": tmax,
    }



def main() -> None:
    parser = argparse.ArgumentParser(description="Generate v2 BTC features")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    info = process_file("4H", args.input, str(output))
    print(f"Generated {len(info['features'])} features; output rows: {info['rows_after']:,}")


if __name__ == "__main__":
    main()
