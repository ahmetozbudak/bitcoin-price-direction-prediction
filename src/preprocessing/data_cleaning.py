# -*- coding: utf-8 -*-
"""Clean OHLCV data before feature engineering.

Example:
    python -m src.preprocessing.data_cleaning \
        --input data/raw/btc_4h_raw.csv \
        --output data/cleaned/btc_4h_cleaned.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def clean_dataset(input_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    df.columns = (
        df.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)
    )

    if "ignore" in df.columns:
        df = df.drop(columns=["ignore"])

    if "open_time" in df.columns:
        df["open_time"] = pd.to_datetime(
            df["open_time"].astype(str).str.strip(), errors="coerce"
        )
        df = df.sort_values("open_time").reset_index(drop=True)

    df = df.drop_duplicates().reset_index(drop=True)
    if "open_time" in df.columns:
        df = df.drop_duplicates(subset=["open_time"]).reset_index(drop=True)

    numeric_cols = [
        "open", "high", "low", "close", "volume", "quote_asset_volume",
        "number_of_trades", "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    required_ohlcv = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    df = df.dropna(subset=required_ohlcv).reset_index(drop=True)
    if "open_time" in df.columns:
        df = df.dropna(subset=["open_time"]).reset_index(drop=True)

    checks = [
        ("high", "low", lambda a, b: a >= b),
        ("high", "open", lambda a, b: a >= b),
        ("high", "close", lambda a, b: a >= b),
        ("low", "open", lambda a, b: a <= b),
        ("low", "close", lambda a, b: a <= b),
    ]
    for a, b, fn in checks:
        if a in df.columns and b in df.columns:
            df = df[fn(df[a], df[b])].reset_index(drop=True)

    if "volume" in df.columns:
        df = df[df["volume"] >= 0].reset_index(drop=True)
    if "number_of_trades" in df.columns:
        df = df[df["number_of_trades"] >= 0].reset_index(drop=True)

    return df


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Clean BTC OHLCV CSV data")
    parser.add_argument("--input", default=str(project_root / "data" / "raw" / "btc_4h_raw.csv"))
    parser.add_argument("--output", default=str(project_root / "data" / "cleaned" / "btc_4h_cleaned.csv"))
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df = clean_dataset(args.input)
    df.to_csv(output, index=False)
    print(f"Saved {len(df):,} cleaned rows to: {output}")


if __name__ == "__main__":
    main()
