# Data

This cleaned repository keeps a single internally consistent 4-hour pipeline.

- `raw/btc_4h_raw.csv` — 17,961 source rows from the original project archive. The final row has an invalid/missing timestamp and is removed by the cleaning pipeline.
- `cleaned/btc_4h_cleaned.csv` — 17,958 valid rows after standardization, duplicate/missing-value checks, numeric conversion and OHLCV consistency checks.
- `processed/btc_4h_features.csv` — 17,919 rows and 35 columns after v2 feature engineering; 24 columns are model features.

The processed feature data spans **2018-01-07 12:00:00 through 2026-03-15 12:00:00**.

Rebuild both derived files with:

```bash
python -m src.preprocessing.data_cleaning
python -m src.preprocessing.feature_engineering
```
