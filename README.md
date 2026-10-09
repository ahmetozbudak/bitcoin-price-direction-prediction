# Bitcoin Price Direction Prediction

A machine-learning project for predicting the **direction of the next 4-hour Bitcoin candle** from historical OHLCV data and engineered market features. The project was developed as a Computer Engineering graduation project and received **2nd place** in CENGA's 8th Electrical, Electronics and Computer Engineering Project Competition (2026).

## What the project does

The pipeline cleans 4-hour BTC market data, generates scale-aware technical and market-behavior features, trains several classification models, and compares them with hybrid ensemble approaches. The reproducible processed dataset contains 17,919 rows, 24 model features, and spans January 2018 to March 2026 after feature generation. A PyQt5 desktop interface is included for model training/evaluation and a second interface can load exported model bundles for prediction.

### Main feature groups

- RSI and RSI change
- MACD histogram
- Bollinger Band position
- Stochastic oscillator
- Normalized ATR and volatility features
- Momentum and rate-of-change features
- Volume ratio/change
- Candlestick body/shadow ratios
- Lagged returns and direction features

### Models and approaches

- XGBoost
- Random Forest
- Multi-Layer Perceptron (MLP)
- SVM (available in the application)
- Moving-average trend signals
- ARIMA (optional; disabled in the default ensemble)
- Feature augmentation
- Validation-weighted voting
- Time-series stacking

## Reproducible benchmark

The GitHub-cleaned version uses one consistent 4H data pipeline and a chronological 80/20 train/test split with no shuffling. The target is binary: `1` if the next candle closes higher than the current candle, otherwise `0`.

| Model | Accuracy | Weighted F1 |
|---|---:|---:|
| XGBoost | **52.79%** | **0.5269** |
| Random Forest | 52.54% | 0.5241 |
| MLP | 51.90% | 0.5190 |
| Hybrid Feature Augmentation | 52.51% | 0.5242 |
| Weighted Voting | 52.06% | 0.5201 |
| Stacking | 52.54% | 0.5241 |

The positive-class share in the test period is approximately **51.00%**, so these results should be interpreted as modest predictive performance on a difficult financial time-series task, not as a profitable trading strategy. **No trading-profit backtest is claimed in this repository.**

## Repository structure

```text
bitcoin-price-direction-prediction/
├── app.py                         # Training/evaluation desktop GUI
├── predict_app.py                 # Load a model bundle and predict
├── requirements.txt
├── data/
│   ├── raw/
│   ├── cleaned/
│   └── processed/
├── src/
│   ├── preprocessing/
│   │   ├── data_cleaning.py
│   │   └── feature_engineering.py
│   └── modeling/
├── scripts/
│   ├── benchmark_models.py
│   └── export_ensemble.py
├── results/
└── docs/
```

## Installation

```bash
git clone <your-repository-url>
cd bitcoin-price-direction-prediction
python -m venv .venv
```

Activate the environment and install dependencies:

```bash
pip install -r requirements.txt
```

## Rebuild the data pipeline

Clean the included raw 4H dataset:

```bash
python -m src.preprocessing.data_cleaning
```

Generate the v2 feature set:

```bash
python -m src.preprocessing.feature_engineering
```

## Reproduce model results

```bash
python scripts/benchmark_models.py
```

The script writes the benchmark table to `results/model_comparison.csv`.

## Desktop applications

Training and evaluation interface:

```bash
python app.py
```

Prediction interface for exported model bundles:

```bash
python predict_app.py
```

To export an ensemble model bundle:

```bash
python scripts/export_ensemble.py voting \
  --csv data/processed/btc_4h_features.csv \
  --out artifacts/voting.joblib
```

## Project notes

This repository is a cleaned, reproducible version of the original graduation-project workspace. Development-only files, IDE metadata, duplicate datasets, temporary exports, stale reports, large pre-trained bundles, and personal-document artifacts were intentionally removed. The final repository focuses on the 4H v2 pipeline used for reproducible demonstration.

The project is provided for educational and portfolio purposes and does not constitute financial advice.

## Author

**Ahmet Özbudak**  
Computer Engineering graduate, Çukurova University
