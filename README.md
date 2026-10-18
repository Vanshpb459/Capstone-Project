# Subscription Revenue Forecasting Service

A reproducible, containerized time-series forecasting service for subscription revenue across multiple countries. It ingests invoice data, builds daily/monthly country-level series and an `all` aggregate, compares forecasting models, versions trained artifacts, exposes a Flask API, logs predictions, and includes lightweight performance/drift monitoring.

> **Currency and accounting note:** the sample data uses a single synthetic reporting currency. If source countries bill in different currencies, convert amounts to a shared reporting currency before summing the `all` series. This example treats each row's `price`/mapped revenue field as a signed revenue amount; adjust for quantity, tax, discounts, refunds, invoice status, and revenue-recognition rules as needed.

## Features

- CSV, JSON, and JSONL ingestion with validation and daily/monthly aggregation.
- Per-country and `all` time series; missing periods inside the observed global date range are filled with zero.
- EDA charts saved to `eda/figures/`.
- Naive last-value baseline, Random Forest, Gradient Boosting, and Linear Regression compared on a chronological holdout using RMSE and MAE. Best ML model is selected by holdout RMSE, then MAE.
- Timestamped model artifacts with a `latest.json` pointer for each country.
- Flask endpoints: `POST /train`, `GET/POST /predict`, `GET /logs`, `POST /monitor`, and `GET /health`.
- CSV audit logs, performance evaluation, and a simple standardized mean-shift drift check.
- Unit tests use temporary directories and do not read/write production model or log paths.

## Repository layout

```text
subscription-revenue-forecast/
├── app.py                 # Flask API
├── data_ingestion.py      # CLI ingestion pipeline
├── model.py               # Training, evaluation, versioning, prediction
├── logger.py              # CSV logging
├── monitoring.py          # Performance/drift monitoring
├── run_tests.py           # Test runner
├── requirements.txt
├── Dockerfile
├── README.md
├── data/raw/invoices.csv  # Small synthetic sample
├── data/processed/        # Generated processed data
├── models/production/     # Generated production model artifacts
├── models/test/           # Isolated test artifacts
├── logs/production/       # Generated production logs
├── logs/test/             # Isolated test logs
├── eda/eda.py             # EDA chart generation
└── tests/
```

## Requirements

Python 3.10+ is recommended. Python 3.12 is used in Docker.

## Install and run locally

```bash
python3 -m venv .venv
source .venv/bin/activate       # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Build monthly series from the sample data:

```bash
python data_ingestion.py --input-dir data/raw --output-dir data/processed --frequency monthly
```

To build daily series instead:

```bash
python data_ingestion.py --input-dir data/raw --output-dir data/processed --frequency daily
```

If raw columns differ from `date`, `country`, and `price`, provide mappings:

```bash
python data_ingestion.py --date-column invoice_date --country-column billing_country --revenue-column amount
```

Generate exploratory plots:

```bash
python eda/eda.py --input data/processed/revenue_timeseries.csv --output-dir eda/figures
```

## Train models

In one terminal:

```bash
python app.py
```

In another terminal, train models using the processed CSV:

```bash
curl -X POST http://localhost:5000/train \
  -H 'Content-Type: application/json' \
  -d '{}'
```

You can train directly from Python as well:

```bash
python -c "import pandas as pd; from model import train_models; print(train_models(pd.read_csv('data/processed/revenue_timeseries.csv')))"
```

The `/train` endpoint accepts optional `random_state`, `test_size`, `data_path`, and `test` fields. For example, `{"test": true}` routes model artifacts and training logs to test directories. It is intended for isolated test/dev use, not production training.

## API examples

Forecast one country for a date after the last date in its training history:

```bash
curl 'http://localhost:5000/predict?country=India&date=2028-01-01'
```

Forecast all countries combined:

```bash
curl 'http://localhost:5000/predict?country=all&date=2028-01-01'
```

POST is also supported:

```bash
curl -X POST http://localhost:5000/predict \
  -H 'Content-Type: application/json' \
  -d '{"country":"USA","date":"2028-01-01"}'
```

View recent logs:

```bash
curl 'http://localhost:5000/logs?kind=all&limit=100'
curl 'http://localhost:5000/logs?kind=predictions&limit=20'
```

Performance monitoring using the model's holdout metrics:

```bash
curl -X POST http://localhost:5000/monitor \
  -H 'Content-Type: application/json' \
  -d '{"country":"all","threshold":1000}'
```

Example drift check:

```bash
curl -X POST http://localhost:5000/monitor \
  -H 'Content-Type: application/json' \
  -d '{"country":"all","reference_values":[100,110,105,98],"current_values":[250,260,255]}'
```

## Run tests

```bash
python run_tests.py
```

The tests use temporary model and log directories. They do not depend on existing production artifacts or production logs.

## Docker

Build the image:

```bash
docker build -t subscription-revenue-forecast:latest .
```

Run the API, mounting persistent local data/model/log folders:

```bash
docker run --rm -p 5000:5000 \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/logs:/app/logs" \
  subscription-revenue-forecast:latest
```

Then, in another terminal, run ingestion and training in the container (the container must be running):

```bash
docker exec "$(docker ps -q --filter ancestor=subscription-revenue-forecast:latest | head -n 1)" \
  python data_ingestion.py --input-dir data/raw --output-dir data/processed --frequency monthly

docker exec "$(docker ps -q --filter ancestor=subscription-revenue-forecast:latest | head -n 1)" \
  python -c "import pandas as pd; from model import train_models; train_models(pd.read_csv('data/processed/revenue_timeseries.csv'))"
```

API smoke tests:

```bash
curl -f http://localhost:5000/health
curl -f 'http://localhost:5000/predict?country=all&date=2028-01-01'
```

Run the test suite inside Docker:

```bash
docker run --rm subscription-revenue-forecast:latest python run_tests.py
```

## Model comparison summary

The training pipeline evaluates a naive last-observation baseline, Random Forest, Gradient Boosting, and Linear Regression using the same chronological holdout and reports RMSE and MAE. The best of the three machine-learning candidates is selected by holdout RMSE, with MAE as a tie-breaker, and refit on all available historical training examples. The exact winner and metrics depend on the data; they are returned by `POST /train` and saved in each model artifact. The naive baseline is retained for comparison and is not silently treated as the selected ML model.

This is a practical capstone baseline, not a substitute for production-grade time-series validation. For a business deployment, add rolling-origin backtesting, currency normalization, forecast intervals, reconciliation between country forecasts and the aggregate, model registry governance, and domain-specific revenue definitions.

## Final checklist and proof commands

| Requirement | Verification command / action |
|---|---|
| 1. Data ingestion | `python data_ingestion.py --input-dir data/raw --output-dir data/processed --frequency monthly` |
| 2. EDA | `python eda/eda.py --input data/processed/revenue_timeseries.csv --output-dir eda/figures` |
| 3. Model comparison | `curl -X POST localhost:5000/train -H 'Content-Type: application/json' -d '{}'` and inspect returned RMSE/MAE |
| 4. Baseline visualization | See `eda/figures/` for EDA plots; holdout model comparison is returned in artifacts. |
| 5. Versioned model module | `python -c "from model import load_model; print(load_model('all')['version'])"` after training |
| 6. Flask endpoints | `curl -f localhost:5000/health`; `curl 'localhost:5000/predict?country=all&date=2028-01-01'`; `curl localhost:5000/logs` |
| 7. CSV logging | `cat logs/production/training.csv logs/production/predictions.csv` after training/prediction |
| 8. Monitoring | `curl -X POST localhost:5000/monitor -H 'Content-Type: application/json' -d '{"country":"all","threshold":1000}'` |
| 9. Unit tests | `python run_tests.py` |
| 10. Test isolation | `python run_tests.py`; tests create temporary directories and the `/train` test flag uses `models/test`, `logs/test`. |
| 11. Single test runner | `python run_tests.py` |
| 12. Docker | `docker build -t subscription-revenue-forecast:latest .` then run the container and curl `/health`. |

## Publish to GitHub

Create a **public, empty** repository on GitHub named `subscription-revenue-forecast` (do not initialize it with a README if you already have this local README), then run:

```bash
git init
git add .
git commit -m "Build subscription revenue forecasting service"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/subscription-revenue-forecast.git
git push -u origin main
```

Replace `YOUR-USERNAME` with your GitHub username. Review `.gitignore` before pushing to avoid committing real customer data, model artifacts, secrets, or production logs.
