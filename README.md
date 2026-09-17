# nl-energy-pipeline

Engineering-grade data pipeline on public Dutch energy data (ENTSO-E + Open-Meteo):
ingest → PostgreSQL → cleaning/feature engineering → forecasting/anomaly detection →
Docker → GitHub Actions CI → cloud deployment.

## Setup

```bash
uv sync
cp .env.example .env   # fill in ENTSOE_API_TOKEN
uv run pre-commit install
```

## Dev commands

```bash
uv run ruff check .        # lint
uv run ruff format .       # format
uv run pytest              # tests
```

## Project layout

```
src/nl_energy_pipeline/
  ingestion/   # ENTSO-E + Open-Meteo clients
  db/          # SQLAlchemy models, Alembic migrations
  features/    # cleaning + feature engineering (pure functions)
  models/      # forecasting / anomaly detection
  api/         # FastAPI service
```
