# Interview notes

Private prep sheet: what the project does, how to describe it, and the questions it invites.
Numbers come from the real backfill (2024-01-01 to 2026-10-21).

## 30-second pitch

I built an end-to-end data pipeline on Dutch electricity-market data. It ingests prices and load from ENTSO-E
and weather from Open-Meteo into PostgreSQL with idempotent upserts, builds leakage-safe features, trains a
LightGBM day-ahead price forecaster with an expanding-window backtest, flags anomalous hours from the
out-of-sample residuals, and serves it all through FastAPI. It is containerised, tested in GitHub Actions
against a real Postgres, and re-ingests daily on a schedule.

## CV bullets (pick three or four)

- Built an end-to-end pipeline on public ENTSO-E and Open-Meteo data: month-by-month ingestion with retries into
  PostgreSQL (SQLAlchemy, Alembic migrations, `INSERT ... ON CONFLICT` upserts), ~175k raw rows over 2.8 years.
- Trained a LightGBM day-ahead price forecaster with leakage-tested features and an expanding-window backtest:
  22.6 EUR/MWh MAE, 17% better than a same-hour-yesterday baseline and 32% better than same-hour-last-week.
- Added residual-based anomaly detection (trailing robust z-score, no look-ahead) and a FastAPI service exposing
  prices, predictions, anomalies and model metrics.
- Packaged with a multi-stage, non-root Docker image and docker-compose; GitHub Actions CI runs ruff, pytest
  (44 tests, ~90% coverage) against a Postgres service container and builds the image; scheduled workflow ingests daily.
- *(Add only once it is actually deployed)* Deployed the API and database on Railway with health checks.

## SQL

- **Schema**: one table per source, primary key = UTC `timestamptz`. The timestamp is the natural key, so there is
  no surrogate id and the PK index serves every range query.
- **Idempotence**: `INSERT ... ON CONFLICT (ts) DO UPDATE`. Re-running a window, retrying after a crash, or the daily
  job re-fetching 3 days never duplicates rows, and late-published values overwrite earlier NULLs.
- **Edge cases handled**: duplicate timestamps inside one batch (Postgres refuses to update the same row twice in
  one statement, so they are de-duplicated first), NaN to NULL, and a batch size that keeps rows x columns under
  Postgres' 65,535 bind-parameter limit.
- **Other Postgres features used**: a composite index `(model_version, is_anomaly, ts)` for the anomaly endpoint, and
  `JSONB` for per-run metrics.
- **Migrations**: Alembic; tests and CI build the schema from the migrations, so the schema that ships is the one tested.

Likely questions: *Why timestamptz? Why no surrogate key? How do you make ingestion idempotent? What happens on a
DST fold? How would you partition this at 100x the size?* (Time-range partitioning by month on `ts`.)

## Docker

- Multi-stage build with `uv`: dependencies in their own layer (cached until `uv.lock` changes), runtime image has no
  build tools or dev dependencies, runs as a non-root user.
- `libgomp1` is installed explicitly because LightGBM needs OpenMP and the slim base image does not ship it.
- `HEALTHCHECK` hits `/health`, which does a real `SELECT 1`; compose uses it to gate dependent services.
- docker-compose: Postgres with a healthcheck, the API, and `ingest` / `train` as one-off jobs behind a profile.
- Migrations run on container start. Fine for one instance; with several replicas I would run them as a separate
  release step to avoid races.

Be upfront: I could not run Docker on my own machine (no Docker installed). The image is verified in CI (build plus an
import check), and I checked the runtime dependency set locally in a clean venv.

## CI/CD

- `ci.yml`: ruff lint + format check, pytest with a Postgres 16 service container and coverage, then a Docker build.
- The same ruff rules run in pre-commit, but a local hook can be skipped with `--no-verify`, so CI is the real gate.
- `ingest.yml`: cron daily, `concurrency` group so runs never overlap, secrets for the token and database URL,
  retrain on Mondays.
- Railway deploys the `Dockerfile` (`railway.json`) with `/health` as the health check; `DATABASE_URL` is normalised
  because hosted Postgres hands out `postgres://` URLs.

## Data and ML

- **Granularity**: ENTSO-E day-ahead prices moved to 15-minute resolution (a week of prices came back as 673 rows,
  7 x 96 + 1); load is 15-minute, weather hourly. Everything is averaged to a complete hourly UTC grid so
  `shift(24)` means exactly 24 hours.
- **Time zones**: store UTC; derive hour, weekday and holiday in `Europe/Amsterdam` (DST-aware).
- **Leakage**: the forecast is made at 10:00 UTC on D-1 for day D. Price lags >= 24 h, actual-load lags >= 48 h. A
  test tampers with the target hour and everything after it and asserts the features do not change.
- **Backtest**: expanding window, 24 h gap, compared with two naive baselines. Random k-fold would leak the future.
- **Anomalies**: residual vs. trailing 30-day median and MAD (x1.4826), threshold 4 sigma, computed from past values
  only so a spike cannot raise its own threshold.
- **Honest limitations**: weather in the backtest is observed, not forecast, so the error is optimistic; no
  live forecast endpoint; no model registry or drift monitoring.

## Stories that make good answers

- **A long backfill died on an ENTSO-E HTTP 599.** I split errors into transient (5xx, timeouts: retry with
  backoff) and permanent (401, bad parameters: fail immediately). Because each month is committed separately and
  writes are upserts, "resume" is just running the command again. Two more 599s later recovered on their own.
- **The API token showed up in error messages**, because ENTSO-E takes it as a query parameter and `requests`
  echoes the URL. Errors are now scrubbed before they are raised or logged, with a test.
- **The price data silently changed resolution.** I noticed from row counts, which is why resampling is part of
  the pipeline rather than an assumption.
- **A local module named `entsoe.py` shadowed the `entsoe` library** when Python was started from inside the
  package directory, giving a circular-import error. A good example of debugging with `sys.path` reasoning.
- **LightGBM would not load on macOS without `libomp`** (Homebrew was broken on this machine); fixed locally by pointing
  the library at the OpenMP runtime that ships with scikit-learn, and documented `brew install libomp` for others.

## Where to find things

| Topic | File |
|---|---|
| Schema and migrations | `src/nl_energy_pipeline/db/models.py`, `alembic/versions/` |
| Idempotent writes | `src/nl_energy_pipeline/db/repository.py` |
| Ingestion, retries, token scrubbing | `src/nl_energy_pipeline/ingestion/` |
| Features and leakage contract | `src/nl_energy_pipeline/features/transform.py` |
| Backtest, training, anomalies | `src/nl_energy_pipeline/models/` |
| API | `src/nl_energy_pipeline/api/main.py` |
| Docker, CI, deploy | `Dockerfile`, `docker-compose.yml`, `.github/workflows/`, `railway.json` |
