import argparse
import logging

import pandas as pd

from nl_energy_pipeline.config import settings
from nl_energy_pipeline.db.session import get_engine
from nl_energy_pipeline.ingestion.run import ingest


def _ts(value: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz="UTC")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nl-energy-pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="fetch ENTSO-E + Open-Meteo data into Postgres")
    p_ingest.add_argument("--start", type=_ts, help="UTC date, e.g. 2024-01-01")
    p_ingest.add_argument("--end", type=_ts, help="UTC date (exclusive); default: now")
    p_ingest.add_argument(
        "--days",
        type=int,
        default=settings.ingest_days,
        help="shortcut: last N days when --start is absent (default: $INGEST_DAYS or 7)",
    )
    p_ingest.add_argument("--no-parquet", action="store_true", help="skip raw parquet landing")

    sub.add_parser(
        "train", help="train the price model, store out-of-sample predictions + anomalies"
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = build_parser().parse_args(argv)
    engine = get_engine()

    if args.command == "ingest":
        end = args.end or pd.Timestamp.now(tz="UTC").floor("h")
        start = args.start or (end - pd.Timedelta(days=args.days)).floor("D")
        totals = ingest(
            start, end, engine, save_parquet=settings.save_parquet and not args.no_parquet
        )
        logging.getLogger(__name__).info("done: %s", totals)
    elif args.command == "train":
        from nl_energy_pipeline.models.train import train

        train(engine)


if __name__ == "__main__":
    main()
