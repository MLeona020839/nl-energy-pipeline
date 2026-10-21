import pandas as pd

from nl_energy_pipeline import cli


def test_ingest_days_shortcut_resolves_a_utc_window(monkeypatch):
    seen = {}
    monkeypatch.setattr(cli, "get_engine", lambda: "engine")
    monkeypatch.setattr(
        cli,
        "ingest",
        lambda start, end, engine, save_parquet: seen.update(start=start, end=end, p=save_parquet),
    )

    cli.main(["ingest", "--days", "3", "--no-parquet"])

    assert seen["end"] - seen["start"] >= pd.Timedelta(days=3)
    assert str(seen["start"].tz) == "UTC" and seen["start"] == seen["start"].floor("D")
    assert seen["p"] is False


def test_explicit_start_and_end_are_parsed_as_utc(monkeypatch):
    seen = {}
    monkeypatch.setattr(cli, "get_engine", lambda: "engine")
    monkeypatch.setattr(
        cli, "ingest", lambda start, end, engine, save_parquet: seen.update(start=start, end=end)
    )

    cli.main(["ingest", "--start", "2024-01-01", "--end", "2024-02-01"])

    assert seen["start"] == pd.Timestamp("2024-01-01", tz="UTC")
    assert seen["end"] == pd.Timestamp("2024-02-01", tz="UTC")


def test_train_command_dispatches_to_training(monkeypatch):
    called = []
    monkeypatch.setattr(cli, "get_engine", lambda: "engine")
    monkeypatch.setattr(
        "nl_energy_pipeline.models.train.train", lambda engine: called.append(engine)
    )

    cli.main(["train"])

    assert called == ["engine"]
