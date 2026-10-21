import pytest

from nl_energy_pipeline.config import Settings


@pytest.mark.parametrize(
    "given",
    [
        "postgres://u:p@host:5432/db",
        "postgresql://u:p@host:5432/db",
        "postgresql+psycopg://u:p@host:5432/db",
    ],
)
def test_database_url_is_normalised_for_the_psycopg3_driver(given):
    assert Settings(database_url=given).database_url == "postgresql+psycopg://u:p@host:5432/db"


def test_settings_load_without_an_entsoe_token(monkeypatch):
    monkeypatch.delenv("ENTSOE_API_TOKEN", raising=False)
    assert Settings(_env_file=None).entsoe_api_token is None
