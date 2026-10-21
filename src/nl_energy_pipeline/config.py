from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_DATABASE_URL = "postgresql+psycopg://nl_energy:nl_energy@localhost:5432/nl_energy"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Only the ingestion job needs the token; the API and tests must import settings without it.
    entsoe_api_token: str | None = None
    database_url: str = DEFAULT_DATABASE_URL
    artifact_dir: Path = Path("artifacts")

    @field_validator("database_url")
    @classmethod
    def _use_psycopg3_driver(cls, url: str) -> str:
        # Hosted Postgres (Railway, Azure, ...) hands out plain postgres:// or postgresql:// URLs.
        for prefix in ("postgres://", "postgresql://"):
            if url.startswith(prefix):
                return "postgresql+psycopg://" + url[len(prefix) :]
        return url


settings = Settings()
