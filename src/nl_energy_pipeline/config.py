from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    entsoe_api_token: str
    database_url: str = "postgresql+psycopg://nl_energy:nl_energy@localhost:5432/nl_energy"


settings = Settings()
