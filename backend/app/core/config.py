from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    postgresql_url: str = "postgresql+psycopg2://postgres:postgres@localhost:5432/ai_administration"
    sqlite_url: str = "sqlite:///./app.db"
    use_sqlite: bool = False

    @property
    def database_url(self) -> str:
        return self.sqlite_url if self.use_sqlite else self.postgresql_url

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()