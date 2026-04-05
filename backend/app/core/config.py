from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    postgresql_url: str = "postgresql+psycopg2://postgres:postgres@localhost:5432/ai_administration"
    sqlite_url: str = "sqlite:///./app.db"
    use_sqlite: bool = False

    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-1.5-flash", alias="GEMINI_MODEL")

    @property
    def database_url(self) -> str:
        return self.sqlite_url if self.use_sqlite else self.postgresql_url

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", populate_by_name=True,
                                      extra="ignore", )


settings = Settings()