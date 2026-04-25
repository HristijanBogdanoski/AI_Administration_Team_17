from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, AliasChoices


class Settings(BaseSettings):
    postgresql_url: str = "postgresql+psycopg2://postgres:postgres@localhost:5433/ai_administration"
    sqlite_url: str = "sqlite:///./app.db"
    use_sqlite: bool = False

    gemini_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("GEMINI_API_KEY", "GEMINI_KEY"),
    )
    gemini_model: str = Field(default="gemini-1.5-flash", alias="GEMINI_MODEL")
    tavily_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("TAVILY_API_KEY", "TAVILY_KEY"),
    )
    tavily_max_results: int = Field(default=3, alias="TAVILY_MAX_RESULTS")
    tavily_search_depth: str = Field(default="basic", alias="TAVILY_SEARCH_DEPTH")

    @property
    def database_url(self) -> str:
        return self.sqlite_url if self.use_sqlite else self.postgresql_url

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", populate_by_name=True,
                                      extra="ignore", )


settings = Settings()