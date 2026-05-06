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
    tavily_include_domains_raw: str = Field(
        default="gov.mk,mvr.gov.mk,ujp.gov.mk,mtsp.gov.mk,mioa.gov.mk,uslugi.gov.mk,katastar.gov.mk,fzo.org.mk,av.gov.mk",
        alias="TAVILY_INCLUDE_DOMAINS",
    )
    tavily_strict_sources: bool = Field(default=True, alias="TAVILY_STRICT_SOURCES")

    @property
    def database_url(self) -> str:
        return self.sqlite_url if self.use_sqlite else self.postgresql_url

    @property
    def tavily_include_domains(self) -> list[str]:
        domains: list[str] = []
        for raw in self.tavily_include_domains_raw.split(","):
            domain = raw.strip().lower()
            if domain:
                domains.append(domain)
        return domains

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", populate_by_name=True,
                                      extra="ignore", )


settings = Settings()