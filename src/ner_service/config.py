from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = Field(default="ner_service", alias="APP_NAME")
    app_env: str = Field(default="dev", alias="APP_ENV")
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    ner_model_path: Path = Field(default=Path("artifacts/ner_model"), alias="NER_MODEL_PATH")
    ner_device: str = Field(default="cpu", alias="NER_DEVICE")
    ner_max_length: int = Field(default=256, alias="NER_MAX_LENGTH")
    ner_batch_size: int = Field(default=16, alias="NER_BATCH_SIZE")

    llm_mode: str = Field(default="stub", alias="LLM_MODE")
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    groq_model: str = Field(default="llama-3.3-70b-versatile", alias="GROQ_MODEL")
    groq_timeout: int = Field(default=60, alias="GROQ_TIMEOUT")
    groq_temperature: float = Field(default=0.0, alias="GROQ_TEMPERATURE")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()