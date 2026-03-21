from functools import lru_cache
from typing import Literal

from pydantic import Field, computed_field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized runtime settings with production-safe validation."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_env: Literal["development", "staging", "production"] = Field(
        default="development",
        alias="APP_ENV",
    )
    deployment_mode: Literal["polling", "webhook"] = Field(
        default="polling",
        alias="DEPLOYMENT_MODE",
    )
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    telegram_bot_token: str = Field(default="", alias="TELEGRAM_BOT_TOKEN")
    telegram_webhook_base_url: str | None = Field(
        default=None,
        alias="TELEGRAM_WEBHOOK_BASE_URL",
    )
    telegram_webhook_path: str = Field(
        default="/telegram/webhook",
        alias="TELEGRAM_WEBHOOK_PATH",
    )
    telegram_webhook_secret: str | None = Field(
        default=None,
        alias="TELEGRAM_WEBHOOK_SECRET",
    )
    auto_set_webhook: bool = Field(default=False, alias="AUTO_SET_WEBHOOK")

    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    groq_text_model: str = Field(default="llama-3.1-8b-instant", alias="GROQ_TEXT_MODEL")
    groq_vision_model: str = Field(
        default="llama-3.2-11b-vision-preview",
        alias="GROQ_VISION_MODEL",
    )

    data_dir: str = Field(default="data", alias="DATA_DIR")
    local_db_dir: str = Field(default="db", alias="LOCAL_DB_DIR")

    vector_backend: Literal["qdrant"] = Field(default="qdrant", alias="VECTOR_BACKEND")
    qdrant_url: str | None = Field(default=None, alias="QDRANT_URL")
    qdrant_api_key: str | None = Field(default=None, alias="QDRANT_API_KEY")
    qdrant_local_path: str = Field(default="db/qdrant", alias="QDRANT_LOCAL_PATH")
    knowledge_collection_name: str = Field(
        default="knowledge_base",
        alias="KNOWLEDGE_COLLECTION_NAME",
    )
    cache_collection_name: str = Field(
        default="semantic_cache",
        alias="CACHE_COLLECTION_NAME",
    )

    embedding_model: str = Field(default="all-MiniLM-L6-v2", alias="EMBEDDING_MODEL")
    embedding_vector_size: int = Field(default=384, alias="EMBEDDING_VECTOR_SIZE")

    chunk_size: int = Field(default=900, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=120, alias="CHUNK_OVERLAP")
    top_k_retrieval: int = Field(default=4, alias="TOP_K_RETRIEVAL")
    cache_similarity_threshold: float = Field(
        default=0.92,
        alias="CACHE_SIMILARITY_THRESHOLD",
    )
    min_relevance_score: float = Field(default=0.35, alias="MIN_RELEVANCE_SCORE")
    max_history_messages: int = Field(default=6, alias="MAX_HISTORY_MESSAGES")
    max_context_chars: int = Field(default=4000, alias="MAX_CONTEXT_CHARS")

    request_timeout_seconds: int = Field(default=30, alias="REQUEST_TIMEOUT_SECONDS")

    @computed_field
    @property
    def telegram_webhook_url(self) -> str | None:
        if not self.telegram_webhook_base_url:
            return None
        return f"{self.telegram_webhook_base_url.rstrip('/')}{self.telegram_webhook_path}"

    @computed_field
    @property
    def use_cloud_vector_store(self) -> bool:
        return bool(self.qdrant_url)

    @model_validator(mode="after")
    def validate_runtime_requirements(self) -> "Settings":
        if not self.telegram_bot_token:
            raise ValueError("TELEGRAM_BOT_TOKEN is required.")
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY is required.")

        if self.deployment_mode == "webhook":
            if not self.telegram_webhook_base_url:
                raise ValueError(
                    "TELEGRAM_WEBHOOK_BASE_URL is required when DEPLOYMENT_MODE=webhook."
                )
            if not self.telegram_webhook_secret:
                raise ValueError(
                    "TELEGRAM_WEBHOOK_SECRET is required when DEPLOYMENT_MODE=webhook."
                )

        if self.app_env == "production" and not self.qdrant_url:
            raise ValueError(
                "QDRANT_URL is required in production so vector storage is not tied to local disk."
            )

        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
