# settings.py
import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ───────────────────────────
    # Google Cloud
    # ───────────────────────────
    GOOGLE_PROJECT_ID: str = "macro-dreamer-459709-p4"
    GOOGLE_LOCATION: str = "asia-southeast1"
    GOOGLE_APPLICATION_CREDENTIALS: str = "credentials.json"
    GOOGLE_AI_STUDIO_API_KEY: str = ""

    # ───────────────────────────
    # Langfuse
    # ───────────────────────────
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_HOST: str = "https://cloud.langfuse.com"

    # ───────────────────────────
    # Model defaults
    # ───────────────────────────
    LLM_MODEL: str = "gemma-2-27b-it"
    IMAGE_MODEL: str = "imagen-3.0-generate-001"

    # ───────────────────────────
    # Cache
    # ───────────────────────────
    CACHE_TTL: int = 3600
    SIMILARITY_THRESHOLD: float = 0.85

    # ───────────────────────────
    # API server
    # ───────────────────────────
    API_HOST: str = "127.0.0.1"
    API_PORT: int = 8000

    # ───────────────────────────
    # PostgreSQL (used by TrendManager)
    # ───────────────────────────
    PG_HOST: str = "localhost"
    PG_PORT: int = 5432
    PG_DB: str = "trend_engine_db"
    PG_USER: str = "trend_user"
    PG_PASSWORD: str = "your_secure_password"

    # tell pydantic-settings **where** to look for overrides
    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"),
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )


settings = Settings()        # loads .env and os.environ at import time
