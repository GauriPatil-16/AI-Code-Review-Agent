from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BACKEND_ROOT / ".env", extra="ignore")
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-120b"
    github_token: str = ""
    github_webhook_secret: str = ""
    database_url: str = "sqlite:///./data/reviews.db"
    chroma_path: str = "./data/chroma"
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    provider_timeout: float = 60
    review_timeout: float = 210
    scanner_timeout: float = 25
    fix_max_attempts: int = 2
    fix_sandbox_image: str = "code-review-sandbox:local"
    fix_test_timeout: float = 60
    fix_worker_enabled: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()
