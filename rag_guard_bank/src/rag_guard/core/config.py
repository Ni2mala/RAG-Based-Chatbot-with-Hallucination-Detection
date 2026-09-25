from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    APP_ENV: str = "development"
    LOG_LEVEL: str = "INFO"
    ADMIN_TOKEN: str = "change_me"

    OLLAMA_HOST: str = "http://localhost:11434"
    LLM_MODEL: str = "llama3.2"
    EMBEDDING_MODEL: str = "nomic-embed-text"

    TOP_K: int = 5
    RELEVANCE_MIN_SCORE: float = 0.55
    GROUNDING_THRESHOLD: float = 0.7

    NLI_MODEL: str = "cross-encoder/nli-deberta-v3-small"
    DETECTOR_MODE: str = "both"

    DATA_RAW_DIR: Path = PROJECT_ROOT / "data/raw"
    DATA_EVAL_DIR: Path = PROJECT_ROOT / "data/evaluation"
    CHROMA_DIR: Path = PROJECT_ROOT / "storage/chroma"
    COLLECTION_NAME: str = "bank_faq"


settings = Settings()
