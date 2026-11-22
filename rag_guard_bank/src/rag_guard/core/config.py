from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ADMIN_TOKEN = "change_me"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    APP_ENV: str = "development"
    LOG_LEVEL: str = "INFO"
    ADMIN_TOKEN: str = DEFAULT_ADMIN_TOKEN

    OLLAMA_HOST: str = "http://localhost:11434"
    OLLAMA_TIMEOUT_S: float = 120.0
    LLM_MODEL: str = "llama3.2"
    EMBEDDING_MODEL: str = "nomic-embed-text"

    TOP_K: int = 5
    RELEVANCE_MIN_SCORE: float = 0.55
    # If the model refuses although retrieval looks strongly on-topic, retry once with more chunks.
    RETRY_MIN_RELEVANCE: float = 0.75
    RETRY_TOP_K: int = 8
    GROUNDING_THRESHOLD: float = 0.7

    NLI_MODEL: str = "cross-encoder/nli-deberta-v3-small"
    DETECTOR_MODE: str = "both"

    DATA_RAW_DIR: Path = PROJECT_ROOT / "data/raw"
    DATA_EVAL_DIR: Path = PROJECT_ROOT / "data/evaluation"
    CHROMA_DIR: Path = PROJECT_ROOT / "storage/chroma"
    COLLECTION_NAME: str = "bank_faq"


settings = Settings()
