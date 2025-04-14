import os
from functools import lru_cache

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    """Application settings"""

    model_config = SettingsConfigDict(
        case_sensitive=True, env_file=".env", extra="allow"
    )

    API_PREFIX: str = "api"
    PROJECT_NAME: str = "GL Code Predictor"

    # Cohere API settings
    COHERE_API_KEY: str = os.getenv(
        "COHERE_API_KEY", "Gwiu5tTMOeV2uLLA13g30SUXYliv2p4FgInZ2Ki3"
    )

    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    FORCE_INDEX_OVERWRITE: bool = True

    GL_CODES_PATH: str = os.getenv("GL_CODES_PATH", "data/gl_codes_genai_takehome.csv")
    TRANSACTIONS_PATH: str = os.getenv(
        "TRANSACTIONS_PATH", "data/transactions_genai_takehome.csv"
    )

    VECTOR_STORE_DIR: str = os.getenv("VECTOR_STORE_DIR", "data/vector_store")
    VECTOR_SCORE_THHRESHOLD: float = float(os.getenv("VECTOR_SCORE_THHRESHOLD", "0.6"))
    SEMANTIC_SCORE_THHRESHOLD: float = float(
        os.getenv("SEMANTIC_SCORE_THHRESHOLD", "0.2")
    )

    TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "3"))


@lru_cache()
def get_settings():
    return Settings()
