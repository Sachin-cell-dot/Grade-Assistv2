from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-backed runtime settings."""
    ollama_host: str = "http://localhost:11434"
    ollama_vision_model: str = "qwen3-vl:4b-instruct"
    ollama_timeout_seconds: int = 90
    ollama_num_predict: int = 512
    ollama_num_ctx: int = 4096
    groq_api_key: str | None = None
    groq_vision_model: str = "qwen/qwen3.8-27b"
    groq_timeout_seconds: int = 120
    groq_max_image_bytes: int = 20 * 1024 * 1024
    capture_sample_every_n_frames: int = 5
    capture_page_change_threshold: float = 18.0
    capture_stability_threshold: float = 3.0
    capture_stable_frames_required: int = 3
    capture_blur_threshold: float = 80.0
    capture_min_frames_between_saves: int = 30
    database_path: Path = Path("data/gradeassist.db")
    log_level: str = "INFO"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
