"""Configuration module for Agentic Risk Analysis.

Centralizes paths, environment variables, default model names, and shield thresholds.
"""

from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


# Base paths
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
DOCS_DIR = ROOT_DIR / "docs"
EXPERIMENTS_DIR = ROOT_DIR / "experiments"


class Settings(BaseSettings):
    """Application and research pipeline settings."""

    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Environment & Reproducibility
    agent_risk_env: str = Field(default="development", alias="AGENT_RISK_ENV")
    random_seed: int = Field(default=42, alias="RANDOM_SEED")
    device: str = Field(default="cpu", alias="DEVICE")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # API Keys
    groq_api_key: Optional[str] = Field(default=None, alias="GROQ_API_KEY")
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    hf_token: Optional[str] = Field(default=None, alias="HF_TOKEN")

    # Model Defaults
    default_llm_model: str = Field(
        default="llama-3.1-70b-versatile",
        alias="DEFAULT_LLM_MODEL"
    )
    local_model_path: Path = Field(
        default=MODELS_DIR / "lora_adapters" / "stage_02_lora",
        alias="LOCAL_MODEL_PATH"
    )
    xgboost_model_path: Path = Field(
        default=MODELS_DIR / "xgboost" / "risk_classifier.json",
        alias="XGBOOST_MODEL_PATH"
    )

    # Risk Engine Safeguard Thresholds
    risk_threshold_warning: float = Field(default=0.35, alias="RISK_THRESHOLD_WARNING")
    risk_threshold_block: float = Field(default=0.70, alias="RISK_THRESHOLD_BLOCK")
    necessity_threshold: float = Field(default=0.40, alias="NECESSITY_THRESHOLD")

    # Directory Paths
    root_dir: Path = ROOT_DIR
    data_dir: Path = DATA_DIR
    raw_data_dir: Path = DATA_DIR / "raw"
    processed_data_dir: Path = DATA_DIR / "processed"
    ground_truth_dir: Path = DATA_DIR / "ground_truth"
    models_dir: Path = MODELS_DIR
    xgboost_dir: Path = MODELS_DIR / "xgboost"
    lora_dir: Path = MODELS_DIR / "lora_adapters"


# Singleton instance
settings = Settings()

