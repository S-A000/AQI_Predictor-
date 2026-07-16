"""
Project wide configuration constants.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_DIRECTORY = PROJECT_ROOT / "configs"

DEFAULT_CONFIG_FILE = CONFIG_DIRECTORY / "config.yaml"

DEFAULT_MODEL_CONFIG = CONFIG_DIRECTORY / "model_config.yaml"

DEFAULT_FEATURE_CONFIG = CONFIG_DIRECTORY / "feature_config.yaml"

DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"