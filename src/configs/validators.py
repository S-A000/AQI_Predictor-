"""
Validation helpers.
"""

from pathlib import Path

from .exceptions import ConfigurationError


def ensure_file_exists(path: Path):

    if not path.exists():

        raise ConfigurationError(
            f"Configuration file not found: {path}"
        )


def create_directory(path: Path):

    path.mkdir(parents=True, exist_ok=True)