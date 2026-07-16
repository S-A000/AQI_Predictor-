"""
Configuration loading utilities.
"""

import yaml

from pathlib import Path

from dotenv import load_dotenv


def load_environment(env_path: Path):

    load_dotenv(env_path)


def load_yaml(file_path: Path):

    with open(file_path, "r", encoding="utf-8") as file:

        return yaml.safe_load(file)