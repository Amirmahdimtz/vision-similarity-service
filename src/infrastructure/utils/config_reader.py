from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from src.infrastructure.di.inject import inject


@inject
class ConfigReader:
    __di_singleton__ = True

    def __init__(self) -> None:
        env_type = os.getenv("ENV_TYPE", "development").strip().lower()

        src_dir = Path(__file__).resolve().parents[2]
        res_dir = src_dir / "host" / "res"

        config_path = (
            res_dir / "appsettings.development.yaml"
            if env_type == "development"
            else res_dir / "appsettings.yaml"
        )

        if not config_path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with config_path.open("r", encoding="utf-8") as file:
            self._config = yaml.safe_load(file) or {}

    def get(self, key: str, default: Any = None) -> Any:
        value: Any = self._config

        for part in key.split("."):
            if not isinstance(value, dict) or part not in value:
                return default
            value = value[part]

        return value

    def get_api_prefix(self) -> str:
        return str(self.get("api.prefix", "/api/v1")).rstrip("/")
