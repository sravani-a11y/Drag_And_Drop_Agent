"""Loads the component, feature, and API registries used by the assistant.

Read-only: does not call the LLM, tools, backend APIs, or the frontend.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_REGISTRY_DIR = Path(__file__).resolve().parent / "registries"


class KnowledgeLoader:
    def __init__(self, registry_dir: str | Path = _REGISTRY_DIR):
        self.registry_dir = Path(registry_dir)

    def _load_registry(self, filename: str, key: str) -> list:
        path = self.registry_dir / filename
        if not path.exists():
            logger.warning("Registry file not found: %s", path)
            return []

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            logger.warning("Registry file is empty or contains invalid JSON: %s", path)
            return []

        return data.get(key, [])

    def load_components(self) -> list:
        return self._load_registry("component_catalog.json", "components")

    def load_features(self) -> list:
        return self._load_registry("feature_registry.json", "features")

    def load_apis(self) -> list:
        return self._load_registry("api_registry.json", "apis")

    def load_all(self) -> dict:
        return {
            "components": self.load_components(),
            "features": self.load_features(),
            "apis": self.load_apis(),
        }
