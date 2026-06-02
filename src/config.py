"""Configuration loader and management."""
import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any

class ConfigManager:
    """Loads and manages configuration from YAML file."""

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize configuration manager.

        Args:
            config_path: Path to config.yaml file relative to project root
        """
        self.project_root = Path(__file__).parent.parent
        self.config_path = self.project_root / config_path

        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        self.config = self._load_yaml()
        self._setup_paths()

    def _load_yaml(self) -> Dict[str, Any]:
        """Load YAML configuration file."""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _setup_paths(self):
        """Convert relative paths to absolute paths."""
        paths = self.config.get('paths', {})
        for key, value in paths.items():
            if isinstance(value, str) and not value.startswith('/'):
                abs_path = self.project_root / value
                paths[key] = str(abs_path)
                # Create directories if they don't exist (skip files like CSV)
                if (key.startswith('outputs_') or key.startswith('data_')) and not value.endswith('.csv'):
                    os.makedirs(abs_path, exist_ok=True)
        self.config['paths'] = paths

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot notation.

        Args:
            key: Configuration key (e.g., 'data.target_column')
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value

    def get_path(self, key: str) -> Path:
        """Get path configuration and return as Path object.

        Args:
            key: Path key (e.g., 'data_raw', 'models')

        Returns:
            Path object
        """
        path_str = self.get(f"paths.{key}")
        if path_str is None:
            raise KeyError(f"Path '{key}' not found in configuration")
        return Path(path_str)

    def __repr__(self) -> str:
        return f"<ConfigManager: {self.config_path}>"


# Global instance
_config_manager = None

def get_config() -> ConfigManager:
    """Get or create global ConfigManager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager
