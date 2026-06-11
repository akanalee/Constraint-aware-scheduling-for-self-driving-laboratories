"""
Configuration Loading Module with AttributeDict Support.
Supports both dot-notation and dictionary access for config values.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class Config(dict):
    """
    Enhanced dict supporting both dot-notation and dictionary access.
    Recursively converts nested dicts to Config objects.
    """

    def __init__(self, config_dict: Dict[str, Any] = None):
        super().__init__()
        if config_dict:
            for key, value in config_dict.items():
                self[key] = self._convert(value)

    def _convert(self, value):
        """Recursively convert nested dicts to Config objects."""
        if isinstance(value, dict) and not isinstance(value, Config):
            return Config(value)
        elif isinstance(value, list):
            return [self._convert(item) for item in value]
        else:
            return value

    def __getattr__(self, item: str):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(
                f"'Config' object has no attribute '{item}'. "
                f"Available keys: {list(self.keys())}"
            )

    def __setattr__(self, key: str, value):
        self[key] = self._convert(value)

    def __delattr__(self, item: str):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"'Config' object has no attribute '{item}'")

    def get(self, key: str, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def to_dict(self) -> Dict[str, Any]:
        """Convert back to plain dict (for serialization)."""
        result = {}
        for key, value in self.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            elif isinstance(value, list):
                result[key] = [
                    item.to_dict() if isinstance(item, Config) else item
                    for item in value
                ]
            else:
                result[key] = value
        return result

    def __repr__(self):
        return f"Config({super().__repr__()})"

    def __str__(self):
        return self._format_config()

    def _format_config(self, indent=0) -> str:
        lines = []
        for key, value in self.items():
            if isinstance(value, Config):
                lines.append(f"{'  ' * indent}{key}:")
                lines.append(value._format_config(indent + 1))
            else:
                lines.append(f"{'  ' * indent}{key}: {value}")
        return '\n'.join(lines)


def load_config(config_path: str) -> Config:
    """
    Load a YAML config file and return a Config object.

    Args:
        config_path: path to config file (relative or absolute)

    Returns:
        Config object supporting both dot-notation and dict access
    """
    config_path = Path(config_path)

    if not config_path.is_absolute():
        possible_paths = [
            config_path,
            Path.cwd() / config_path,
            Path(__file__).parent.parent / config_path,
        ]

        found_path = None
        for p in possible_paths:
            if p.exists():
                found_path = p
                break

        if found_path is None:
            tried_paths = '\n  - '.join(str(p) for p in possible_paths)
            raise FileNotFoundError(
                f"Config file not found. Tried:\n  - {tried_paths}"
            )
        config_path = found_path

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML config at {config_path}: {e}")

    if config_dict is None:
        config_dict = {}

    print(f"Config loaded from {config_path}")
    return Config(config_dict)


def merge_configs(base_config: Config, override_config: Config) -> Config:
    """Recursively merge two configs (override takes precedence)."""
    merged = Config(base_config.to_dict())
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], Config) and isinstance(value, Config):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged


def save_config(config: Config, save_path: str) -> None:
    """Save config to a YAML file."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False,
                  allow_unicode=True, indent=2)
    logger.info(f"Config saved to {save_path}")


def get_default_config() -> Config:
    """Get default fallback configuration."""
    return Config({
        'env_config': {
            'n_jobs': 20,
            'n_devices': 5,
            'max_steps': 500,
            'buffer_size': 100,
            'reward_type': 'dense',
        },
        'model': {
            'hidden_dim': 128,
            'n_layers': 3,
            'dropout': 0.1,
        },
        'training': {
            'n_iterations': 5000,
            'batch_size': 64,
            'learning_rate': 1e-4,
            'gamma': 0.99,
            'gae_lambda': 0.95,
        },
    })


__all__ = [
    'Config',
    'load_config',
    'merge_configs',
    'save_config',
    'get_default_config',
]
