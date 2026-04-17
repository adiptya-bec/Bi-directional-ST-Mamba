"""
config_loader.py — YAML configuration loader with defaults fallback.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional


def load_config(config_path: Optional[str] = "config.yaml") -> Dict[str, Any]:
    """Load pipeline configuration from a YAML file.

    Falls back to an empty dict if PyYAML is not installed or the file
    does not exist. All pipeline code uses dict.get() with defaults,
    so this is safe for environments without PyYAML.

    Parameters
    ----------
    config_path : path to config.yaml

    Returns
    -------
    cfg : dict with configuration values
    """
    if config_path is None:
        return {}

    path = Path(config_path)
    if not path.exists():
        return {}

    try:
        import yaml
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}
        return cfg
    except ImportError:
        # PyYAML not available — return empty dict; callers use defaults
        return {}
    except Exception as e:
        import warnings
        warnings.warn(f"Failed to load {config_path}: {e}", stacklevel=2)
        return {}
