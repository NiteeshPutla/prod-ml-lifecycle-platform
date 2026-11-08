"""Configuration management for the ML lifecycle platform."""

import os
from pathlib import Path
from typing import Optional, Type, TypeVar
from core.schemas import (
    FeatureStoreConfig,
    TrainingConfig,
    ServingConfig,
    ExperimentConfig,
    MonitoringConfig,
)

T = TypeVar('T', bound=object)

DEFAULT_CONFIG_DIR = Path(__file__).parent.parent / "configs"


def load_config(config_path: str, config_type: Type[T]) -> T:
    """Load configuration from YAML file."""
    import yaml
    from enum import Enum
    from pydantic import BaseModel

    def convert_enums(obj, model_fields=None):
        """Recursively convert string values to Enum types based on model field annotations."""
        if model_fields is None and hasattr(config_type, 'model_fields'):
            model_fields = config_type.model_fields

        if isinstance(obj, dict):
            result = {}
            for k, v in obj.items():
                # Check if this field has an Enum annotation
                field_type = None
                if model_fields and k in model_fields:
                    field_type = model_fields[k].annotation

                if field_type and isinstance(field_type, type) and issubclass(field_type, Enum):
                    # Convert string to Enum
                    try:
                        result[k] = field_type(v)
                    except (ValueError, TypeError):
                        result[k] = v
                elif isinstance(v, dict):
                    # For nested models, get their fields
                    nested_fields = None
                    if hasattr(v, 'model_fields'):
                        nested_fields = v.model_fields
                    result[k] = convert_enums(v, nested_fields)
                elif isinstance(v, list):
                    result[k] = [convert_enums(item) for item in v]
                else:
                    result[k] = convert_enums(v)
            return result
        elif isinstance(obj, list):
            return [convert_enums(item) for item in obj]
        else:
            return obj

    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)

    data = convert_enums(data)
    return config_type(**data)


def load_config_from_env(config_type: Type[T], prefix: str = "") -> T:
    """Load configuration from environment variables."""
    import yaml

    # Map of config types to their default YAML paths
    default_paths = {
        FeatureStoreConfig: "feature_store.yaml",
        TrainingConfig: "training.yaml",
        ServingConfig: "serving.yaml",
        ExperimentConfig: "experiment.yaml",
        MonitoringConfig: "monitoring.yaml",
    }

    default_path = DEFAULT_CONFIG_DIR / default_paths.get(config_type, "config.yaml")

    if default_path.exists():
        return load_config(str(default_path), config_type)

    # Return default instance
    return config_type()


def get_config_dir() -> Path:
    """Get the configuration directory, creating if needed."""
    config_dir = Path(os.getenv("MLOPS_CONFIG_DIR", DEFAULT_CONFIG_DIR))
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def save_config(config: object, config_path: str) -> None:
    """Save configuration to YAML file."""
    import yaml
    from pydantic import BaseModel
    from enum import Enum

    def convert_for_yaml(obj):
        """Recursively convert objects to YAML-serializable types."""
        if isinstance(obj, Enum):
            return obj.value
        elif isinstance(obj, dict):
            return {k: convert_for_yaml(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_yaml(v) for v in obj]
        elif isinstance(obj, BaseModel):
            return convert_for_yaml(obj.model_dump())
        elif hasattr(obj, '__dict__'):
            return convert_for_yaml(obj.__dict__)
        else:
            return obj

    if isinstance(config, BaseModel):
        data = config.model_dump()
    else:
        data = config.__dict__

    data = convert_for_yaml(data)

    Path(config_path).parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.safe_dump(data, f, default_flow_style=False)