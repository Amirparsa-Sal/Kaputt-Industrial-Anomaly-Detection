"""
Configuration management for SAM3 defect detection.

Defines the Config dataclass, default values, domain constants,
and YAML loading with optional deep-merge overrides.
"""

from dataclasses import dataclass

import yaml

LOGICAL_DEFECT_TYPES = ["missing_unit", "actuation"]

STRUCTURAL_DEFECT_TYPES = [
    "deformation",
    "deconstruction",
    "penetration",
    "superficial",
    "spillage",
]

DEFAULTS = {
    "data_dir": None,
    "split": "train",
    "model_name": "facebook/sam3",
    "cache_dir": None,
    "is_defect": "any",
    "major_defect": "any",
    "defect_type": "any",
    "mask": False,
    "crop": False,
    "input_size": None,
    "batch_size": 16,
    "thresholds": [0.5],
    "mask_threshold": 0.5,
    "num_data": -1,
    "exp_name": None,
    "prompts": None,
    "gpu_id": 0,
    "samples_per_save": 0,
}


@dataclass
class Config:
    """Typed experiment configuration loaded from YAML."""

    data_dir: str
    split: str = "train"
    model_name: str = "facebook/sam3"
    cache_dir: str | None = None
    is_defect: str = "any"
    major_defect: str = "any"
    defect_type: str = "any"
    mask: bool = False
    crop: bool = False
    input_size: int | None = None
    batch_size: int = 16
    thresholds: list | None = None
    mask_threshold: float = 0.5
    num_data: int = -1
    exp_name: str | None = None
    prompts: dict[str, list[str]] | None = None
    gpu_id: int = 0
    samples_per_save: int = 0

    def __post_init__(self) -> None:
        if not self.data_dir:
            raise ValueError("'data_dir' is required in the config file.")
        if self.split not in ("train", "validation", "test"):
            raise ValueError(f"'split' must be train/validation/test, got '{self.split}'")

        for attr in ("is_defect", "major_defect"):
            val = getattr(self, attr)
            if isinstance(val, bool):
                object.__setattr__(self, attr, str(val).lower())

        if self.is_defect not in ("true", "false", "any"):
            raise ValueError(f"'is_defect' must be true/false/any, got '{self.is_defect}'")
        if self.major_defect not in ("true", "false", "any"):
            raise ValueError(f"'major_defect' must be true/false/any, got '{self.major_defect}'")
        if self.defect_type not in ("structural", "logical", "any"):
            raise ValueError(f"'defect_type' must be structural/logical/any, got '{self.defect_type}'")
        if isinstance(self.input_size, str):
            val = None if self.input_size.lower() in ("none", "null", "") else int(self.input_size)
            object.__setattr__(self, "input_size", val)
        elif self.input_size is not None:
            object.__setattr__(self, "input_size", int(self.input_size))
        if self.crop and self.mask:
            raise ValueError("'crop' and 'mask' cannot both be true — masks correspond to full images, not crops.")

        if self.thresholds is None:
            object.__setattr__(self, "thresholds", [0.5])
        elif isinstance(self.thresholds, (int, float)):
            object.__setattr__(self, "thresholds", [float(self.thresholds)])
        else:
            object.__setattr__(self, "thresholds", sorted(float(t) for t in self.thresholds))

        if self.prompts is not None:
            for key, vals in self.prompts.items():
                if not isinstance(vals, list) or not all(isinstance(v, str) for v in vals):
                    raise ValueError(
                        f"Each prompts entry must be a list of strings, "
                        f"but '{key}' maps to {type(vals)}"
                    )


def _deep_merge(base: dict, override: dict) -> dict:
    """
    Recursively merge *override* into *base*.

    Scalar values and lists are replaced outright; nested dicts are merged
    so that only the keys present in the override are updated.
    """
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def load_config(path: str, override_path: str | None = None) -> Config:
    """
    Load a YAML config file, optionally deep-merged with an override file.

    Supports backward compatibility: singular ``threshold`` is converted to
    the list-based ``thresholds`` field.
    """
    with open(path) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a YAML mapping, got {type(raw)}")

    if override_path:
        with open(override_path) as f:
            overrides = yaml.safe_load(f)
        if not isinstance(overrides, dict):
            raise ValueError(f"Override config must be a YAML mapping, got {type(overrides)}")
        raw = _deep_merge(raw, overrides)

    merged = {**DEFAULTS, **raw}

    # Backward compat: singular 'threshold' → list 'thresholds'
    if "threshold" in merged:
        if "thresholds" not in raw:
            merged["thresholds"] = [float(merged["threshold"])]
        del merged["threshold"]

    valid_keys = set(Config.__dataclass_fields__.keys())
    filtered = {k: v for k, v in merged.items() if k in valid_keys}

    return Config(**filtered)
