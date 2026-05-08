"""
Configuration management for VLM-based anomaly classification.

All prompts are loaded from YAML — there are no default prompt strings in code.
"""

from dataclasses import dataclass

import yaml


VLM_DEFAULTS: dict = {
    "data_dir": None,
    "split": "train",
    "model_name": "Qwen/Qwen3.5-9B",
    "cache_dir": None,
    "is_defect": "any",
    "major_defect": "any",
    "defect_type": "any",
    "mask": False,
    "crop": False,
    "input_size": None,
    "gpu_id": 0,
    "num_data": -1,
    "thresholds": [0.5],
    "exp_name": None,
    "scoring_mode": "logits",
    "shot_mode": "zero_shot",
    "temperature": 0.0,
    "max_new_tokens": 256,
    "load_in_4bit": False,
    "enable_thinking": False,
    "report_interval_minutes": 30,
    "samples_per_save": 0,
    "min_pixels": 200704,
    "max_pixels": 401408,
    # Few-shot: if reference parquet has 1–2 images per item, cycle them to
    # fill three panels (default). Set false to require exactly three.
    "pad_short_references": True,
    "system_prompt_text_zero_shot": None,
    "system_prompt_text_few_shot": None,
    "system_prompt_logits_zero_shot": None,
    "system_prompt_logits_few_shot": None,
    "user_prompt_text_zero_shot": None,
    "user_prompt_text_few_shot": None,
    "user_prompt_logits_zero_shot": None,
    "user_prompt_logits_few_shot": None,
}


@dataclass
class VLMConfig:
    """Typed configuration for VLM anomaly classification experiments."""

    data_dir: str
    split: str = "train"
    is_defect: str = "any"
    major_defect: str = "any"
    defect_type: str = "any"
    num_data: int = -1
    crop: bool = False
    mask: bool = False
    input_size: int | None = None

    model_name: str = "Qwen/Qwen3.5-9B"
    cache_dir: str | None = None
    gpu_id: int = 0
    load_in_4bit: bool = False
    min_pixels: int = 200704
    max_pixels: int = 401408
    pad_short_references: bool = True

    scoring_mode: str = "logits"
    shot_mode: str = "zero_shot"
    temperature: float = 0.0
    max_new_tokens: int = 256
    enable_thinking: bool = False
    report_interval_minutes: float = 30
    samples_per_save: int = 0

    thresholds: list | None = None

    exp_name: str | None = None

    system_prompt_text_zero_shot: str | None = None
    system_prompt_text_few_shot: str | None = None
    system_prompt_logits_zero_shot: str | None = None
    system_prompt_logits_few_shot: str | None = None
    user_prompt_text_zero_shot: str | None = None
    user_prompt_text_few_shot: str | None = None
    user_prompt_logits_zero_shot: str | None = None
    user_prompt_logits_few_shot: str | None = None

    def __post_init__(self) -> None:
        if not self.data_dir:
            raise ValueError("'data_dir' is required in the config file.")
        if self.split not in ("train", "validation", "test"):
            raise ValueError(
                f"'split' must be train/validation/test, got '{self.split}'"
            )

        for attr in ("is_defect", "major_defect"):
            val = getattr(self, attr)
            if isinstance(val, bool):
                object.__setattr__(self, attr, str(val).lower())

        if self.is_defect not in ("true", "false", "any"):
            raise ValueError(
                f"'is_defect' must be true/false/any, got '{self.is_defect}'"
            )
        if self.major_defect not in ("true", "false", "any"):
            raise ValueError(
                f"'major_defect' must be true/false/any, got '{self.major_defect}'"
            )
        if self.defect_type not in ("structural", "logical", "any"):
            raise ValueError(
                f"'defect_type' must be structural/logical/any, "
                f"got '{self.defect_type}'"
            )

        if isinstance(self.input_size, str):
            val = (
                None
                if self.input_size.lower() in ("none", "null", "")
                else int(self.input_size)
            )
            object.__setattr__(self, "input_size", val)
        elif self.input_size is not None:
            object.__setattr__(self, "input_size", int(self.input_size))

        if self.crop and self.mask:
            raise ValueError(
                "'crop' and 'mask' cannot both be true — masks correspond "
                "to full images, not crops."
            )

        if self.thresholds is None:
            object.__setattr__(self, "thresholds", [0.5])
        elif isinstance(self.thresholds, (int, float)):
            object.__setattr__(self, "thresholds", [float(self.thresholds)])
        else:
            object.__setattr__(
                self, "thresholds",
                sorted(float(t) for t in self.thresholds),
            )

        if self.scoring_mode not in ("text", "logits"):
            raise ValueError(
                f"'scoring_mode' must be 'text' or 'logits', "
                f"got '{self.scoring_mode}'"
            )

        if self.shot_mode not in ("zero_shot", "few_shot"):
            raise ValueError(
                f"'shot_mode' must be 'zero_shot' or 'few_shot', "
                f"got '{self.shot_mode}'"
            )

        if self.shot_mode == "few_shot" and (
            self.input_size is None or int(self.input_size) < 2
        ):
            raise ValueError(
                "When shot_mode is 'few_shot', set 'input_size' to an "
                "integer ≥ 2 (final 2×2 grid is resized to this square).",
            )

        def _need(label: str, raw: str | None) -> str:
            s = (raw if raw is not None else "").strip()
            if not s:
                raise ValueError(f"Config must set non-empty '{label}'.")
            return s

        object.__setattr__(
            self,
            "system_prompt_text_zero_shot",
            _need(
                "system_prompt_text_zero_shot",
                self.system_prompt_text_zero_shot,
            ),
        )
        object.__setattr__(
            self,
            "system_prompt_text_few_shot",
            _need(
                "system_prompt_text_few_shot",
                self.system_prompt_text_few_shot,
            ),
        )
        object.__setattr__(
            self,
            "system_prompt_logits_zero_shot",
            _need(
                "system_prompt_logits_zero_shot",
                self.system_prompt_logits_zero_shot,
            ),
        )
        object.__setattr__(
            self,
            "system_prompt_logits_few_shot",
            _need(
                "system_prompt_logits_few_shot",
                self.system_prompt_logits_few_shot,
            ),
        )
        object.__setattr__(
            self,
            "user_prompt_text_zero_shot",
            _need("user_prompt_text_zero_shot", self.user_prompt_text_zero_shot),
        )
        object.__setattr__(
            self,
            "user_prompt_text_few_shot",
            _need("user_prompt_text_few_shot", self.user_prompt_text_few_shot),
        )
        object.__setattr__(
            self,
            "user_prompt_logits_zero_shot",
            _need(
                "user_prompt_logits_zero_shot",
                self.user_prompt_logits_zero_shot,
            ),
        )
        object.__setattr__(
            self,
            "user_prompt_logits_few_shot",
            _need(
                "user_prompt_logits_few_shot",
                self.user_prompt_logits_few_shot,
            ),
        )

    def system_message_for_run(self) -> str:
        """System prompt for the current ``scoring_mode`` and ``shot_mode``."""
        if self.scoring_mode == "logits":
            if self.shot_mode == "few_shot":
                return self.system_prompt_logits_few_shot
            return self.system_prompt_logits_zero_shot
        if self.shot_mode == "few_shot":
            return self.system_prompt_text_few_shot
        return self.system_prompt_text_zero_shot

    def user_message_for_run(self) -> str:
        """User-facing task text for the current ``scoring_mode`` and ``shot_mode``."""
        if self.scoring_mode == "logits":
            if self.shot_mode == "few_shot":
                return self.user_prompt_logits_few_shot
            return self.user_prompt_logits_zero_shot
        if self.shot_mode == "few_shot":
            return self.user_prompt_text_few_shot
        return self.user_prompt_text_zero_shot


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def load_vlm_config(
    path: str, override_path: str | None = None,
) -> VLMConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a YAML mapping, got {type(raw)}")

    if override_path:
        with open(override_path) as f:
            overrides = yaml.safe_load(f)
        if not isinstance(overrides, dict):
            raise ValueError(
                f"Override config must be a YAML mapping, got {type(overrides)}"
            )
        raw = _deep_merge(raw, overrides)

    merged = {**VLM_DEFAULTS, **raw}

    valid_keys = set(VLMConfig.__dataclass_fields__.keys())
    filtered = {k: v for k, v in merged.items() if k in valid_keys}

    return VLMConfig(**filtered)
