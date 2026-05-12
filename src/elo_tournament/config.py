"""
Configuration for ELO tournament-based VLM anomaly detection.

Three-step pipeline:
  1. Zero-shot scoring (initial ELO) for all images.
  2. Reference-only pairwise tournament with ELO updates.
  3. Query vs. references pairwise tournament with ELO updates.

Match modes:
  - ``confidence``: model outputs ``CONFIDENCE: <float>`` in [0, 1].
  - ``wdl``: model outputs ``RESULT: WIN/DRAW/LOSS``.

ELO update formula (per match between A and B with result *s*):
  E_A = 1 / (1 + 10^((R_B - R_A) / d))
  R_A = clip(R_A + k * (s - E_A), 0, 1)
  R_B = clip(R_B - k * (s - E_A), 0, 1)
"""

from dataclasses import dataclass

import yaml


ELO_TOURNAMENT_DEFAULTS: dict = {
    "data_dir": None,
    "split": "train",
    "model_name": "Qwen/Qwen3.5-9B",
    "cache_dir": None,
    "is_defect": "any",
    "major_defect": "any",
    "defect_type": "any",
    "mask": False,
    "crop": False,
    "input_size": 1024,
    "gpu_id": 0,
    "num_data": -1,
    "thresholds": [0.5],
    "exp_name": None,
    "temperature": 0.0,
    "max_new_tokens": 256,
    "load_in_4bit": False,
    "enable_thinking": False,
    "report_interval_minutes": 30,
    "samples_per_save": 0,
    "min_pixels": 200704,
    "max_pixels": 401408,
    # ELO parameters
    "elo_d": 0.5,
    "elo_k": [0.1],
    # Match mode
    "match_mode": "confidence",
    # Zero-shot
    "zero_shot_scoring_mode": "text",
    "zero_shot_query_csv": None,
    "zero_shot_reference_csv": None,
    # Tournament
    "num_references": 3,
    "use_grid": True,
    # Prompts — zero-shot
    "system_prompt_zero_shot_text": None,
    "user_prompt_zero_shot_text": None,
    "system_prompt_zero_shot_logits": None,
    "user_prompt_zero_shot_logits": None,
    # Prompts — match (confidence)
    "system_prompt_match_confidence": None,
    "user_prompt_match_confidence": None,
    # Prompts — match (WDL)
    "system_prompt_match_wdl": None,
    "user_prompt_match_wdl": None,
}


@dataclass
class EloTournamentConfig:
    """Typed configuration for ELO tournament-based VLM anomaly detection."""

    # ---- Data ----
    data_dir: str
    split: str = "train"
    is_defect: str = "any"
    major_defect: str = "any"
    defect_type: str = "any"
    num_data: int = -1
    crop: bool = False
    mask: bool = False
    input_size: int = 1024

    # ---- Model ----
    model_name: str = "Qwen/Qwen3.5-9B"
    cache_dir: str | None = None
    gpu_id: int = 0
    load_in_4bit: bool = False
    min_pixels: int = 200704
    max_pixels: int = 401408

    # ---- Generation ----
    temperature: float = 0.0
    max_new_tokens: int = 256
    enable_thinking: bool = False
    report_interval_minutes: float = 30
    samples_per_save: int = 0

    # ---- ELO parameters ----
    elo_d: float = 0.5
    elo_k: list | float = 0.1

    # ---- Match mode ----
    match_mode: str = "confidence"

    # ---- Zero-shot ----
    zero_shot_scoring_mode: str = "text"
    zero_shot_query_csv: str | None = None
    zero_shot_reference_csv: str | None = None

    # ---- Tournament ----
    num_references: int = 3
    use_grid: bool = True

    # ---- Experiment ----
    thresholds: list | None = None
    exp_name: str | None = None

    # ---- Prompts — zero-shot ----
    system_prompt_zero_shot_text: str | None = None
    user_prompt_zero_shot_text: str | None = None
    system_prompt_zero_shot_logits: str | None = None
    user_prompt_zero_shot_logits: str | None = None

    # ---- Prompts — match (confidence) ----
    system_prompt_match_confidence: str | None = None
    user_prompt_match_confidence: str | None = None

    # ---- Prompts — match (WDL) ----
    system_prompt_match_wdl: str | None = None
    user_prompt_match_wdl: str | None = None

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
            object.__setattr__(self, "input_size", int(self.input_size))
        else:
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

        if self.match_mode not in ("confidence", "wdl"):
            raise ValueError(
                f"'match_mode' must be 'confidence' or 'wdl', "
                f"got '{self.match_mode}'"
            )

        if self.zero_shot_scoring_mode not in ("text", "logits"):
            raise ValueError(
                f"'zero_shot_scoring_mode' must be 'text' or 'logits', "
                f"got '{self.zero_shot_scoring_mode}'"
            )

        if not (1 <= self.num_references <= 3):
            raise ValueError(
                f"'num_references' must be 1–3, got {self.num_references}"
            )

        if self.elo_d <= 0:
            raise ValueError(f"'elo_d' must be > 0, got {self.elo_d}")

        # Normalize elo_k to a sorted list of floats.
        if isinstance(self.elo_k, (int, float)):
            object.__setattr__(self, "elo_k", [float(self.elo_k)])
        elif isinstance(self.elo_k, list):
            object.__setattr__(
                self, "elo_k",
                sorted(float(k) for k in self.elo_k),
            )
        if not self.elo_k:
            raise ValueError("'elo_k' must contain at least one value.")
        for k_val in self.elo_k:
            if k_val <= 0:
                raise ValueError(
                    f"All 'elo_k' values must be > 0, got {k_val}"
                )

        def _need(label: str, raw: str | None) -> str:
            s = (raw if raw is not None else "").strip()
            if not s:
                raise ValueError(f"Config must set non-empty '{label}'.")
            return s

        # Zero-shot prompts are required unless BOTH CSV paths are provided.
        both_csvs = (
            self.zero_shot_query_csv is not None
            and self.zero_shot_reference_csv is not None
        )
        if not both_csvs:
            if self.zero_shot_scoring_mode == "text":
                object.__setattr__(
                    self, "system_prompt_zero_shot_text",
                    _need("system_prompt_zero_shot_text",
                          self.system_prompt_zero_shot_text),
                )
                object.__setattr__(
                    self, "user_prompt_zero_shot_text",
                    _need("user_prompt_zero_shot_text",
                          self.user_prompt_zero_shot_text),
                )
            else:
                object.__setattr__(
                    self, "system_prompt_zero_shot_logits",
                    _need("system_prompt_zero_shot_logits",
                          self.system_prompt_zero_shot_logits),
                )
                object.__setattr__(
                    self, "user_prompt_zero_shot_logits",
                    _need("user_prompt_zero_shot_logits",
                          self.user_prompt_zero_shot_logits),
                )

        # Match prompts are always required for the active match mode.
        if self.match_mode == "confidence":
            object.__setattr__(
                self, "system_prompt_match_confidence",
                _need("system_prompt_match_confidence",
                      self.system_prompt_match_confidence),
            )
            object.__setattr__(
                self, "user_prompt_match_confidence",
                _need("user_prompt_match_confidence",
                      self.user_prompt_match_confidence),
            )
        else:
            object.__setattr__(
                self, "system_prompt_match_wdl",
                _need("system_prompt_match_wdl",
                      self.system_prompt_match_wdl),
            )
            object.__setattr__(
                self, "user_prompt_match_wdl",
                _need("user_prompt_match_wdl",
                      self.user_prompt_match_wdl),
            )

    # ---- Prompt accessors ----

    def zero_shot_system_prompt(self) -> str:
        """System prompt for the configured zero-shot scoring mode."""
        if self.zero_shot_scoring_mode == "logits":
            return self.system_prompt_zero_shot_logits
        return self.system_prompt_zero_shot_text

    def zero_shot_user_prompt(self) -> str:
        """User prompt for the configured zero-shot scoring mode."""
        if self.zero_shot_scoring_mode == "logits":
            return self.user_prompt_zero_shot_logits
        return self.user_prompt_zero_shot_text

    def match_system_prompt(self) -> str:
        """System prompt for the configured match mode."""
        if self.match_mode == "wdl":
            return self.system_prompt_match_wdl
        return self.system_prompt_match_confidence

    def match_user_prompt_template(self) -> str:
        """User prompt template (with ``{image_description}``) for matches."""
        if self.match_mode == "wdl":
            return self.user_prompt_match_wdl
        return self.user_prompt_match_confidence


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def load_elo_tournament_config(
    path: str, override_path: str | None = None,
) -> EloTournamentConfig:
    """Load a YAML config, optionally deep-merged with an override file."""
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

    merged = {**ELO_TOURNAMENT_DEFAULTS, **raw}

    valid_keys = set(EloTournamentConfig.__dataclass_fields__.keys())
    filtered = {k: v for k, v in merged.items() if k in valid_keys}

    return EloTournamentConfig(**filtered)
