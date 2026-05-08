"""
Configuration management for tournament-based VLM anomaly detection.

Supports two strategies:
  - ``simple_ranking``: rank all images at once; derive score from query rank.
  - ``league``: pairwise comparisons with confidence scoring.  Two league
    types are available:
      * ``swiss`` — Swiss-system pairing over ⌈log₂(m+1)⌉ rounds.
      * ``complete`` — full round-robin (every pair plays).

Prompts use ``{image_description}`` and ``{num_images}`` placeholders that
are filled at inference time based on ``use_grid`` and the actual image count.
"""

from dataclasses import dataclass

import yaml


TOURNAMENT_DEFAULTS: dict = {
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
    "min_pixels": 200704,
    "max_pixels": 401408,
    "tournament_strategy": "simple_ranking",
    "num_references": 3,
    "use_grid": True,
    "repeat": 1,
    "league_type": "swiss",
    "system_prompt_ranking": None,
    "user_prompt_ranking": None,
    "scoring_mode": "text",
    "system_prompt_league": None,
    "user_prompt_league": None,
    "system_prompt_league_logits": None,
    "user_prompt_league_logits": None,
}


@dataclass
class TournamentConfig:
    """Typed configuration for tournament-based VLM anomaly detection."""

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

    # ---- Tournament ----
    tournament_strategy: str = "simple_ranking"
    scoring_mode: str = "text"
    num_references: int = 3
    use_grid: bool = True
    repeat: int = 1
    league_type: str = "swiss"

    # ---- Experiment ----
    thresholds: list | None = None
    exp_name: str | None = None

    # ---- Prompts (contain {image_description} and {num_images} placeholders) ----
    system_prompt_ranking: str | None = None
    user_prompt_ranking: str | None = None
    system_prompt_league: str | None = None
    user_prompt_league: str | None = None
    system_prompt_league_logits: str | None = None
    user_prompt_league_logits: str | None = None

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

        if self.scoring_mode not in ("text", "logits"):
            raise ValueError(
                f"'scoring_mode' must be 'text' or 'logits', "
                f"got '{self.scoring_mode}'"
            )

        if self.tournament_strategy not in ("simple_ranking", "league"):
            raise ValueError(
                f"'tournament_strategy' must be 'simple_ranking' or 'league', "
                f"got '{self.tournament_strategy}'"
            )

        if self.tournament_strategy == "league":
            if self.league_type not in ("swiss", "complete"):
                raise ValueError(
                    f"'league_type' must be 'swiss' or 'complete', "
                    f"got '{self.league_type}'"
                )

        if not (1 <= self.num_references <= 3):
            raise ValueError(
                f"'num_references' must be 1–3, got {self.num_references}"
            )

        if self.repeat < 1:
            raise ValueError(
                f"'repeat' must be >= 1, got {self.repeat}"
            )

        def _need(label: str, raw: str | None) -> str:
            s = (raw if raw is not None else "").strip()
            if not s:
                raise ValueError(f"Config must set non-empty '{label}'.")
            return s

        if self.tournament_strategy == "simple_ranking":
            if self.scoring_mode == "logits":
                raise ValueError(
                    "Logits scoring is only supported for the 'league' "
                    "tournament strategy, not 'simple_ranking'."
                )
            object.__setattr__(
                self, "system_prompt_ranking",
                _need("system_prompt_ranking", self.system_prompt_ranking),
            )
            object.__setattr__(
                self, "user_prompt_ranking",
                _need("user_prompt_ranking", self.user_prompt_ranking),
            )
        elif self.scoring_mode == "logits":
            object.__setattr__(
                self, "system_prompt_league_logits",
                _need("system_prompt_league_logits", self.system_prompt_league_logits),
            )
            object.__setattr__(
                self, "user_prompt_league_logits",
                _need("user_prompt_league_logits", self.user_prompt_league_logits),
            )
        else:
            object.__setattr__(
                self, "system_prompt_league",
                _need("system_prompt_league", self.system_prompt_league),
            )
            object.__setattr__(
                self, "user_prompt_league",
                _need("user_prompt_league", self.user_prompt_league),
            )

    def system_prompt(self) -> str:
        """Return the system prompt for the configured strategy and scoring mode."""
        if self.tournament_strategy == "simple_ranking":
            return self.system_prompt_ranking
        if self.scoring_mode == "logits":
            return self.system_prompt_league_logits
        return self.system_prompt_league

    def user_prompt_template(self) -> str:
        """Return the user prompt template for the configured strategy and scoring mode."""
        if self.tournament_strategy == "simple_ranking":
            return self.user_prompt_ranking
        if self.scoring_mode == "logits":
            return self.user_prompt_league_logits
        return self.user_prompt_league


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def load_tournament_config(
    path: str, override_path: str | None = None,
) -> TournamentConfig:
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

    merged = {**TOURNAMENT_DEFAULTS, **raw}

    valid_keys = set(TournamentConfig.__dataclass_fields__.keys())
    filtered = {k: v for k, v in merged.items() if k in valid_keys}

    return TournamentConfig(**filtered)
