# Kaputt1 Defect Detection

Multi-pipeline anomaly detection on the [Kaputt1](https://huggingface.co/datasets/kaputt1) dataset, supporting SAM3 segmentation, VLM classification, and tournament-based ranking.

## Pipelines

| Pipeline | Description | Entry Script |
|----------|-------------|--------------|
| **SAM3** | Open-vocabulary segmentation for structural defect detection | `scripts/run_inference.py` |
| **VLM** | Vision-Language Model zero/few-shot anomaly classification | `scripts/run_vlm_inference.py` |
| **Tournament** | Comparative ranking via pairwise or all-at-once VLM prompts | `scripts/run_tournament_inference.py` |

## Project Structure

```
code_base/
├── configs/
│   ├── sam3/                   # SAM3 pipeline configs
│   │   ├── base.yaml
│   │   ├── crop.yaml
│   │   ├── no_mask.yaml
│   │   └── with_mask.yaml
│   ├── vlm/                    # VLM pipeline configs
│   │   ├── vlm_base.yaml
│   │   ├── vlm_few_logits.yaml
│   │   ├── vlm_few_text.yaml
│   │   ├── vlm_zero_logits.yaml
│   │   └── vlm_zero_text.yaml
│   └── tournament/             # Tournament pipeline configs
│       ├── tournament_base.yaml
│       ├── tournament_league_complete.yaml
│       ├── tournament_league_swiss.yaml
│       └── tournament_simple_ranking.yaml
├── src/
│   ├── common/                 # Shared utilities
│   │   ├── data.py             # Data loading, filtering, image prep
│   │   ├── metrics.py          # AUROC, AP, per-prompt metrics
│   │   └── plotting.py         # ROC curves, bar charts, grids
│   ├── sam3/                   # SAM3 segmentation pipeline
│   │   ├── config.py           # Config dataclass + YAML loader
│   │   ├── inference.py        # Batched SAM3 forward pass
│   │   └── experiment.py       # Experiment orchestration
│   ├── vlm/                    # VLM classification pipeline
│   │   ├── config.py           # VLM config dataclass
│   │   ├── inference.py        # Zero/few-shot VLM inference
│   │   └── experiment.py       # VLM experiment orchestration
│   └── tournament/             # Tournament scoring pipeline
│       ├── config.py           # Tournament config dataclass
│       ├── inference.py        # Ranking & league inference
│       └── experiment.py       # Tournament experiment orchestration
├── scripts/                    # CLI entry points
│   ├── run_inference.py        # SAM3 batch runner
│   ├── run_vlm_inference.py    # VLM batch runner
│   ├── run_tournament_inference.py  # Tournament batch runner
│   └── recompute_with_thresholds.py # Post-hoc threshold analysis
├── gui/                        # Gradio applications
│   ├── inference_gui.py        # Interactive SAM3 single-image GUI
│   └── agentic_inference_gui.py # Agentic SAM3 + VLM reasoning GUI
├── notebooks/                  # Exploratory Jupyter notebooks
├── experiments/                # Output artifacts (auto-generated)
├── resources/                  # Documentation assets
├── requirements.txt
├── gui_requirements.txt
└── README.md
```

## Requirements

- Python 3.10+
- CUDA-capable GPU
- Dependencies listed in `requirements.txt`

## Hugging Face authentication

Gated or private models need a Hugging Face access token. Tokens are **not** stored in YAML configs; pass them at runtime:

- **`--hf-token`** on any entry script that downloads from the Hub (batch runners, `recompute_with_thresholds.py`, and the Gradio GUIs).
- Or run **`huggingface-cli login`** once on the machine so the Hub client uses your cached credentials.

Example:

```bash
python scripts/run_inference.py --config configs/sam3/base.yaml --hf-token "$HF_TOKEN"
```

## Quick Start

```bash
# SAM3 pipeline (add --hf-token <token> if the model or dataset is gated)
python scripts/run_inference.py --config configs/sam3/base.yaml

# SAM3 with override
python scripts/run_inference.py --config configs/sam3/base.yaml --override configs/sam3/crop.yaml

# VLM pipeline
python scripts/run_vlm_inference.py --config configs/vlm/vlm_base.yaml

# Tournament pipeline
python scripts/run_tournament_inference.py --config configs/tournament/tournament_base.yaml \
    --override configs/tournament/tournament_simple_ranking.yaml

# Post-hoc threshold analysis
python scripts/recompute_with_thresholds.py --exp-dir experiments/<exp_name>

# Interactive GUI
python gui/inference_gui.py --config configs/sam3/base.yaml
```

## Config System

All pipelines use a **base + override** YAML merge strategy:

- **Scalar keys**: override replaces base value
- **Nested dicts** (e.g. `prompts`): deep-merged (add/replace keys without repeating the rest)
- **Missing keys**: inherited from base config defaults

## Output

When `exp_name` is set, experiments produce:

| File | Description |
|------|-------------|
| `results.json` | Full config, metrics (AUROC, AP, per-threshold), per-prompt analysis |
| `pair_scores.json` | Raw per-(prompt, image) scores for post-hoc recomputation |
| `inference_predictions.csv` | Per-image scores and labels |
| `auroc.png` | ROC curve with threshold markers |
| `ap_chart.png` | Average Precision bar chart by prompt |
| `confusion_matrices/` | Confusion matrices at each threshold |
| `failures/` / `successes/` | 3x3 image grids of worst/best detections |

Session-level comparison charts (AUROC, AP) are generated when multiple experiments share a session directory.
