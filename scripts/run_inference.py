"""
SAM3 Structural Defect Detection Inference Script.

Runs batch inference on the Kaputt1 dataset using a SAM3 model from HuggingFace
and computes AUROC, confusion matrices, and per-prompt metrics for defect detection.

All parameters — data paths, filters, model settings, prompts, and experiment
metadata — are specified in a single YAML config file.

Usage:
    python scripts/run_inference.py --config configs/sam3/base.yaml
    python scripts/run_inference.py --config configs/sam3/base.yaml --override configs/sam3/crop.yaml \\
        --session-dir experiments/session_20260103_120000
"""

import argparse
import logging
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from huggingface_hub import login as hf_login
from transformers import Sam3Model, Sam3Processor

from src.sam3.config import load_config
from src.sam3.experiment import run_experiment, setup_logging
from src.common.plotting import load_session_experiment_summaries, plot_session_comparison

logger = logging.getLogger("sam3")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SAM3 defect detection inference on the Kaputt1 dataset."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the base YAML config file.",
    )
    parser.add_argument(
        "--override",
        type=str,
        nargs="*",
        default=None,
        help="One or more override YAML config files. The model is loaded "
        "once and each override is run as a separate experiment.",
    )
    parser.add_argument(
        "--session-dir",
        type=str,
        default=None,
        help=(
            "Session folder for experiment subdirs. If it already exists, "
            "session.log is appended and comparison charts use every "
            "results.json found. Default: experiments/session_<timestamp>."
        ),
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help=(
            "Hugging Face access token for gated/private models "
            "(optional; can also rely on ``huggingface-cli login``)."
        ),
    )
    args = parser.parse_args()

    overrides = args.override or [None]

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if args.session_dir:
        session_dir = os.path.abspath(args.session_dir)
        session_log = os.path.join(session_dir, "session.log")
        append_session = os.path.isfile(session_log)
    else:
        session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = os.path.join(
            repo_root, "experiments", f"session_{session_ts}",
        )
        append_session = False

    os.makedirs(session_dir, exist_ok=True)
    setup_logging(session_dir, append=append_session)
    if append_session:
        logger.info(
            "\n%s\n# Continuing session (appending session.log)\n%s\n",
            "#" * 60,
            "#" * 60,
        )

    base_cfg = load_config(args.config)

    # ---- GPU selection ----
    device = f"cuda:{base_cfg.gpu_id}"
    assert torch.cuda.is_available(), "CUDA is required but not available."
    torch.cuda.set_device(base_cfg.gpu_id)
    logger.info(f"Using GPU {base_cfg.gpu_id} ({device})")

    if args.hf_token:
        hf_login(token=args.hf_token)

    logger.info("Loading model and processor...")
    model = Sam3Model.from_pretrained(
        base_cfg.model_name, cache_dir=base_cfg.cache_dir
    ).to(device)
    processor = Sam3Processor.from_pretrained(
        base_cfg.model_name, cache_dir=base_cfg.cache_dir
    )
    logger.info("Model loaded.")

    experiment_results: list[dict] = []

    for i, override_path in enumerate(overrides):
        if len(overrides) > 1:
            logger.info("")
            logger.info("#" * 60)
            logger.info(f"# Experiment {i + 1} / {len(overrides)}")
            logger.info("#" * 60)

        cfg = load_config(args.config, override_path)
        result = run_experiment(
            cfg,
            model,
            processor,
            args.config,
            override_path,
            session_dir=session_dir,
        )
        if result is not None:
            experiment_results.append(result)
            csv_path = result.get("predictions_csv")
            if csv_path:
                logger.info(f"Predictions CSV: {csv_path}")

    # ---- Session-level comparison (all experiments under session_dir) ----
    all_summaries = load_session_experiment_summaries(session_dir)
    if len(all_summaries) >= 2:
        logger.info("\n" + "=" * 60)
        logger.info(
            "Session comparison — %d experiment(s) with results.json",
            len(all_summaries),
        )
        logger.info("=" * 60)
        for r in all_summaries:
            auroc_str = f"{r['auroc']:.4f}" if not np.isnan(r["auroc"]) else "N/A"
            ap_str = f"{r['average_precision']:.4f}" if not np.isnan(r["average_precision"]) else "N/A"
            logger.info(
                f"  {r['exp_name']:<30s}  AUROC={auroc_str}  AP={ap_str}  "
                f"Prec={r['best_precision']:.4f}  "
                f"F1={r['best_f1']:.4f}  "
                f"(best_t={r['best_threshold']:.2f})"
            )
        logger.info("=" * 60)
        plot_session_comparison(all_summaries, session_dir)

    logger.info(f"\nSession directory: {session_dir}/")


if __name__ == "__main__":
    main()
