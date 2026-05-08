"""
VLM Anomaly Classification Inference Script.

Runs batch inference on the Kaputt1 dataset using a Vision-Language Model
(default: Qwen3.5-9B) to classify images as defective or non-defective,
then computes AUROC, confusion matrices, and per-prompt metrics.

``shot_mode`` (in YAML):
  - ``zero_shot``: single query image (optionally resized with ``input_size``).
  - ``few_shot``: 2×2 grid — query top-left, three normal references (from
    ``reference-{split}.parquet``) in the other cells; final square side is
    ``input_size``. Query rows must include ``item_identifier``.

``scoring_mode``:
  - ``text``: greedy decoding; parse score from model output.
  - ``logits``: P(Yes) from next-token logits (no generation).

All prompts are defined in the YAML config (system + user strings for each
shot and scoring mode).

Usage:
    python scripts/run_vlm_inference.py --config configs/vlm/vlm_base.yaml
    python scripts/run_vlm_inference.py --config configs/vlm/vlm_base.yaml \\
        --override configs/vlm/vlm_few_logits.yaml configs/vlm/vlm_few_text.yaml
    python scripts/run_vlm_inference.py --config configs/vlm/vlm_base.yaml \\
        --override configs/vlm/vlm_few_text.yaml \\
        --session-dir experiments/session_20260103_120000
"""

import argparse
import logging
import os
import sys
import warnings
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from huggingface_hub import login as hf_login

from src.common.plotting import load_session_experiment_summaries, plot_session_comparison
from src.vlm.config import load_vlm_config
from src.vlm.experiment import run_vlm_experiment, setup_logging
from src.vlm.inference import ensure_pad_token_for_generation

logger = logging.getLogger("vlm")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_model(cfg):
    """
    Load the VLM and its processor onto the configured GPU.

    Tries the native ``Qwen3_5ForConditionalGeneration`` class first;
    falls back to the generic ``AutoModelForImageTextToText`` auto-class
    so that other VL architectures work out of the box.
    """
    from transformers import AutoProcessor

    try:
        from transformers import Qwen3_5ForConditionalGeneration
        ModelClass = Qwen3_5ForConditionalGeneration
        logger.info("Using Qwen3_5ForConditionalGeneration model class.")
    except ImportError:
        from transformers import AutoModelForImageTextToText
        ModelClass = AutoModelForImageTextToText
        logger.info(
            "Qwen3_5ForConditionalGeneration not available; "
            "falling back to AutoModelForImageTextToText."
        )

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    logger.info(f"Loading processor from '{cfg.model_name}' ...")
    processor = AutoProcessor.from_pretrained(
        cfg.model_name,
        trust_remote_code=True,
        cache_dir=cfg.cache_dir,
        min_pixels=cfg.min_pixels,
        max_pixels=cfg.max_pixels,
    )

    model_kwargs = dict(
        device_map=f"cuda:{cfg.gpu_id}",
        torch_dtype=dtype,
        trust_remote_code=True,
        cache_dir=cfg.cache_dir,
    )
    if cfg.load_in_4bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
        )
        model_kwargs.pop("torch_dtype", None)
        logger.info("4-bit quantization enabled (NF4).")

    logger.info(f"Loading model '{cfg.model_name}' ...")
    model = ModelClass.from_pretrained(cfg.model_name, **model_kwargs)
    model.eval()
    ensure_pad_token_for_generation(processor, model)

    return model, processor


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    # Drop stray transformers INFO lines about pad/eos (versions vary).
    class _DropPadEosLog(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            msg = record.getMessage()
            if "pad_token_id" in msg and "eos_token_id" in msg:
                return False
            return True

    _pad_filter = _DropPadEosLog()
    for _log_name in (
        "transformers",
        "transformers.generation",
        "transformers.generation.utils",
    ):
        logging.getLogger(_log_name).addFilter(_pad_filter)

    # Some builds emit a UserWarning from generation code; log filter catches INFO.
    warnings.filterwarnings(
        "ignore",
        message=r".*[Pp]ad_token_id.*[Ee]os_token_id.*",
    )

    parser = argparse.ArgumentParser(
        description=(
            "VLM anomaly classification inference on the Kaputt1 dataset."
        ),
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to the base YAML config file.",
    )
    parser.add_argument(
        "--override", type=str, nargs="*", default=None,
        help=(
            "One or more override YAML config files. The model is loaded "
            "once and each override is run as a separate experiment."
        ),
    )
    parser.add_argument(
        "--session-dir",
        type=str,
        default=None,
        help=(
            "Session folder: each run writes to <session-dir>/<exp_name>/. "
            "If this folder already exists (e.g. continuing later), "
            "session.log is appended and AUROC/AP comparison plots include "
            "every subfolder that has results.json. "
            "Default: create a new experiments/session_<timestamp> directory."
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

    base_cfg = load_vlm_config(args.config)

    # ---- GPU ----
    assert torch.cuda.is_available(), "CUDA is required but not available."
    device = f"cuda:{base_cfg.gpu_id}"
    torch.cuda.set_device(base_cfg.gpu_id)
    logger.info(f"Using GPU {base_cfg.gpu_id} ({device})")

    # ---- HuggingFace auth ----
    if args.hf_token:
        hf_login(token=args.hf_token)

    # ---- Load model once ----
    model, processor = _load_model(base_cfg)
    logger.info("Model and processor loaded.\n")

    # ---- Run experiments ----
    experiment_results: list[dict] = []

    for i, override_path in enumerate(overrides):
        if len(overrides) > 1:
            logger.info("")
            logger.info("#" * 60)
            logger.info(f"# Experiment {i + 1} / {len(overrides)}")
            logger.info("#" * 60)

        cfg = load_vlm_config(args.config, override_path)
        result = run_vlm_experiment(
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
            auroc_str = (
                f"{r['auroc']:.4f}" if not np.isnan(r["auroc"]) else "N/A"
            )
            ap_str = (
                f"{r['average_precision']:.4f}"
                if not np.isnan(r["average_precision"]) else "N/A"
            )
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
