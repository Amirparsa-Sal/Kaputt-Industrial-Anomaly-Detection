"""
Experiment orchestration for tournament-based VLM anomaly detection.

Mirrors ``vlm_experiment.py`` but calls the tournament inference pipeline
instead of the standard VLM inference, and logs tournament-specific
configuration (strategy, num_references, use_grid, repeat).
"""

import json
import logging
import os

import numpy as np
from sklearn.metrics import average_precision_score

from src.common.data import PairKey, load_and_filter_data, load_checkpoint_csv, write_inference_predictions_csv
from src.common.metrics import (
    compute_auroc,
    compute_binary_metrics,
    compute_per_pair_metrics,
    compute_per_prompt_metrics,
    find_best_threshold,
)
from src.common.plotting import (
    plot_ap_by_prompt,
    plot_auroc_by_prompt,
    plot_auroc_curve,
    plot_confusion_matrix,
    plot_image_grid,
    plot_per_prompt_auroc_curves,
    plot_prompt_type_matrix,
)
from src.tournament.config import TournamentConfig
from src.tournament.inference import run_tournament_inference

logger = logging.getLogger("vlm")


# ---------------------------------------------------------------------------
# Logging helpers (reuse the vlm logger namespace)
# ---------------------------------------------------------------------------

def setup_logging(session_dir: str, append: bool = False) -> None:
    """Configure session-level file + console logging."""
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    mode = "a" if append else "w"
    fh = logging.FileHandler(
        os.path.join(session_dir, "session.log"),
        mode=mode,
        encoding="utf-8",
    )
    fh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)


def _add_experiment_handler(exp_dir: str) -> logging.FileHandler:
    fh = logging.FileHandler(
        os.path.join(exp_dir, "experiment.log"), mode="w",
    )
    fh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)
    return fh


def _remove_handler(handler: logging.FileHandler) -> None:
    logger.removeHandler(handler)
    handler.close()


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def _save_pair_scores(
    pair_scores: dict[PairKey, dict[str, list]],
    labels: np.ndarray,
    exp_dir: str,
) -> None:
    """Persist raw per-(strategy, image) scores for post-hoc analysis."""
    serializable: dict[str, dict[str, list]] = {}
    for (strategy, mode), data in pair_scores.items():
        key = json.dumps([strategy, mode])
        serializable[key] = {
            "indices": data["indices"],
            "scores": data["scores"],
        }
    payload = {"labels": labels.tolist(), "pair_scores": serializable}
    out = os.path.join(exp_dir, "pair_scores.json")
    with open(out, "w") as f:
        json.dump(payload, f)
    logger.info("  Raw pair scores saved to %s", out)


def _save_experiment(
    cfg: TournamentConfig,
    config_path: str,
    override_path: str | None,
    auroc_value: float,
    overall_ap: float,
    per_threshold_metrics: list[dict],
    best_threshold: float,
    per_prompt: list[dict],
    per_pair: list[dict],
    pair_scores: dict | None = None,
    labels: np.ndarray | None = None,
    exp_dir: str | None = None,
) -> None:
    """Serialize full tournament experiment results to JSON."""
    if exp_dir is None:
        exp_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "experiments",
            cfg.exp_name,
        )
    os.makedirs(exp_dir, exist_ok=True)
    out_path = os.path.join(exp_dir, "results.json")

    if pair_scores is not None and labels is not None:
        _save_pair_scores(pair_scores, labels, exp_dir)

    per_prompt_ser = []
    for row in per_prompt:
        entry = dict(row)
        entry["ap"] = None if np.isnan(row["ap"]) else row["ap"]
        entry["auroc"] = (
            None if np.isnan(row.get("auroc", float("nan")))
            else row.get("auroc")
        )
        entry["best_threshold"] = (
            None if np.isnan(row.get("best_threshold", float("nan")))
            else row.get("best_threshold")
        )
        entry.pop("fpr", None)
        entry.pop("tpr", None)
        entry.pop("roc_thresholds", None)
        per_prompt_ser.append(entry)

    per_pair_ser = []
    for row in per_pair:
        entry = dict(row)
        entry["success_images"] = [
            {"path": s["path"], "score": s["score"]}
            for s in row.get("success_images", [])
        ]
        per_pair_ser.append(entry)

    auroc_ser = (
        None if isinstance(auroc_value, float) and np.isnan(auroc_value)
        else auroc_value
    )
    ap_ser = (
        None if isinstance(overall_ap, float) and np.isnan(overall_ap)
        else overall_ap
    )

    experiment = {
        "config_file": os.path.abspath(config_path),
        "override_file": (
            os.path.abspath(override_path) if override_path else None
        ),
        "config": {
            "data_dir": cfg.data_dir,
            "split": cfg.split,
            "model_name": cfg.model_name,
            "is_defect": cfg.is_defect,
            "major_defect": cfg.major_defect,
            "defect_type": cfg.defect_type,
            "mask": cfg.mask,
            "crop": cfg.crop,
            "input_size": cfg.input_size,
            "gpu_id": cfg.gpu_id,
            "num_data": cfg.num_data,
            "thresholds": cfg.thresholds,
            "tournament_strategy": cfg.tournament_strategy,
            "scoring_mode": cfg.scoring_mode,
            "league_type": cfg.league_type,
            "num_references": cfg.num_references,
            "use_grid": cfg.use_grid,
            "repeat": cfg.repeat,
            "temperature": cfg.temperature,
            "max_new_tokens": cfg.max_new_tokens,
            "load_in_4bit": cfg.load_in_4bit,
            "enable_thinking": cfg.enable_thinking,
        },
        "auroc": auroc_ser,
        "average_precision": ap_ser,
        "best_threshold": best_threshold,
        "per_threshold_metrics": per_threshold_metrics,
        "per_prompt": per_prompt_ser,
        "per_pair": per_pair_ser,
    }

    with open(out_path, "w") as f:
        json.dump(experiment, f, indent=2)
    logger.info("\nExperiment saved to %s", out_path)


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_tournament_experiment(
    cfg: TournamentConfig,
    model,
    processor,
    config_path: str,
    override_path: str | None,
    session_dir: str | None = None,
) -> dict | None:
    """
    Run a single tournament experiment:
      filter data → tournament inference → AUROC / AP → confusion matrices
      → save results and visualisations.

    Returns a summary dict for session-level comparison, or ``None`` on
    failure.
    """
    _code_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    _experiments_root = os.path.join(_code_root, "experiments")
    exp_parent = session_dir if session_dir is not None else _experiments_root
    exp_dir = os.path.join(exp_parent, cfg.exp_name or "unnamed")
    os.makedirs(exp_dir, exist_ok=True)
    exp_handler = _add_experiment_handler(exp_dir)

    logger.info("")
    logger.info("=" * 60)
    logger.info("Tournament Experiment: %s", cfg.exp_name or "(unnamed)")
    logger.info("=" * 60)
    logger.info("  config:           %s", config_path)
    logger.info("  override:         %s", override_path or "(none)")
    logger.info("  data_dir:         %s", cfg.data_dir)
    logger.info("  split:            %s", cfg.split)
    logger.info("  model_name:       %s", cfg.model_name)
    logger.info("  strategy:         %s", cfg.tournament_strategy)
    logger.info("  scoring_mode:     %s", cfg.scoring_mode)
    if cfg.tournament_strategy == "league":
        logger.info("  league_type:      %s", cfg.league_type)
    logger.info("  num_references:   %d", cfg.num_references)
    logger.info("  use_grid:         %s", cfg.use_grid)
    logger.info("  repeat:           %d", cfg.repeat)
    logger.info("  temperature:      %s", cfg.temperature)
    logger.info("  is_defect:        %s", cfg.is_defect)
    logger.info("  major_defect:     %s", cfg.major_defect)
    logger.info("  defect_type:      %s", cfg.defect_type)
    logger.info("  mask:             %s", cfg.mask)
    logger.info("  crop:             %s", cfg.crop)
    logger.info("  input_size:       %s", cfg.input_size)
    logger.info("  load_in_4bit:     %s", cfg.load_in_4bit)
    logger.info("  thresholds:       %s", cfg.thresholds)
    logger.info("  gpu_id:           %d", cfg.gpu_id)
    logger.info(
        "  num_data:         %s",
        cfg.num_data if cfg.num_data > 0 else "(all)",
    )
    logger.info("=" * 60)

    # ---- Load & filter data ----
    df = load_and_filter_data(cfg)
    if len(df) == 0:
        logger.info("No samples remaining after filtering. Skipping.")
        _remove_handler(exp_handler)
        return None

    # ---- Auto-detect checkpoint CSV from a previous interrupted run ----
    predictions_csv = os.path.join(exp_dir, "inference_predictions.csv")
    resume_data: dict[int, dict] | None = None
    if os.path.isfile(predictions_csv):
        resume_data = load_checkpoint_csv(predictions_csv)
        logger.info("  Resuming: loaded %d rows from %s", len(resume_data), predictions_csv)
    max_scores, pair_scores, text_outputs = run_tournament_inference(
        df, model, processor, cfg,
        checkpoint_path=predictions_csv,
        samples_per_save=cfg.samples_per_save,
        resume_data=resume_data,
    )
    labels = df["defect"].astype(int).values
    write_inference_predictions_csv(
        df,
        max_scores,
        labels,
        cfg.data_dir,
        cfg.crop,
        predictions_csv,
        model_outputs=text_outputs,
    )
    logger.info("  Per-image predictions saved to %s", predictions_csv)

    num_positive = int(labels.sum())
    num_negative = int(len(labels) - num_positive)
    num_detected = int((max_scores > 0.5).sum())

    # ---- AUROC & AP ----
    auroc_value, fpr, tpr, roc_thresholds = compute_auroc(labels, max_scores)
    if num_positive > 0 and num_negative > 0:
        overall_ap = average_precision_score(labels, max_scores)
    else:
        overall_ap = float("nan")

    logger.info("")
    logger.info("=" * 60)
    logger.info("Binary Classification — AUROC & Average Precision")
    logger.info("=" * 60)
    logger.info("  Samples evaluated: %d", len(labels))
    logger.info("  Positive (defect): %d", num_positive)
    logger.info("  Negative (normal): %d", num_negative)
    logger.info("  Detected (score>0.5): %d", num_detected)
    auroc_str = f"{auroc_value:.4f}" if not np.isnan(auroc_value) else "N/A"
    ap_str = f"{overall_ap:.4f}" if not np.isnan(overall_ap) else "N/A"
    logger.info("  AUROC:             %s", auroc_str)
    logger.info("  Average Precision: %s", ap_str)
    logger.info("=" * 60)

    # ---- Per-threshold binary metrics ----
    per_threshold_metrics: list[dict] = []
    logger.info("")
    logger.info("=" * 60)
    logger.info("Classification Metrics per Threshold")
    logger.info("=" * 60)
    logger.info(
        "  %7s  %5s  %5s  %5s  %5s  %7s  %7s  %7s  %7s",
        "Thresh", "TP", "FP", "FN", "TN", "Acc", "Prec", "Rec", "F1",
    )
    logger.info("-" * 72)
    for t in cfg.thresholds:
        m = compute_binary_metrics(labels, max_scores, t)
        per_threshold_metrics.append(m)
        logger.info(
            "  %7.2f  %5d  %5d  %5d  %5d  %7.4f  %7.4f  %7.4f  %7.4f",
            t, m["tp"], m["fp"], m["fn"], m["tn"],
            m["accuracy"], m["precision"], m["recall"], m["f1"],
        )
    logger.info("=" * 60)

    best_threshold, best_metrics = find_best_threshold(
        labels, max_scores, cfg.thresholds,
    )
    logger.info(
        "\n  Best threshold (by Precision): %.2f  "
        "(Precision=%.4f, F1=%.4f, Recall=%.4f)",
        best_threshold,
        best_metrics["precision"],
        best_metrics["f1"],
        best_metrics["recall"],
    )

    # ---- Confusion matrices ----
    cm_dir = os.path.join(exp_dir, "confusion_matrices")
    os.makedirs(cm_dir, exist_ok=True)

    binary_true = [
        "defective" if d else "non_defective" for d in df["defect"].values
    ]
    for t in cfg.thresholds:
        binary_pred = [
            "defective" if s >= t else "non_defective"
            for s in max_scores
        ]
        plot_confusion_matrix(
            binary_true, binary_pred,
            ["non_defective", "defective"], t,
            os.path.join(cm_dir, f"cm_binary_{t:.2f}.png"),
        )

    # ---- Per-prompt / per-pair analysis ----
    per_prompt: list[dict] = []
    per_pair: list[dict] = []
    if pair_scores:
        per_prompt = compute_per_prompt_metrics(
            df, pair_scores, thresholds=cfg.thresholds,
        )
        per_pair = compute_per_pair_metrics(
            df, pair_scores, crop=cfg.crop,
        )

        if per_prompt:
            logger.info("")
            logger.info("=" * 60)
            logger.info("Per-Strategy Metrics  (AP, AUROC, Best Threshold)")
            logger.info("=" * 60)
            logger.info(
                "  %-35s %7s %7s %7s %6s %6s",
                "Strategy", "AP", "AUROC", "BestT", "#Pos", "#Neg",
            )
            logger.info("-" * 75)
            for row in per_prompt:
                ap_s = (
                    f"{row['ap']:.4f}"
                    if not np.isnan(row["ap"]) else "   N/A"
                )
                auroc_s = (
                    f"{row['auroc']:.4f}"
                    if not np.isnan(row["auroc"]) else "   N/A"
                )
                bt_s = (
                    f"{row['best_threshold']:.4f}"
                    if not np.isnan(row["best_threshold"]) else "   N/A"
                )
                logger.info(
                    "  %-35s %7s %7s %7s %6d %6d",
                    row["prompt"], ap_s, auroc_s, bt_s,
                    row["num_positive"], row["num_negative"],
                )
            logger.info("=" * 60)

    # ---- Save & plot ----
    if cfg.exp_name:
        _save_experiment(
            cfg, config_path, override_path,
            auroc_value, overall_ap, per_threshold_metrics, best_threshold,
            per_prompt, per_pair,
            pair_scores=pair_scores, labels=labels,
            exp_dir=exp_dir,
        )

        logger.info("Saving visualizations...")

        if len(fpr) > 0:
            plot_auroc_curve(
                fpr, tpr, auroc_value,
                cfg.thresholds, roc_thresholds,
                os.path.join(exp_dir, "auroc.png"),
            )

        if per_prompt:
            plot_ap_by_prompt(
                per_prompt,
                os.path.join(exp_dir, "ap_chart.png"),
            )
            plot_per_prompt_auroc_curves(
                per_prompt,
                os.path.join(exp_dir, "auroc_per_prompt.png"),
            )
            plot_auroc_by_prompt(
                per_prompt,
                os.path.join(exp_dir, "auroc_by_prompt_chart.png"),
            )

        if pair_scores:
            plot_prompt_type_matrix(
                df, pair_scores,
                os.path.join(exp_dir, "mean_score_matrix.png"),
            )

        if per_pair:
            plot_image_grid(
                per_pair, cfg.data_dir,
                os.path.join(exp_dir, "failures"),
                image_key="failed_images",
                title_prefix="Failures",
            )
            plot_image_grid(
                per_pair, cfg.data_dir,
                os.path.join(exp_dir, "successes"),
                image_key="success_images",
                title_prefix="Successes",
            )

    _remove_handler(exp_handler)

    return {
        "exp_name": cfg.exp_name or "(unnamed)",
        "auroc": auroc_value,
        "average_precision": overall_ap,
        "best_threshold": best_threshold,
        "best_precision": best_metrics.get("precision", 0.0),
        "best_recall": best_metrics.get("recall", 0.0),
        "best_f1": best_metrics.get("f1", 0.0),
        "predictions_csv": predictions_csv,
        "exp_dir": exp_dir,
        "session_dir": session_dir,
    }
