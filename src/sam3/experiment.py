"""
Experiment orchestration: logging setup, single-experiment runner,
and JSON serialization of results.
"""

import json
import logging
import os

import numpy as np
from sklearn.metrics import average_precision_score

from src.sam3.config import Config
from src.common.data import load_and_filter_data, load_checkpoint_csv, write_inference_predictions_csv
from src.sam3.inference import run_inference
from src.common.metrics import (
    compute_auroc,
    compute_binary_metrics,
    compute_per_pair_metrics,
    compute_per_prompt_metrics,
    compute_type_max_scores,
    find_best_threshold,
    predict_classes,
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

logger = logging.getLogger("sam3")


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def setup_logging(session_dir: str, append: bool = False) -> None:
    """Configure session-level file logging."""
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


def add_experiment_log_handler(exp_dir: str) -> logging.FileHandler:
    """Add a per-experiment file handler. Returns the handler for later removal."""
    fh = logging.FileHandler(os.path.join(exp_dir, "experiment.log"), mode="w")
    fh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)
    return fh


def remove_log_handler(handler: logging.FileHandler) -> None:
    logger.removeHandler(handler)
    handler.close()


# ---------------------------------------------------------------------------
# Experiment serialisation
# ---------------------------------------------------------------------------

def _save_pair_scores(
    pair_scores: dict,
    labels: np.ndarray,
    exp_dir: str,
) -> None:
    """
    Persist raw per-(prompt, image) scores so that post-hoc threshold
    tuning can recompute metrics without re-running inference.
    Detections/boxes are omitted to keep the file small.
    """
    serializable: dict[str, dict[str, list]] = {}
    for (dtype, prompt), data in pair_scores.items():
        key = json.dumps([dtype, prompt])
        serializable[key] = {
            "indices": data["indices"],
            "scores": data["scores"],
        }
    payload = {
        "labels": labels.tolist(),
        "pair_scores": serializable,
    }
    out_path = os.path.join(exp_dir, "pair_scores.json")
    with open(out_path, "w") as f:
        json.dump(payload, f)
    logger.info(f"  Raw pair scores saved to {out_path}")


def save_experiment(
    cfg: Config,
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
    """Serialize full experiment results (config + metrics + failures) to JSON."""
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

    per_prompt_serializable = []
    for row in per_prompt:
        entry = dict(row)
        entry["ap"] = None if np.isnan(row["ap"]) else row["ap"]
        entry["auroc"] = None if np.isnan(row.get("auroc", float("nan"))) else row.get("auroc")
        entry["best_threshold"] = (
            None if np.isnan(row.get("best_threshold", float("nan")))
            else row.get("best_threshold")
        )
        entry.pop("fpr", None)
        entry.pop("tpr", None)
        entry.pop("roc_thresholds", None)
        per_prompt_serializable.append(entry)

    per_pair_serializable = []
    for row in per_pair:
        entry = dict(row)
        entry["success_images"] = [
            {"path": s["path"], "score": s["score"]}
            for s in row.get("success_images", [])
        ]
        per_pair_serializable.append(entry)

    auroc_serializable = None if (isinstance(auroc_value, float) and np.isnan(auroc_value)) else auroc_value
    ap_serializable = None if (isinstance(overall_ap, float) and np.isnan(overall_ap)) else overall_ap

    experiment = {
        "config_file": os.path.abspath(config_path),
        "override_file": os.path.abspath(override_path) if override_path else None,
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
            "batch_size": cfg.batch_size,
            "thresholds": cfg.thresholds,
            "mask_threshold": cfg.mask_threshold,
            "num_data": cfg.num_data,
            "gpu_id": cfg.gpu_id,
            "prompts": cfg.prompts,
        },
        "auroc": auroc_serializable,
        "average_precision": ap_serializable,
        "best_threshold": best_threshold,
        "per_threshold_metrics": per_threshold_metrics,
        "per_prompt": per_prompt_serializable,
        "per_pair": per_pair_serializable,
    }

    with open(out_path, "w") as f:
        json.dump(experiment, f, indent=2)
    logger.info(f"\nExperiment saved to {out_path}")


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_experiment(
    cfg: Config,
    model,
    processor,
    config_path: str,
    override_path: str | None,
    session_dir: str | None = None,
) -> dict | None:
    """
    Run a single experiment: filter data -> infer (all prompts, lowest threshold)
    -> compute AUROC & AP -> confusion matrices -> save results and diagrams.

    If *session_dir* is set (e.g. from the CLI), outputs go under
    ``session_dir / (exp_name or 'unnamed')``; otherwise under
    ``experiments / (exp_name or 'unnamed')`` at the repo root.

    Returns a summary dict for session-level comparison, or None on failure.
    """
    prompts_summary = (
        f"{len(cfg.prompts)} defect types, "
        f"{sum(len(v) for v in cfg.prompts.values())} prompts"
        if cfg.prompts else "(single-prompt mode)"
    )

    _code_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    _experiments_root = os.path.join(_code_root, "experiments")
    exp_parent = session_dir if session_dir is not None else _experiments_root
    exp_dir = os.path.join(exp_parent, cfg.exp_name or "unnamed")
    os.makedirs(exp_dir, exist_ok=True)
    exp_log_handler = add_experiment_log_handler(exp_dir)

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"Experiment: {cfg.exp_name or '(unnamed)'}")
    logger.info("=" * 60)
    logger.info(f"  config:         {config_path}")
    logger.info(f"  override:       {override_path or '(none)'}")
    logger.info(f"  data_dir:       {cfg.data_dir}")
    logger.info(f"  split:          {cfg.split}")
    logger.info(f"  model_name:     {cfg.model_name}")
    logger.info(f"  is_defect:      {cfg.is_defect}")
    logger.info(f"  major_defect:   {cfg.major_defect}")
    logger.info(f"  defect_type:    {cfg.defect_type}")
    logger.info(f"  mask:           {cfg.mask}")
    logger.info(f"  crop:           {cfg.crop}")
    logger.info(f"  input_size:     {cfg.input_size or '(original)'}")
    logger.info(f"  batch_size:     {cfg.batch_size}")
    logger.info(f"  thresholds:     {cfg.thresholds}")
    logger.info(f"  mask_threshold: {cfg.mask_threshold}")
    logger.info(f"  gpu_id:         {cfg.gpu_id}")
    logger.info(f"  num_data:       {cfg.num_data if cfg.num_data > 0 else '(all)'}")
    logger.info(f"  prompts:        {prompts_summary}")
    logger.info("=" * 60)

    df = load_and_filter_data(cfg)
    if len(df) == 0:
        logger.info("No samples remaining after filtering. Skipping.")
        remove_log_handler(exp_log_handler)
        return None

    # ---- Auto-detect checkpoint CSV from a previous interrupted run ----
    predictions_csv = os.path.join(exp_dir, "inference_predictions.csv")
    resume_data: dict[int, dict] | None = None
    if os.path.isfile(predictions_csv):
        resume_data = load_checkpoint_csv(predictions_csv)
        logger.info(f"  Resuming: loaded {len(resume_data)} rows from {predictions_csv}")
    max_scores, pair_scores = run_inference(
        df, model, processor, cfg,
        checkpoint_path=predictions_csv,
        samples_per_save=cfg.samples_per_save,
        resume_data=resume_data,
    )
    has_labels = "defect" in df.columns
    labels = df["defect"].astype(int).values if has_labels else None
    write_inference_predictions_csv(
        df, max_scores, labels, cfg.data_dir, cfg.crop, predictions_csv,
    )
    logger.info(f"  Per-image predictions saved to {predictions_csv}")

    # ---- Evaluation metrics (only when ground-truth labels are available) ----
    auroc_value = float("nan")
    overall_ap = float("nan")
    best_threshold = float("nan")
    best_metrics: dict = {}
    per_threshold_metrics: list[dict] = []
    per_prompt: list[dict] = []
    per_pair: list[dict] = []
    fpr: np.ndarray = np.array([])
    tpr: np.ndarray = np.array([])
    roc_thresholds: np.ndarray = np.array([])

    if not has_labels:
        logger.info(
            "No ground-truth 'defect' column in dataset — "
            "skipping evaluation metrics (test mode)."
        )
    else:
        num_positive = int(labels.sum())
        num_negative = int(len(labels) - num_positive)
        num_detected = int((max_scores > 0).sum())

        # ---- AUROC & AP (binary: defective vs non-defective) ----
        auroc_value, fpr, tpr, roc_thresholds = compute_auroc(labels, max_scores)
        if num_positive > 0 and num_negative > 0:
            overall_ap = average_precision_score(labels, max_scores)

        logger.info("")
        logger.info("=" * 60)
        logger.info("Binary Classification — AUROC & Average Precision")
        logger.info("=" * 60)
        logger.info(f"  Samples evaluated: {len(labels)}")
        logger.info(f"  Positive (defect): {num_positive}")
        logger.info(f"  Negative (normal): {num_negative}")
        logger.info(f"  Detected (score>0):{num_detected}")
        auroc_str = f"{auroc_value:.4f}" if not np.isnan(auroc_value) else "N/A"
        ap_str = f"{overall_ap:.4f}" if not np.isnan(overall_ap) else "N/A"
        logger.info(f"  AUROC:             {auroc_str}")
        logger.info(f"  Average Precision: {ap_str}")
        logger.info("=" * 60)

        # ---- Per-threshold binary metrics ----
        logger.info("")
        logger.info("=" * 60)
        logger.info("Classification Metrics per Threshold")
        logger.info("=" * 60)
        logger.info(f"  {'Thresh':>7s}  {'TP':>5s}  {'FP':>5s}  {'FN':>5s}  {'TN':>5s}"
                    f"  {'Acc':>7s}  {'Prec':>7s}  {'Rec':>7s}  {'F1':>7s}")
        logger.info("-" * 72)
        for t in cfg.thresholds:
            m = compute_binary_metrics(labels, max_scores, t)
            per_threshold_metrics.append(m)
            logger.info(
                f"  {t:>7.2f}  {m['tp']:>5d}  {m['fp']:>5d}  {m['fn']:>5d}  {m['tn']:>5d}"
                f"  {m['accuracy']:>7.4f}  {m['precision']:>7.4f}"
                f"  {m['recall']:>7.4f}  {m['f1']:>7.4f}"
            )
        logger.info("=" * 60)

        best_threshold, best_metrics = find_best_threshold(labels, max_scores, cfg.thresholds)
        logger.info(f"\n  Best threshold (by Precision): {best_threshold:.2f}  "
                    f"(Precision={best_metrics['precision']:.4f}, "
                    f"F1={best_metrics['f1']:.4f}, "
                    f"Recall={best_metrics['recall']:.4f})")

        # ---- Multi-class confusion matrices (one per threshold) ----
        type_max_scores: dict[str, np.ndarray] | None = None
        class_names: list[str] = []
        if cfg.prompts:
            type_max_scores = compute_type_max_scores(pair_scores, cfg.prompts, len(df))
            class_names = ["non_defective"] + sorted(cfg.prompts.keys())

            primary_defects = df["defect_types"].str.split(",").str[0].fillna("")
            true_labels: list[str] = []
            for is_defect, dtype in zip(df["defect"].values, primary_defects):
                if not is_defect:
                    true_labels.append("non_defective")
                else:
                    true_labels.append(dtype if dtype in cfg.prompts else dtype)

            for lbl in set(true_labels):
                if lbl not in class_names:
                    class_names.append(lbl)

            cm_dir = os.path.join(exp_dir, "confusion_matrices")
            os.makedirs(cm_dir, exist_ok=True)

            logger.info("\nGenerating confusion matrices per threshold...")
            for t in cfg.thresholds:
                pred_labels = predict_classes(type_max_scores, max_scores, t)
                plot_confusion_matrix(
                    true_labels, pred_labels, class_names, t,
                    os.path.join(cm_dir, f"cm_threshold_{t:.2f}.png"),
                )

        # ---- Per-prompt / per-pair analysis ----
        if pair_scores:
            per_prompt = compute_per_prompt_metrics(df, pair_scores, thresholds=cfg.thresholds)
            per_pair = compute_per_pair_metrics(df, pair_scores, crop=cfg.crop)

            logger.info("")
            logger.info("=" * 60)
            logger.info("Per-Prompt Metrics  (AP, AUROC, Best Threshold)")
            logger.info("=" * 60)
            logger.info(f"  {'Prompt':<35s} {'AP':>7s} {'AUROC':>7s} {'BestT':>7s}"
                        f" {'#Pos':>6s} {'#Neg':>6s}")
            logger.info("-" * 75)
            for row in per_prompt:
                ap_str = f"{row['ap']:.4f}" if not np.isnan(row["ap"]) else "   N/A"
                auroc_s = f"{row['auroc']:.4f}" if not np.isnan(row["auroc"]) else "   N/A"
                bt_s = f"{row['best_threshold']:.4f}" if not np.isnan(row["best_threshold"]) else "   N/A"
                logger.info(
                    f"  {row['prompt']:<35s} {ap_str:>7s} {auroc_s:>7s} {bt_s:>7s}"
                    f" {row['num_positive']:>6d} {row['num_negative']:>6d}"
                )
            logger.info("=" * 60)

            logger.info("")
            logger.info("=" * 60)
            logger.info("Mean Detection Score per (Defect Type, Prompt)")
            logger.info("=" * 60)
            logger.info(f"  {'Defect Type':<20s} {'Prompt':<35s} "
                        f"{'Mean':>7s} {'#Samples':>8s}")
            logger.info("-" * 75)
            for row in per_pair:
                logger.info(f"  {row['defect_type']:<20s} {row['prompt']:<35s} "
                            f"{row['mean_score']:>7.4f} {row['num_samples']:>8d}")
            logger.info("=" * 60)

    # ---- Save experiment ----
    if cfg.exp_name:
        save_experiment(
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
        if pair_scores and has_labels:
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

    remove_log_handler(exp_log_handler)

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
