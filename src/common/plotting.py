"""
Visualization functions for defect-detection experiments.

All plotting helpers are collected here: ROC curves, bar charts,
confusion matrices, heatmaps, image grids, and session comparisons.
"""

import json
import logging
import os
from collections import defaultdict

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src.common.data import PairKey

logger = logging.getLogger("sam3")


def plot_auroc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auroc: float,
    thresholds_to_mark: list[float],
    roc_thresholds: np.ndarray,
    out_path: str,
) -> None:
    """Plot ROC curve with AUC annotation and threshold markers."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(fpr, tpr, color="steelblue", linewidth=2, label=f"ROC (AUC = {auroc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="Random")

    for t in thresholds_to_mark:
        idx = np.argmin(np.abs(roc_thresholds - t))
        if idx < len(fpr):
            ax.plot(fpr[idx], tpr[idx], "o", markersize=8)
            ax.annotate(
                f"t={t:.2f}",
                xy=(fpr[idx], tpr[idx]),
                xytext=(fpr[idx] + 0.03, tpr[idx] - 0.03),
                fontsize=8,
                arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
            )

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve — Binary Defect Detection", fontsize=14)
    ax.legend(loc="lower right", fontsize=11)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  AUROC curve saved to {out_path}")


def plot_per_prompt_auroc_curves(
    per_prompt: list[dict], out_path: str,
) -> None:
    """
    Overlay ROC curves for every prompt on a single figure.
    Each curve is labelled with the prompt text and its AUC value.
    """
    valid = [r for r in per_prompt if len(r["fpr"]) > 0]
    if not valid:
        return

    fig, ax = plt.subplots(figsize=(10, 10))
    cmap = plt.cm.get_cmap("tab20", max(len(valid), 1))

    for i, row in enumerate(valid):
        auroc_val = row["auroc"]
        label = f'{row["prompt"]}  (AUC={auroc_val:.3f})'
        ax.plot(row["fpr"], row["tpr"], color=cmap(i), linewidth=1.4, label=label)

    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves per Prompt — Binary Defect Detection", fontsize=14)
    ax.legend(fontsize=7, loc="lower right", ncol=1)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Per-prompt AUROC curves saved to {out_path}")


def plot_auroc_by_prompt(per_prompt: list[dict], out_path: str) -> None:
    """
    Horizontal bar chart showing AUROC per prompt (sorted descending),
    with the best threshold annotated on each bar.
    """
    rows = sorted(
        per_prompt,
        key=lambda r: (0.0 if np.isnan(r["auroc"]) else r["auroc"]),
        reverse=True,
    )
    prompts = [r["prompt"] for r in rows]
    values = [r["auroc"] if not np.isnan(r["auroc"]) else 0.0 for r in rows]
    best_ts = [r["best_threshold"] for r in rows]

    fig_height = max(4, len(prompts) * 0.45)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    y_pos = np.arange(len(prompts))
    bars = ax.barh(y_pos, values, height=0.6, color="darkorange", edgecolor="white")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(prompts, fontsize=9)
    ax.invert_yaxis()

    for bar, val, bt in zip(bars, values, best_ts):
        if val > 0:
            t_str = f"{bt:.2f}" if not np.isnan(bt) else "?"
            ax.text(
                bar.get_width() + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}  (t={t_str})",
                va="center", ha="left", fontsize=8,
            )

    ax.set_xlabel("AUROC")
    ax.set_title("AUROC by Prompt  (best threshold annotated)")
    x_max = max(values) + 0.15 if values else 1.0
    ax.set_xlim(0, min(1.25, x_max))
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  AUROC-by-prompt chart saved to {out_path}")


def plot_confusion_matrix(
    true_labels: list[str],
    pred_labels: list[str],
    class_names: list[str],
    threshold: float,
    out_path: str,
) -> None:
    """Plot a confusion matrix heatmap for the given threshold."""
    from sklearn.metrics import confusion_matrix as sk_confusion_matrix

    cm = sk_confusion_matrix(true_labels, pred_labels, labels=class_names)

    fig_size = max(8, len(class_names) * 1.2)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, shrink=0.8)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)

    thresh_color = cm.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            color = "white" if cm[i, j] > thresh_color else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color=color, fontsize=10)

    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(f"Confusion Matrix (threshold={threshold:.2f})", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Confusion matrix saved to {out_path}")


def plot_ap_by_prompt(per_prompt: list[dict], out_path: str) -> None:
    """
    Horizontal bar chart showing Average Precision per prompt,
    sorted by AP descending.
    """
    rows = sorted(
        per_prompt,
        key=lambda r: (0.0 if np.isnan(r["ap"]) else r["ap"]),
        reverse=True,
    )
    prompts = [r["prompt"] for r in rows]
    values = [r["ap"] if not np.isnan(r["ap"]) else 0.0 for r in rows]

    fig_height = max(4, len(prompts) * 0.45)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    y_pos = np.arange(len(prompts))
    bars = ax.barh(y_pos, values, height=0.6, color="steelblue", edgecolor="white")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(prompts, fontsize=9)
    ax.invert_yaxis()

    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(
                bar.get_width() + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}",
                va="center", ha="left", fontsize=8,
            )

    ax.set_xlabel("Average Precision")
    ax.set_title("Average Precision by Prompt")
    x_max = max(values) + 0.1 if values else 1.0
    ax.set_xlim(0, min(1.15, x_max))
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  AP chart saved to {out_path}")


def plot_prompt_type_matrix(
    df: pd.DataFrame,
    pair_scores: dict[PairKey, dict[str, list]],
    out_path: str,
) -> None:
    """
    Heatmap of mean detection scores with rows = prompts and
    columns = true defect types + non_defective.

    Rows are sorted by discrimination:
        mean(score on target type) - mean(score on all other types).
    The target-type cell for each prompt is bordered in blue, and its
    value is printed in bold.
    """
    labels_all = df["defect"].astype(int).values
    primary_defects_all = (
        df["defect_types"].str.split(",").str[0].fillna("").values
    )

    true_types = sorted(set(primary_defects_all[labels_all == 1]) - {""})
    columns = true_types + ["non_defective"]

    prompt_data: dict[str, dict[str, list]] = defaultdict(
        lambda: {"indices": [], "scores": []}
    )
    prompt_config_type: dict[str, str] = {}
    for (dtype, prompt), data in pair_scores.items():
        prompt_data[prompt]["indices"].extend(data["indices"])
        prompt_data[prompt]["scores"].extend(data["scores"])
        if prompt not in prompt_config_type:
            prompt_config_type[prompt] = dtype

    matrix: dict[str, dict[str, float]] = {}
    disc: dict[str, float] = {}

    for prompt, pdata in prompt_data.items():
        indices = np.array(pdata["indices"])
        scores = np.array(pdata["scores"])
        labels = labels_all[indices]
        ptypes = primary_defects_all[indices]

        row: dict[str, float] = {}
        for col in columns:
            mask = (labels == 0) if col == "non_defective" else (ptypes == col)
            row[col] = float(scores[mask].mean()) if mask.sum() > 0 else 0.0
        matrix[prompt] = row

        cfg_type = prompt_config_type[prompt]
        mean_target = row.get(cfg_type, 0.0)
        other_vals = [v for k, v in row.items() if k != cfg_type]
        mean_other = float(np.mean(other_vals)) if other_vals else 0.0
        disc[prompt] = mean_target - mean_other

    sorted_prompts = sorted(
        prompt_data.keys(),
        key=lambda p: disc.get(p, 0.0),
        reverse=True,
    )

    data = np.zeros((len(sorted_prompts), len(columns)))
    for i, prompt in enumerate(sorted_prompts):
        for j, col in enumerate(columns):
            data[i, j] = matrix[prompt].get(col, 0.0)

    fig_height = max(6, len(sorted_prompts) * 0.55)
    fig_width = max(8, len(columns) * 1.4 + 4)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    im = ax.imshow(data, aspect="auto", cmap="YlOrRd", vmin=0)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Mean Detection Score")

    ax.set_xticks(np.arange(len(columns)))
    ax.set_xticklabels(columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(sorted_prompts)))
    ylabels = [
        f"{p}  (\u0394={disc[p]:+.3f})" for p in sorted_prompts
    ]
    ax.set_yticklabels(ylabels, fontsize=8)

    for i, prompt in enumerate(sorted_prompts):
        cfg_type = prompt_config_type[prompt]
        for j, col in enumerate(columns):
            val = data[i, j]
            color = "white" if val > data.max() * 0.65 else "black"
            weight = "bold" if col == cfg_type else "normal"
            ax.text(
                j, i, f"{val:.3f}", ha="center", va="center",
                color=color, fontsize=8, fontweight=weight,
            )

    for i, prompt in enumerate(sorted_prompts):
        cfg_type = prompt_config_type[prompt]
        if cfg_type in columns:
            j = columns.index(cfg_type)
            ax.add_patch(mpatches.Rectangle(
                (j - 0.5, i - 0.5), 1, 1,
                fill=False, edgecolor="blue", linewidth=2,
            ))

    ax.set_xlabel("True Defect Type", fontsize=12)
    ax.set_ylabel("Prompt", fontsize=12)
    ax.set_title(
        "Mean Detection Score Matrix (Prompt \u00d7 True Type)\n"
        "Sorted by: mean(target type) \u2212 mean(other types)  |  "
        "Target cell bordered in blue",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Prompt-type score matrix saved to {out_path}")


def plot_image_grid(
    per_pair: list[dict],
    data_dir: str,
    out_dir: str,
    image_key: str,
    title_prefix: str,
) -> None:
    """
    Generic 3x3 image grid plotter.

    Saves grids to ``<out_dir>/<defect_type>/<prompt>.png``.
    """
    for row in per_pair:
        items = row.get(image_key, [])[:9]
        if not items:
            continue

        n = len(items)
        cols = 3
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
        axes = np.array(axes).reshape(-1)

        dtype = row["defect_type"]
        prompt = row["prompt"]
        fig.suptitle(
            f"{title_prefix} — {dtype} / \"{prompt}\"",
            fontsize=13, fontweight="bold", y=1.02,
        )

        for idx, ax in enumerate(axes):
            if idx < n:
                item = items[idx]
                img_path = os.path.join(data_dir, item["path"])
                score = item["score"]
                try:
                    img = Image.open(img_path).convert("RGB")
                    ax.imshow(img)
                except FileNotFoundError:
                    ax.text(0.5, 0.5, "not found", ha="center", va="center",
                            transform=ax.transAxes, fontsize=10, color="red")

                dets = item.get("detections", {})
                det_boxes = dets.get("boxes", [])
                det_scores_list = dets.get("scores", [])
                for box, ds in zip(det_boxes, det_scores_list):
                    x1, y1, x2, y2 = box
                    rect = mpatches.Rectangle(
                        (x1, y1), x2 - x1, y2 - y1,
                        fill=False, edgecolor="lime", linewidth=2,
                    )
                    ax.add_patch(rect)
                    ax.text(
                        x1, max(y1 - 4, 0), f"{ds:.2f}",
                        color="white", fontsize=7,
                        bbox=dict(facecolor="black", alpha=0.6, pad=1),
                    )

                ax.set_title(f"score={score:.4f}", fontsize=9)
            ax.axis("off")

        safe_dtype = "".join(
            c if c.isalnum() or c in "-_ " else "_" for c in dtype
        ).strip().replace(" ", "_") or "non-defective"
        safe_prompt = "".join(
            c if c.isalnum() or c in "-_ " else "_" for c in prompt
        ).strip().replace(" ", "_")[:40]

        type_dir = os.path.join(out_dir, safe_dtype)
        os.makedirs(type_dir, exist_ok=True)
        fig.tight_layout()
        fig.savefig(
            os.path.join(type_dir, f"{safe_prompt}.png"),
            dpi=120, bbox_inches="tight",
        )
        plt.close(fig)

    logger.info(f"  {title_prefix} grids saved to {out_dir}/")


def load_session_experiment_summaries(session_dir: str) -> list[dict]:
    """
    Collect summary dicts from every immediate subdirectory of *session_dir*
    that contains ``results.json``, in sorted folder-name order.

    Each dict matches the structure expected by :func:`plot_session_comparison`
    (``exp_name``, ``auroc``, ``average_precision``, ``best_threshold``,
    ``best_precision``, ``best_recall``, ``best_f1``).

    Skips unreadable or incomplete ``results.json`` files.
    """
    if not os.path.isdir(session_dir):
        return []

    summaries: list[dict] = []
    for name in sorted(os.listdir(session_dir)):
        sub = os.path.join(session_dir, name)
        if not os.path.isdir(sub):
            continue
        rpath = os.path.join(sub, "results.json")
        if not os.path.isfile(rpath):
            continue
        try:
            with open(rpath, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            logger.warning("Could not read %s; skipping for session summary.", rpath)
            continue

        auroc = data.get("auroc")
        if auroc is None:
            auroc = float("nan")
        else:
            auroc = float(auroc)

        ap = data.get("average_precision")
        if ap is None:
            ap = float("nan")
        else:
            ap = float(ap)

        best_t = float(data.get("best_threshold", 0.5))
        best_prec = 0.0
        best_rec = 0.0
        best_f1 = 0.0
        for row in data.get("per_threshold_metrics") or []:
            try:
                t = float(row.get("threshold", -1.0))
            except (TypeError, ValueError):
                continue
            if abs(t - best_t) < 1e-5:
                best_prec = float(row.get("precision", 0.0))
                best_rec = float(row.get("recall", 0.0))
                best_f1 = float(row.get("f1", 0.0))
                break

        cfg = data.get("config")
        if isinstance(cfg, dict) and cfg.get("exp_name"):
            exp_name = str(cfg["exp_name"])
        else:
            exp_name = name

        summaries.append({
            "exp_name": exp_name,
            "auroc": auroc,
            "average_precision": ap,
            "best_threshold": best_t,
            "best_precision": best_prec,
            "best_recall": best_rec,
            "best_f1": best_f1,
        })

    return summaries


def plot_session_comparison(
    experiment_results: list[dict],
    session_dir: str,
) -> None:
    """
    Plot comparison diagrams across all experiments in the session:
    1) AUROC comparison bar chart
    2) Average Precision comparison bar chart
    """
    if len(experiment_results) < 2:
        return

    names = [r["exp_name"] for r in experiment_results]
    aurocs = [r["auroc"] for r in experiment_results]
    avg_precisions = [r["average_precision"] for r in experiment_results]

    x = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.5), 6))
    bars = ax.bar(x, aurocs, color="steelblue", edgecolor="white", width=0.6)
    for bar, val in zip(bars, aurocs):
        if not np.isnan(val):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("AUROC", fontsize=12)
    ax.set_title("AUROC Comparison Across Experiments", fontsize=14)
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(
        os.path.join(session_dir, "auroc_comparison.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.5), 6))
    bars = ax.bar(x, avg_precisions, color="coral", edgecolor="white", width=0.6)
    for bar, val in zip(bars, avg_precisions):
        if not np.isnan(val):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Average Precision", fontsize=12)
    ax.set_title("Average Precision Comparison Across Experiments", fontsize=14)
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(
        os.path.join(session_dir, "ap_comparison.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)

    logger.info(f"Session comparison diagrams saved to {session_dir}/")
