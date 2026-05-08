"""
Metric computation for binary and multi-class defect classification.

Includes AUROC, Average Precision, per-prompt metrics, per-pair analysis,
and threshold selection helpers.
"""

from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    roc_curve,
)

from src.common.data import PairKey


def compute_type_max_scores(
    pair_scores: dict[PairKey, dict[str, list]],
    prompts: dict[str, list[str]],
    num_images: int,
) -> dict[str, np.ndarray]:
    """
    For each defect type in the prompts config, compute the maximum score
    across its prompts for each image.
    """
    type_max: dict[str, np.ndarray] = {}
    for dtype in prompts:
        scores = np.zeros(num_images, dtype=np.float64)
        for (ptype, _prompt), data in pair_scores.items():
            if ptype == dtype:
                for idx, s in zip(data["indices"], data["scores"]):
                    scores[idx] = max(scores[idx], s)
        type_max[dtype] = scores
    return type_max


def predict_classes(
    type_max_scores: dict[str, np.ndarray],
    max_scores: np.ndarray,
    threshold: float,
) -> list[str]:
    """
    Assign a predicted class to each image at the given threshold.
    If max_score < threshold the image is predicted as non_defective;
    otherwise the defect type whose prompts scored highest is assigned.
    """
    dtypes = list(type_max_scores.keys())
    scores_matrix = np.stack([type_max_scores[dt] for dt in dtypes], axis=0)
    best_type_indices = np.argmax(scores_matrix, axis=0)

    predictions: list[str] = []
    for i, max_s in enumerate(max_scores):
        if max_s < threshold:
            predictions.append("non_defective")
        else:
            predictions.append(dtypes[best_type_indices[i]])
    return predictions


def compute_binary_metrics(
    labels: np.ndarray, scores: np.ndarray, threshold: float,
) -> dict[str, float]:
    """Compute TP/FP/FN/TN/accuracy/precision/recall/F1 at a single threshold."""
    predictions = (scores >= threshold).astype(int)
    tp = int(((predictions == 1) & (labels == 1)).sum())
    tn = int(((predictions == 0) & (labels == 0)).sum())
    fp = int(((predictions == 1) & (labels == 0)).sum())
    fn = int(((predictions == 0) & (labels == 1)).sum())

    accuracy = (tp + tn) / len(labels) if len(labels) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "threshold": threshold,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compute_auroc(
    labels: np.ndarray, scores: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute AUROC and ROC curve for binary classification.
    Returns (auroc, fpr, tpr, roc_thresholds).
    """
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan"), np.array([]), np.array([]), np.array([])
    auroc = roc_auc_score(labels, scores)
    fpr, tpr, roc_thresholds = roc_curve(labels, scores)
    return auroc, fpr, tpr, roc_thresholds


def find_best_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    thresholds: list[float],
) -> tuple[float, dict]:
    """Find the threshold that maximises precision among the given candidates."""
    best_prec = -1.0
    best_threshold = thresholds[0]
    best_metrics: dict = {}
    for t in thresholds:
        m = compute_binary_metrics(labels, scores, t)
        if m["precision"] > best_prec:
            best_prec = m["precision"]
            best_threshold = t
            best_metrics = m
    return best_threshold, best_metrics


def compute_per_prompt_metrics(
    df: pd.DataFrame,
    pair_scores: dict[PairKey, dict[str, list]],
    thresholds: list[float] | None = None,
) -> list[dict]:
    """
    Compute Average Precision, AUROC, and best threshold per prompt.

    The best threshold is selected as the point on the prompt's ROC curve
    that maximises precision, giving each prompt its individually optimal
    operating point.
    """
    labels_all = df["defect"].astype(int).values

    prompt_data: dict[str, dict[str, list]] = defaultdict(
        lambda: {"indices": [], "scores": []}
    )
    for (_dtype, prompt), data in pair_scores.items():
        prompt_data[prompt]["indices"].extend(data["indices"])
        prompt_data[prompt]["scores"].extend(data["scores"])

    results = []
    for prompt, pdata in sorted(prompt_data.items()):
        p_indices = np.array(pdata["indices"])
        p_scores = np.array(pdata["scores"])
        p_labels = labels_all[p_indices]
        n_pos = int(p_labels.sum())
        n_neg = int(len(p_labels) - n_pos)

        if n_pos == 0 or n_neg == 0:
            ap = float("nan")
            p_auroc = float("nan")
            p_fpr, p_tpr, p_roc_t = np.array([]), np.array([]), np.array([])
            best_t = float("nan")
        else:
            ap = average_precision_score(p_labels, p_scores)
            p_auroc, p_fpr, p_tpr, p_roc_t = compute_auroc(p_labels, p_scores)
            candidates = thresholds if thresholds else [0.5]
            best_prec = -1.0
            best_t = candidates[0]
            for ct in candidates:
                preds = (p_scores >= ct).astype(int)
                tp = int(((preds == 1) & (p_labels == 1)).sum())
                fp = int(((preds == 1) & (p_labels == 0)).sum())
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                if prec > best_prec:
                    best_prec = prec
                    best_t = float(ct)

        results.append({
            "prompt": prompt,
            "ap": ap,
            "auroc": p_auroc,
            "best_threshold": best_t,
            "fpr": p_fpr,
            "tpr": p_tpr,
            "roc_thresholds": p_roc_t,
            "num_samples": len(p_labels),
            "num_positive": n_pos,
            "num_negative": n_neg,
        })

    return results


def compute_per_pair_metrics(
    df: pd.DataFrame,
    pair_scores: dict[PairKey, dict[str, list]],
    num_samples: int = 9,
    crop: bool = False,
) -> list[dict]:
    """
    Compute mean detection score and collect failure/success images for
    each (defect_type, prompt) pair.
    """
    labels_all = df["defect"].astype(int).values
    col = "query_crop" if crop else "query_image"
    image_paths_all = df[col].values

    results = []

    for (dtype, prompt), data in sorted(pair_scores.items()):
        indices = np.array(data["indices"])
        scores = np.array(data["scores"])
        labels = labels_all[indices]

        num_pos = int(labels.sum())
        num_neg = int(len(labels) - num_pos)
        mean_score = float(scores.mean())

        detections = data.get("detections", [])

        if num_pos > 0:
            positive_mask = labels == 1
            pos_positions = np.where(positive_mask)[0]
            pos_indices = indices[positive_mask]
            pos_scores = scores[positive_mask]

            worst_order = np.argsort(pos_scores)[:num_samples]
            failed_images = [
                {"path": str(image_paths_all[pos_indices[i]]),
                 "score": float(pos_scores[i])}
                for i in worst_order
            ]

            best_order = np.argsort(pos_scores)[::-1][:num_samples]
            success_images = [
                {"path": str(image_paths_all[pos_indices[i]]),
                 "score": float(pos_scores[i]),
                 "detections": detections[pos_positions[i]] if detections else {}}
                for i in best_order
            ]
        else:
            worst_order = np.argsort(scores)[::-1][:num_samples]
            failed_images = [
                {"path": str(image_paths_all[indices[i]]),
                 "score": float(scores[i]),
                 "detections": detections[i] if detections else {}}
                for i in worst_order
            ]
            success_images = []

        results.append({
            "defect_type": dtype or "(non-defective)",
            "prompt": prompt,
            "mean_score": mean_score,
            "num_samples": len(labels),
            "num_positive": num_pos,
            "num_negative": num_neg,
            "failed_images": failed_images,
            "success_images": success_images,
        })

    return results
