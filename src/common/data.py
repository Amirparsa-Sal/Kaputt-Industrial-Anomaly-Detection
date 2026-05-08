"""
Data loading, filtering, image preparation, prompt-pair expansion,
and checkpoint utilities for failure recovery.
"""

import csv
import logging
import os
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

LOGICAL_DEFECT_TYPES = ["missing_unit", "actuation"]

logger = logging.getLogger("sam3")

# Type alias for (prompt_defect_type, prompt_text) keys used across modules.
PairKey = tuple[str, str]


def load_and_filter_data(cfg: Any) -> pd.DataFrame:
    """Load the query parquet and apply config filters for defect/major/type."""
    parquet_path = os.path.join(cfg.data_dir, f"query-{cfg.split}.parquet")
    df = pd.read_parquet(parquet_path)
    logger.info(f"Loaded {len(df)} samples from {parquet_path}")
    # Preserve parquet row index before any filtering (used in prediction CSVs).
    df = df.copy()
    df["original_index"] = df.index.to_numpy()

    has_labels = "defect" in df.columns

    if has_labels:
        if cfg.is_defect == "true":
            df = df[df["defect"]].copy()
        elif cfg.is_defect == "false":
            df = df[~df["defect"]].copy()

        if "major_defect" in df.columns:
            if cfg.major_defect == "true":
                df = df[~df["defect"] | df["major_defect"]].copy()
            elif cfg.major_defect == "false":
                df = df[~df["defect"] | ~df["major_defect"]].copy()

        if cfg.defect_type != "any" and "defect_types" in df.columns:
            primary = df["defect_types"].str.split(",").str[0].fillna("")
            if cfg.defect_type == "structural":
                keep = ~df["defect"] | ~primary.isin(LOGICAL_DEFECT_TYPES)
            elif cfg.defect_type == "logical":
                keep = ~df["defect"] | primary.isin(LOGICAL_DEFECT_TYPES)
            df = df[keep].copy()
    else:
        if cfg.is_defect != "any":
            logger.warning(
                f"is_defect={cfg.is_defect!r} ignored — no 'defect' column "
                "in dataset (test mode)."
            )
        if cfg.major_defect != "any":
            logger.warning(
                f"major_defect={cfg.major_defect!r} ignored — no "
                "'major_defect' column in dataset (test mode)."
            )
        if cfg.defect_type != "any":
            logger.warning(
                f"defect_type={cfg.defect_type!r} ignored — no 'defect' "
                "column in dataset (test mode)."
            )

    if cfg.num_data > 0 and len(df) > cfg.num_data:
        if has_labels:
            # Stratified sampling to preserve the defective / non-defective ratio
            defective = df[df["defect"]]
            non_defective = df[~df["defect"]]
            ratio = len(defective) / len(df)
            n_defective = round(cfg.num_data * ratio)
            n_non_defective = cfg.num_data - n_defective

            n_defective = min(n_defective, len(defective))
            n_non_defective = min(n_non_defective, len(non_defective))

            sampled = pd.concat([
                defective.sample(n=n_defective, random_state=42),
                non_defective.sample(n=n_non_defective, random_state=42),
            ])
            df = sampled.copy()
        else:
            df = df.sample(n=cfg.num_data, random_state=42).copy()

    logger.info(f"After filtering: {len(df)} samples")
    df = df.sort_values("original_index", kind="mergesort").reset_index(drop=True)
    return df


def prepare_image(
    image_path: str,
    base_dir: str,
    apply_mask: bool,
    input_size: int | None = None,
) -> Image.Image:
    """Load an image, optionally mask out the background, and resize."""
    full_path = os.path.join(base_dir, image_path)
    image = Image.open(full_path).convert("RGB")

    if apply_mask:
        mask_path = full_path.replace("image", "mask").replace(".jpg", ".png")
        gt_mask = np.array(Image.open(mask_path))
        image_np = np.array(image)
        image_np[(gt_mask <= 1)] = [0, 0, 0]
        image = Image.fromarray(image_np)

    if input_size is not None:
        image = image.resize((input_size, input_size), Image.LANCZOS)

    return image


def compose_vlm_few_shot_grid(
    query_image: Image.Image,
    reference_images: list[Image.Image],
    output_size: int,
) -> Image.Image:
    """
    Build a 2×2 montage: query (top-left), three references (top-right,
    bottom-left, bottom-right). Each cell is resized to fit; the canvas is
    then scaled to ``output_size`` × ``output_size`` if edge rounding leaves
    a small mismatch.
    """
    if len(reference_images) != 3:
        raise ValueError(
            f"Few-shot grid needs exactly 3 reference images, "
            f"got {len(reference_images)}."
        )
    if output_size < 2:
        raise ValueError("output_size must be at least 2 for a 2×2 grid.")

    cell = max(1, output_size // 2)
    canvas = Image.new("RGB", (cell * 2, cell * 2))

    tiles = [
        query_image,
        reference_images[0],
        reference_images[1],
        reference_images[2],
    ]
    positions = [(0, 0), (cell, 0), (0, cell), (cell, cell)]
    for tile, (x0, y0) in zip(tiles, positions):
        im = tile.resize((cell, cell), Image.LANCZOS)
        canvas.paste(im, (x0, y0))

    if canvas.size != (output_size, output_size):
        canvas = canvas.resize(
            (output_size, output_size), Image.LANCZOS,
        )
    return canvas


def build_reference_path_lookup(
    data_dir: str,
    split: str,
    crop: bool,
    *,
    pad_short: bool = True,
) -> dict[str, list[str]]:
    """
    Load ``reference-{split}.parquet`` and map ``item_identifier`` → three
    relative paths (``reference_crop`` or ``reference_image``).

    When *pad_short* is True (default), items with 1 or 2 references are
    padded to length 3 by **cycling** those paths (keeps the 2×2 grid and
    prompts unchanged). Items with 0 references still raise. More than three
    paths: the first three are kept after sorting.

    When *pad_short* is False, each item must have exactly three rows or a
    ``ValueError`` is raised.
    """
    log = logging.getLogger("vlm")
    ref_path = os.path.join(data_dir, f"reference-{split}.parquet")
    if not os.path.isfile(ref_path):
        raise FileNotFoundError(
            f"Few-shot mode requires {ref_path}."
        )
    ref_df = pd.read_parquet(ref_path)
    col = "reference_crop" if crop else "reference_image"
    if "item_identifier" not in ref_df.columns:
        raise ValueError(
            "Reference parquet must contain an 'item_identifier' column.",
        )
    if col not in ref_df.columns:
        raise ValueError(
            f"Reference parquet must contain '{col}' for crop={crop}.",
        )
    if "index" in ref_df.columns:
        ref_df = ref_df.sort_values("index", kind="mergesort")
    lookup: dict[str, list[str]] = {}
    reference_count_summary: dict[int, int] = {}
    for item_id, group in ref_df.groupby("item_identifier", sort=False):
        paths = group[col].tolist()
        key = str(item_id)
        n = len(paths)
        reference_count_summary[n] = reference_count_summary.get(n, 0) + 1

        if n == 0:
            raise ValueError(
                f"No reference images for item_identifier={key!r}.",
            )

        if n > 3:
            paths = paths[:3]
            n = 3

        if n < 3:
            if not pad_short:
                raise ValueError(
                    f"Expected exactly 3 reference images for "
                    f"item_identifier={key!r}, found {n}. "
                    f"Set pad_short_references: true to cycle shorter lists.",
                )
            original = list(paths)
            paths = [original[i % len(original)] for i in range(3)]

        lookup[key] = paths

    # Summarize reference availability once (instead of per-item logging).
    for n_ref in (3, 2, 1):
        count = reference_count_summary.get(n_ref, 0)
        log.info("%d references: %d images", n_ref, count)
    more_than_three = sum(
        count for n_ref, count in reference_count_summary.items() if n_ref > 3
    )
    if more_than_three > 0:
        log.info(">=4 references: %d images (using first 3)", more_than_three)

    return lookup


def build_all_prompt_pairs(
    df: pd.DataFrame,
    prompts: dict[str, list[str]] | None,
    crop: bool = False,
) -> list[tuple[int, str, str, str]]:
    """
    Expand ALL configured prompts for EVERY image, regardless of the image's
    actual defect type.  This removes the need to know the ground-truth defect
    type at inference time.

    Returns list of (image_index, image_path, prompt_text, prompt_defect_type).
    ``prompt_defect_type`` is the category the prompt belongs to in the config,
    NOT the image's actual defect type.
    """
    col = "query_crop" if crop else "query_image"
    image_paths = list(df[col])

    all_prompts: list[tuple[str, str]] = []
    if prompts:
        for dtype, prompt_list in prompts.items():
            for p in prompt_list:
                all_prompts.append((dtype, p))
    else:
        all_prompts = [("defect", "defect")]

    expanded: list[tuple[int, str, str, str]] = []
    for idx, path in enumerate(image_paths):
        for prompt_dtype, prompt in all_prompts:
            expanded.append((idx, path, prompt, prompt_dtype))
    return expanded


def write_inference_predictions_csv(
    df: pd.DataFrame,
    max_scores: np.ndarray,
    labels: np.ndarray | None,
    data_dir: str,
    crop: bool,
    out_path: str,
    model_outputs: list[str] | None = None,
) -> None:
    """
    Write one row per image: original parquet row index, absolute path, score,
    and binary label (when available).

    When *labels* is ``None`` (e.g. test split with no ground truth), the
    ``label`` column is omitted from the output CSV.

    When *model_outputs* is provided (same length as *df*), an extra column
    ``model_output`` stores a human-readable inference log (e.g. tournament
    match lines, rankings, and truncated raw VLM text depending on mode).

    ``max_scores`` and ``labels`` must align with ``df`` row order (same as
    inference loops). ``original_index`` comes from ``load_and_filter_data``
    (index before filtering); if absent, falls back to the row position in *df*.
    The path column matches ``build_all_prompt_pairs`` / VLM assignment:
    ``query_crop`` when *crop* else ``query_image``.
    """
    col = "query_crop" if crop else "query_image"
    has_orig = "original_index" in df.columns
    n = len(df)
    if model_outputs is not None and len(model_outputs) != n:
        raise ValueError(
            f"model_outputs length {len(model_outputs)} != df length {n}",
        )

    header = ["original_index", "image_path", "predicted_score"]
    if labels is not None:
        header.append("label")
    if model_outputs is not None:
        header.append("model_output")

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(n):
            orig_idx = df.iloc[i]["original_index"] if has_orig else i
            rel = df.iloc[i][col]
            full_path = os.path.abspath(os.path.join(data_dir, rel))
            row = [orig_idx, full_path, float(max_scores[i])]
            if labels is not None:
                row.append(int(labels[i]))
            if model_outputs is not None:
                row.append(model_outputs[i])
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Checkpoint utilities for failure recovery
# ---------------------------------------------------------------------------

def load_checkpoint_csv(csv_path: str) -> dict[int, dict]:
    """
    Load a checkpoint (or partial) predictions CSV from a previous
    interrupted run.

    Returns a dict mapping ``original_index`` → ``{"score": float,
    "model_output": str | None}``.  The ``model_output`` key is only
    present when the CSV includes that column.
    """
    result: dict[int, dict] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        has_model_output = "model_output" in (reader.fieldnames or [])
        for row in reader:
            orig_idx = int(row["original_index"])
            entry: dict = {"score": float(row["predicted_score"])}
            if has_model_output:
                entry["model_output"] = row.get("model_output", "")
            result[orig_idx] = entry
    return result


def write_checkpoint_csv(
    df: pd.DataFrame,
    max_scores: np.ndarray,
    labels: np.ndarray | None,
    completed_mask: np.ndarray,
    data_dir: str,
    crop: bool,
    out_path: str,
    model_outputs: list[str] | None = None,
) -> int:
    """
    Write only the *completed* rows to a checkpoint CSV.

    Same column format as :func:`write_inference_predictions_csv` so that
    the file can be used directly with ``--resume-csv`` or as a final
    predictions file if the experiment never finishes.

    When *labels* is ``None`` (e.g. test split with no ground truth), the
    ``label`` column is omitted from the output CSV.

    Parameters
    ----------
    completed_mask : np.ndarray[bool]
        Boolean array aligned with *df* — ``True`` for images whose
        inference is finished and whose score is final.

    Returns
    -------
    int
        Number of rows written.
    """
    col = "query_crop" if crop else "query_image"
    has_orig = "original_index" in df.columns
    n = len(df)

    header = ["original_index", "image_path", "predicted_score"]
    if labels is not None:
        header.append("label")
    has_model = model_outputs is not None
    if has_model:
        header.append("model_output")

    count = 0
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(n):
            if not completed_mask[i]:
                continue
            orig_idx = df.iloc[i]["original_index"] if has_orig else i
            rel = df.iloc[i][col]
            full_path = os.path.abspath(os.path.join(data_dir, rel))
            row = [orig_idx, full_path, float(max_scores[i])]
            if labels is not None:
                row.append(int(labels[i]))
            if has_model:
                row.append(model_outputs[i])
            writer.writerow(row)
            count += 1

    return count
