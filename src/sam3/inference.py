"""
Batched SAM3 model inference with checkpoint/resume support.
"""

import logging
import time

import numpy as np
import torch
from tqdm import tqdm
from transformers import Sam3Model, Sam3Processor

from src.sam3.config import Config
from src.common.data import PairKey, build_all_prompt_pairs, prepare_image, write_checkpoint_csv

logger = logging.getLogger("sam3")


def run_inference(
    df,
    model: Sam3Model,
    processor: Sam3Processor,
    cfg: Config,
    *,
    checkpoint_path: str | None = None,
    samples_per_save: int = 0,
    resume_data: dict[int, dict] | None = None,
) -> tuple[np.ndarray, dict[PairKey, dict[str, list]]]:
    """
    Run batched SAM3 inference using ALL prompts on ALL images.
    Uses the lowest configured threshold for maximum detection sensitivity.

    Parameters
    ----------
    checkpoint_path : str | None
        If set together with *samples_per_save*, the predictions CSV is
        written to this path every *samples_per_save* newly completed
        images so that partial results survive a crash.
    samples_per_save : int
        Number of newly completed images between periodic checkpoint
        writes.  ``0`` (default) disables periodic saving.
    resume_data : dict[int, dict] | None
        Pre-computed results loaded from a previous checkpoint CSV via
        :func:`~src.common.data.load_checkpoint_csv`.  Keys are
        ``original_index`` values; each value has at least ``"score"``.
        Images present in *resume_data* are skipped during inference.

    Returns
    -------
    max_scores : np.ndarray
        One aggregated (max-over-all-prompts) prediction score per image.
    pair_scores : dict[(prompt_defect_type, prompt) -> {"indices", "scores", "detections"}]
        Raw per-image scores broken down by each (prompt_defect_type, prompt) pair.
        **Note:** resumed images do not contribute to *pair_scores*
        because per-prompt breakdown is not stored in the checkpoint CSV.
    """
    expanded = build_all_prompt_pairs(df, cfg.prompts, crop=cfg.crop)
    num_images = len(df)
    total_pairs_original = len(expanded)
    prompts_per_image = int(total_pairs_original / num_images) if num_images else 0

    inference_threshold = min(cfg.thresholds)

    device = f"cuda:{cfg.gpu_id}"
    max_scores = np.zeros(num_images, dtype=np.float64)
    pair_scores: dict[PairKey, dict[str, list]] = {}

    # --- Resume: pre-fill scores and filter out already-processed images ---
    orig_to_row: dict[int, int] = {}
    has_orig = "original_index" in df.columns
    for i in range(num_images):
        orig_idx = int(df.iloc[i]["original_index"]) if has_orig else i
        orig_to_row[orig_idx] = i

    resumed_rows: set[int] = set()
    if resume_data:
        for orig_idx, data in resume_data.items():
            if orig_idx in orig_to_row:
                row_idx = orig_to_row[orig_idx]
                max_scores[row_idx] = data["score"]
                resumed_rows.add(row_idx)
        logger.info(
            f"Resumed {len(resumed_rows)} / {num_images} images from checkpoint."
        )
        expanded = [p for p in expanded if p[0] not in resumed_rows]

    total_pairs = len(expanded)

    logger.info(
        f"Expanded to {total_pairs} (image, prompt) pairs "
        f"({prompts_per_image:.1f} prompts/image)"
    )
    logger.info(f"Using inference threshold: {inference_threshold}")

    # --- Checkpoint tracking ---
    labels = df["defect"].astype(int).values
    completed_mask = np.zeros(num_images, dtype=bool)
    prompts_done = np.zeros(num_images, dtype=int)
    for ri in resumed_rows:
        completed_mask[ri] = True
    last_save_count = 0

    num_batches = (total_pairs + cfg.batch_size - 1) // cfg.batch_size
    total_inference_time = 0.0

    for batch_idx in tqdm(range(num_batches), desc="Inference"):
        start = batch_idx * cfg.batch_size
        end = min(total_pairs, start + cfg.batch_size)
        batch = expanded[start:end]

        batch_images = [
            prepare_image(path, cfg.data_dir, cfg.mask, cfg.input_size)
            for _, path, _, _ in batch
        ]
        batch_prompts = [prompt for _, _, prompt, _ in batch]

        inputs = processor(
            images=batch_images,
            text=batch_prompts,
            input_boxes=None,
            input_boxes_labels=None,
            return_tensors="pt",
        ).to(device)

        t0 = time.time()
        with torch.no_grad():
            outputs = model(**inputs)
        total_inference_time += time.time() - t0

        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=inference_threshold,
            mask_threshold=cfg.mask_threshold,
            target_sizes=inputs.get("original_sizes").tolist(),
        )

        for pair, result in zip(batch, results):
            img_idx, _, prompt, prompt_dtype = pair
            score = (
                float(result["scores"].max()) if len(result["scores"]) > 0 else 0.0
            )
            max_scores[img_idx] = max(max_scores[img_idx], score)

            boxes = (
                result["boxes"].detach().float().cpu().tolist()
                if len(result["scores"]) > 0 else []
            )
            det_scores = (
                result["scores"].detach().float().cpu().tolist()
                if len(result["scores"]) > 0 else []
            )

            key: PairKey = (prompt_dtype, prompt)
            if key not in pair_scores:
                pair_scores[key] = {
                    "indices": [], "scores": [], "detections": [],
                }
            pair_scores[key]["indices"].append(img_idx)
            pair_scores[key]["scores"].append(score)
            pair_scores[key]["detections"].append({
                "boxes": boxes, "scores": det_scores,
            })

            # Track per-image prompt completion
            prompts_done[img_idx] += 1
            if prompts_per_image > 0 and prompts_done[img_idx] >= prompts_per_image:
                completed_mask[img_idx] = True

        # --- Periodic checkpoint ---
        if samples_per_save > 0 and checkpoint_path:
            newly_completed = int(completed_mask.sum()) - len(resumed_rows)
            if newly_completed - last_save_count >= samples_per_save:
                n_saved = write_checkpoint_csv(
                    df, max_scores, labels, completed_mask,
                    cfg.data_dir, cfg.crop, checkpoint_path,
                )
                logger.info(
                    f"  [Checkpoint] Saved {n_saved} rows to {checkpoint_path}"
                )
                last_save_count = newly_completed

    if total_pairs > 0:
        logger.info(f"Total inference time: {total_inference_time:.2f}s")
        logger.info(f"Average per pair:     {total_inference_time / total_pairs * 1000:.1f}ms")
    return max_scores, pair_scores
