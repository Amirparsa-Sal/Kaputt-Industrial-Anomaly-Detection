"""
Batched SAM3 model inference.
"""

import logging
import time

import numpy as np
import torch
from tqdm import tqdm
from transformers import Sam3Model, Sam3Processor

from src.sam3.config import Config
from src.common.data import PairKey, build_all_prompt_pairs, prepare_image

logger = logging.getLogger("sam3")


def run_inference(
    df,
    model: Sam3Model,
    processor: Sam3Processor,
    cfg: Config,
) -> tuple[np.ndarray, dict[PairKey, dict[str, list]]]:
    """
    Run batched SAM3 inference using ALL prompts on ALL images.
    Uses the lowest configured threshold for maximum detection sensitivity.

    Returns
    -------
    max_scores : np.ndarray
        One aggregated (max-over-all-prompts) prediction score per image.
    pair_scores : dict[(prompt_defect_type, prompt) -> {"indices", "scores", "detections"}]
        Raw per-image scores broken down by each (prompt_defect_type, prompt) pair.
    """
    expanded = build_all_prompt_pairs(df, cfg.prompts, crop=cfg.crop)
    num_images = len(df)
    total_pairs = len(expanded)
    prompts_per_image = total_pairs / num_images if num_images else 0

    inference_threshold = min(cfg.thresholds)

    logger.info(
        f"Expanded to {total_pairs} (image, prompt) pairs "
        f"({prompts_per_image:.1f} prompts/image)"
    )
    logger.info(f"Using inference threshold: {inference_threshold}")

    device = f"cuda:{cfg.gpu_id}"
    max_scores = np.zeros(num_images, dtype=np.float64)
    pair_scores: dict[PairKey, dict[str, list]] = {}
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

    logger.info(f"Total inference time: {total_inference_time:.2f}s")
    logger.info(f"Average per pair:     {total_inference_time / total_pairs * 1000:.1f}ms")
    return max_scores, pair_scores
