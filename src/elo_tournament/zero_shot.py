"""
Step 1: Zero-shot scoring for query and reference images.

Computes initial ELO ratings by running each image through the VLM
independently.  Supports ``text`` mode (parse ``SCORE: <float>`` from
generated output) and ``logits`` mode (P(Yes) from next-token logits).

Both query and reference CSVs support incremental resume: if a CSV
exists with partial results, only the missing images are scored.
"""

import csv
import logging
import os
import re

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from src.common.data import prepare_image, write_checkpoint_csv
from src.elo_tournament.config import EloTournamentConfig
from src.vlm.inference import (
    _apply_chat_template,
    _get_yes_no_token_ids,
    ensure_pad_token_for_generation,
)

logger = logging.getLogger("vlm")


# ---------------------------------------------------------------------------
# Zero-shot message building
# ---------------------------------------------------------------------------

def _build_zero_shot_messages(
    image: Image.Image,
    system_prompt: str,
    user_prompt: str,
) -> list[dict]:
    """Build chat messages for zero-shot single-image scoring."""
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]


# ---------------------------------------------------------------------------
# Zero-shot scoring functions
# ---------------------------------------------------------------------------

def _zero_shot_score_text(
    model,
    processor,
    image: Image.Image,
    system_prompt: str,
    user_prompt: str,
    cfg: EloTournamentConfig,
) -> tuple[float, str]:
    """
    Generate text and parse ``SCORE: <float>`` from the reply.

    Falls back to the first numeric value found; defaults to 0.5.
    Returns ``(score, raw_reply)``.
    """
    ensure_pad_token_for_generation(processor, model)
    messages = _build_zero_shot_messages(image, system_prompt, user_prompt)
    inputs = _apply_chat_template(processor, messages, cfg.enable_thinking)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    gen_kwargs: dict = {"max_new_tokens": cfg.max_new_tokens}
    if cfg.temperature > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = cfg.temperature
    else:
        gen_kwargs["do_sample"] = False

    tok = processor.tokenizer
    pad_id = getattr(tok, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tok, "eos_token_id", None)
    if pad_id is not None:
        gen_kwargs["pad_token_id"] = pad_id

    with torch.no_grad():
        generated_ids = model.generate(**inputs, **gen_kwargs)

    new_tokens = generated_ids[0][input_len:]
    reply = processor.tokenizer.decode(
        new_tokens, skip_special_tokens=True,
    ).strip()

    del inputs, generated_ids

    match = re.search(
        r"SCORE:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))", reply, re.IGNORECASE,
    )
    if match:
        return min(max(float(match.group(1)), 0.0), 1.0), reply

    numeric = re.search(r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))", reply)
    if numeric:
        return min(max(float(numeric.group(1)), 0.0), 1.0), reply

    return 0.5, reply


def _zero_shot_score_logits(
    model,
    processor,
    image: Image.Image,
    system_prompt: str,
    user_prompt: str,
    cfg: EloTournamentConfig,
    yes_ids: list[int],
    no_ids: list[int],
) -> float:
    """Compute P(Yes) from next-token logits (no generation)."""
    messages = _build_zero_shot_messages(image, system_prompt, user_prompt)
    inputs = _apply_chat_template(processor, messages, cfg.enable_thinking)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    next_logits = outputs.logits[:, -1, :]
    yes_logit = next_logits[:, yes_ids].max(dim=-1).values
    no_logit = next_logits[:, no_ids].max(dim=-1).values
    probs = torch.softmax(
        torch.stack([yes_logit, no_logit], dim=-1), dim=-1,
    )
    score = probs[0, 0].item()

    del inputs, outputs
    return score


# ---------------------------------------------------------------------------
# Reference CSV I/O
# ---------------------------------------------------------------------------

def _load_reference_csv(csv_path: str) -> dict[str, dict]:
    """Load reference zero-shot scores keyed by relative image path."""
    result: dict[str, dict] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        has_output = "model_output" in (reader.fieldnames or [])
        for row in reader:
            entry: dict = {"score": float(row["predicted_score"])}
            if has_output:
                entry["model_output"] = row.get("model_output", "")
            result[row["image_path"]] = entry
    return result


def _save_reference_csv(
    scores: dict[str, float],
    out_path: str,
    model_outputs: dict[str, str] | None = None,
) -> None:
    """Write reference zero-shot scores to CSV."""
    has_output = model_outputs is not None
    header = ["image_path", "predicted_score"]
    if has_output:
        header.append("model_output")

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for path in sorted(scores):
            row: list = [path, f"{scores[path]:.6f}"]
            if has_output:
                row.append(model_outputs.get(path, ""))
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Query CSV I/O
# ---------------------------------------------------------------------------

def _load_query_csv(csv_path: str) -> dict[int, dict]:
    """Load query zero-shot scores keyed by ``original_index``."""
    result: dict[int, dict] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        has_output = "model_output" in (reader.fieldnames or [])
        for row in reader:
            orig_idx = int(row["original_index"])
            entry: dict = {"score": float(row["predicted_score"])}
            if has_output:
                entry["model_output"] = row.get("model_output", "")
            result[orig_idx] = entry
    return result


# ---------------------------------------------------------------------------
# Compute zero-shot scores for all query images
# ---------------------------------------------------------------------------

def compute_zero_shot_queries(
    df,
    model,
    processor,
    cfg: EloTournamentConfig,
    exp_dir: str,
) -> tuple[np.ndarray, list[str] | None]:
    """
    Score every query image with zero-shot VLM inference.

    Resume priority:
      1. ``cfg.zero_shot_query_csv`` (explicit path from config).
      2. ``<exp_dir>/zero_shot_queries.csv`` (auto-saved from previous run).
      3. Compute from scratch.

    Partial CSVs are continued — only missing rows are scored.
    Results are always saved to ``<exp_dir>/zero_shot_queries.csv``.

    Returns ``(scores_array, model_outputs_list_or_None)``.
    """
    col = "query_crop" if cfg.crop else "query_image"
    num_images = len(df)
    scores = np.full(num_images, 0.5, dtype=np.float64)
    seen = np.zeros(num_images, dtype=bool)
    is_text = cfg.zero_shot_scoring_mode == "text"
    outputs: list[str] | None = [""] * num_images if is_text else None

    has_orig = "original_index" in df.columns
    orig_to_row: dict[int, int] = {}
    for i in range(num_images):
        orig_idx = int(df.iloc[i]["original_index"]) if has_orig else i
        orig_to_row[orig_idx] = i

    # Resolve CSV path for loading
    load_path = cfg.zero_shot_query_csv
    auto_path = os.path.join(exp_dir, "zero_shot_queries.csv")
    if load_path is None and os.path.isfile(auto_path):
        load_path = auto_path

    if load_path and os.path.isfile(load_path):
        loaded = _load_query_csv(load_path)
        for orig_idx, data in loaded.items():
            if orig_idx in orig_to_row:
                row = orig_to_row[orig_idx]
                scores[row] = data["score"]
                seen[row] = True
                if outputs is not None and "model_output" in data:
                    outputs[row] = data["model_output"]
        logger.info(
            "Loaded %d/%d query zero-shot scores from %s",
            int(seen.sum()), num_images, load_path,
        )

    remaining = int((~seen).sum())
    if remaining == 0:
        logger.info("All query zero-shot scores already available.")
        return scores, outputs

    logger.info(
        "Computing %d remaining query zero-shot scores (mode=%s)...",
        remaining, cfg.zero_shot_scoring_mode,
    )

    sys_prompt = cfg.zero_shot_system_prompt()
    usr_prompt = cfg.zero_shot_user_prompt()

    yes_ids: list[int] = []
    no_ids: list[int] = []
    if cfg.zero_shot_scoring_mode == "logits":
        yes_ids, no_ids = _get_yes_no_token_ids(processor.tokenizer)

    labels = df["defect"].astype(int).values if "defect" in df.columns else None
    save_path = auto_path
    computed = 0

    for idx in tqdm(range(num_images), desc="ZS queries"):
        if seen[idx]:
            continue

        image_path = df.iloc[idx][col]
        image = prepare_image(
            image_path, cfg.data_dir, cfg.mask, cfg.input_size,
        )

        if is_text:
            score, reply = _zero_shot_score_text(
                model, processor, image, sys_prompt, usr_prompt, cfg,
            )
            outputs[idx] = reply
        else:
            score = _zero_shot_score_logits(
                model, processor, image, sys_prompt, usr_prompt, cfg,
                yes_ids, no_ids,
            )

        scores[idx] = score
        seen[idx] = True
        computed += 1

        torch.cuda.empty_cache()

        if cfg.samples_per_save > 0 and computed % cfg.samples_per_save == 0:
            write_checkpoint_csv(
                df, scores, labels, seen,
                cfg.data_dir, cfg.crop, save_path,
                model_outputs=outputs,
            )
            logger.info(
                "  [ZS Query Checkpoint] %d/%d saved",
                int(seen.sum()), num_images,
            )

    write_checkpoint_csv(
        df, scores, labels, seen,
        cfg.data_dir, cfg.crop, save_path,
        model_outputs=outputs,
    )
    logger.info("Query zero-shot scores saved to %s", save_path)

    return scores, outputs


# ---------------------------------------------------------------------------
# Compute zero-shot scores for all unique reference images
# ---------------------------------------------------------------------------

def compute_zero_shot_references(
    ref_lookup: dict[str, list[str]],
    model,
    processor,
    cfg: EloTournamentConfig,
    exp_dir: str,
) -> tuple[dict[str, float], dict[str, str] | None]:
    """
    Score every unique reference image with zero-shot VLM inference.

    Same resume logic as :func:`compute_zero_shot_queries` but keyed
    by relative image path instead of ``original_index``.

    Returns ``(path_to_score, path_to_model_output_or_None)``.
    """
    all_ref_paths: set[str] = set()
    for paths in ref_lookup.values():
        all_ref_paths.update(paths)
    all_ref_paths_sorted = sorted(all_ref_paths)

    ref_scores: dict[str, float] = {}
    is_text = cfg.zero_shot_scoring_mode == "text"
    ref_outputs: dict[str, str] | None = {} if is_text else None

    load_path = cfg.zero_shot_reference_csv
    auto_path = os.path.join(exp_dir, "zero_shot_references.csv")
    if load_path is None and os.path.isfile(auto_path):
        load_path = auto_path

    if load_path and os.path.isfile(load_path):
        loaded = _load_reference_csv(load_path)
        for path, data in loaded.items():
            if path in all_ref_paths:
                ref_scores[path] = data["score"]
                if ref_outputs is not None and "model_output" in data:
                    ref_outputs[path] = data["model_output"]
        logger.info(
            "Loaded %d/%d reference zero-shot scores from %s",
            len(ref_scores), len(all_ref_paths_sorted), load_path,
        )

    remaining = [p for p in all_ref_paths_sorted if p not in ref_scores]
    if not remaining:
        logger.info("All reference zero-shot scores already available.")
        return ref_scores, ref_outputs

    logger.info(
        "Computing %d remaining reference zero-shot scores (mode=%s)...",
        len(remaining), cfg.zero_shot_scoring_mode,
    )

    sys_prompt = cfg.zero_shot_system_prompt()
    usr_prompt = cfg.zero_shot_user_prompt()

    yes_ids: list[int] = []
    no_ids: list[int] = []
    if cfg.zero_shot_scoring_mode == "logits":
        yes_ids, no_ids = _get_yes_no_token_ids(processor.tokenizer)

    save_path = auto_path
    computed = 0

    for ref_path in tqdm(remaining, desc="ZS references"):
        image = prepare_image(
            ref_path, cfg.data_dir, cfg.mask, cfg.input_size,
        )

        if is_text:
            score, reply = _zero_shot_score_text(
                model, processor, image, sys_prompt, usr_prompt, cfg,
            )
            ref_outputs[ref_path] = reply
        else:
            score = _zero_shot_score_logits(
                model, processor, image, sys_prompt, usr_prompt, cfg,
                yes_ids, no_ids,
            )

        ref_scores[ref_path] = score
        computed += 1

        torch.cuda.empty_cache()

        if cfg.samples_per_save > 0 and computed % cfg.samples_per_save == 0:
            _save_reference_csv(ref_scores, save_path, ref_outputs)
            logger.info(
                "  [ZS Ref Checkpoint] %d/%d saved",
                len(ref_scores), len(all_ref_paths_sorted),
            )

    _save_reference_csv(ref_scores, save_path, ref_outputs)
    logger.info("Reference zero-shot scores saved to %s", save_path)

    return ref_scores, ref_outputs
