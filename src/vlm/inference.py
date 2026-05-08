"""
VLM-based anomaly classification inference.

Supports:
  - ``shot_mode`` ``zero_shot`` (single query image) or ``few_shot`` (2×2 grid:
    query top-left, three reference images from ``reference-{split}.parquet``).
  - ``scoring_mode`` ``text`` (generated score) or ``logits`` (P(Yes) from
    yes/no token logits).
"""

import logging
import re
import time

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import average_precision_score
from tqdm import tqdm

from src.common.data import (
    PairKey,
    build_reference_path_lookup,
    compose_vlm_few_shot_grid,
    prepare_image,
    write_checkpoint_csv,
)
from src.vlm.config import VLMConfig

logger = logging.getLogger("vlm")


def ensure_pad_token_for_generation(processor, model) -> None:
    """
    Align pad token/ids on the tokenizer and model so ``generate()`` does not
    emit the common fallback log about setting ``pad_token_id`` to ``eos_token_id``.

    Some VL stacks only set a string ``pad_token``; we also set the integer ids
    and copy them onto ``model.config`` / ``model.generation_config`` when
    present.
    """
    tok = getattr(processor, "tokenizer", None)
    if tok is None:
        return
    if getattr(tok, "pad_token", None) is None and getattr(
        tok, "eos_token", None,
    ):
        tok.pad_token = tok.eos_token
    if getattr(tok, "pad_token_id", None) is None and getattr(
        tok, "eos_token_id", None,
    ) is not None:
        tok.pad_token_id = tok.eos_token_id
    pad_id = getattr(tok, "pad_token_id", None)
    if pad_id is None:
        return
    mcfg = getattr(model, "config", None)
    if mcfg is not None and getattr(mcfg, "pad_token_id", None) is None:
        mcfg.pad_token_id = pad_id
    gcfg = getattr(model, "generation_config", None)
    if gcfg is not None and getattr(gcfg, "pad_token_id", None) is None:
        gcfg.pad_token_id = pad_id


def _get_yes_no_token_ids(
    tokenizer,
) -> tuple[list[int], list[int]]:
    """
    Collect token IDs for common Yes / No surface forms.
    Returns ``(yes_ids, no_ids)`` — deduplicated lists.
    """
    yes_ids: set[int] = set()
    for variant in ("Yes", "yes", "YES"):
        ids = tokenizer.encode(variant, add_special_tokens=False)
        if ids:
            yes_ids.add(ids[0])

    no_ids: set[int] = set()
    for variant in ("No", "no", "NO"):
        ids = tokenizer.encode(variant, add_special_tokens=False)
        if ids:
            no_ids.add(ids[0])

    return list(yes_ids), list(no_ids)


def _build_messages(
    image: Image.Image,
    prompt: str,
    cfg: VLMConfig,
) -> list[dict]:
    """Build Qwen-style chat messages with an image attachment."""
    system = cfg.system_message_for_run()

    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        },
    ]


def _apply_chat_template(processor, messages, enable_thinking: bool):
    """
    Tokenize chat messages via the processor's template.

    Tries to pass ``enable_thinking`` when supported, falling back
    gracefully on older transformers / model versions.
    """
    kwargs = dict(
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    if not enable_thinking:
        kwargs["enable_thinking"] = False

    try:
        return processor.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return processor.apply_chat_template(messages, **kwargs)


def _score_from_logits(
    model,
    processor,
    image: Image.Image,
    prompt: str,
    cfg: VLMConfig,
    yes_ids: list[int],
    no_ids: list[int],
    enable_thinking: bool = False,
) -> float:
    """
    Compute anomaly probability from next-token logit scores.

    Performs one forward pass (no generation) and returns
    ``softmax(max_yes_logit, max_no_logit)[0]``  →  P(Yes).
    """
    messages = _build_messages(image, prompt, cfg)
    inputs = _apply_chat_template(processor, messages, enable_thinking)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    next_logits = outputs.logits[:, -1, :]  # (1, vocab)

    yes_logit = next_logits[:, yes_ids].max(dim=-1).values
    no_logit = next_logits[:, no_ids].max(dim=-1).values
    probs = torch.softmax(
        torch.stack([yes_logit, no_logit], dim=-1), dim=-1,
    )

    score = probs[0, 0].item()

    del inputs, outputs
    return score


def _score_from_text(
    model,
    processor,
    image: Image.Image,
    prompt: str,
    cfg: VLMConfig,
    temperature: float = 0.0,
    max_new_tokens: int = 256,
    enable_thinking: bool = False,
) -> tuple[float, str]:
    """
    Generate text and parse a numeric confidence score in ``[0, 1]``.

    Preferred format is ``SCORE: <float>``. If absent, falls back to the first
    numeric token found in the reply; if none exists, returns ``0.5``.

    Returns ``(score, raw_reply)``.
    """
    ensure_pad_token_for_generation(processor, model)

    messages = _build_messages(image, prompt, cfg)
    inputs = _apply_chat_template(processor, messages, enable_thinking)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    gen_kwargs: dict = {"max_new_tokens": max_new_tokens}
    if temperature > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
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

    match = re.search(r"SCORE:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))", reply, re.IGNORECASE)
    if match:
        score = float(match.group(1))
        return min(max(score, 0.0), 1.0), reply

    # If the model does not respect the explicit "SCORE:" prefix, still accept
    # a plain numeric reply such as "0.73".
    numeric_match = re.search(r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))", reply)
    if numeric_match:
        score = float(numeric_match.group(1))
        return min(max(score, 0.0), 1.0), reply

    return 0.5, reply


def _prepare_model_input_image(
    image_path: str,
    row_item_id: str,
    cfg: VLMConfig,
    ref_lookup: dict[str, list[str]] | None,
) -> Image.Image:
    """
    Return the PIL image fed to the VLM: either a resized query crop/image
    (zero-shot) or a 2×2 grid (few-shot) at ``cfg.input_size`` square.
    """
    if cfg.shot_mode == "zero_shot":
        return prepare_image(
            image_path, cfg.data_dir, cfg.mask, cfg.input_size,
        )

    assert ref_lookup is not None
    assert cfg.input_size is not None

    query_pil = prepare_image(
        image_path, cfg.data_dir, cfg.mask, input_size=None,
    )
    ref_paths = ref_lookup[row_item_id]
    ref_pils: list[Image.Image] = []
    for rp in ref_paths:
        ref_pils.append(
            prepare_image(rp, cfg.data_dir, cfg.mask, input_size=None),
        )
    return compose_vlm_few_shot_grid(
        query_pil, ref_pils, int(cfg.input_size),
    )


def run_vlm_inference(
    df,
    model,
    processor,
    cfg: VLMConfig,
    *,
    checkpoint_path: str | None = None,
    samples_per_save: int = 0,
    resume_data: dict[int, dict] | None = None,
) -> tuple[np.ndarray, dict[PairKey, dict[str, list]], list[str] | None]:
    """
    Run VLM anomaly classification on every row of *df*.

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
        ``original_index`` values; each value has at least ``"score"``
        and optionally ``"model_output"``.

    Returns
    -------
    max_scores : np.ndarray
        One anomaly score per image (continuous 0-1 probability).
    pair_scores : dict[PairKey, {"indices", "scores", "detections"}]
        Scores keyed by ``(shot_mode, scoring_mode)`` for downstream metrics.
    text_outputs : list[str] | None
        Decoded model replies when ``scoring_mode`` is ``"text"`` (aligned with
        *df* rows); ``None`` for ``"logits"`` mode.
    """
    col = "query_crop" if cfg.crop else "query_image"
    prompt_text = cfg.user_message_for_run()
    pair_label: PairKey = (cfg.shot_mode, cfg.scoring_mode)

    ref_lookup: dict[str, list[str]] | None = None
    if cfg.shot_mode == "few_shot":
        if "item_identifier" not in df.columns:
            raise ValueError(
                "Few-shot mode requires an 'item_identifier' column in the "
                "query dataframe (from the query parquet).",
            )
        ref_lookup = build_reference_path_lookup(
            cfg.data_dir,
            cfg.split,
            cfg.crop,
            pad_short=cfg.pad_short_references,
        )

    assignments: list[tuple[int, str, str]] = []
    for idx in range(len(df)):
        image_path = df.iloc[idx][col]
        item_id = ""
        if cfg.shot_mode == "few_shot":
            item_id = str(df.iloc[idx]["item_identifier"])
        assignments.append((idx, image_path, item_id))

    num_images = len(df)

    max_scores = np.zeros(num_images, dtype=np.float64)
    seen_mask = np.zeros(num_images, dtype=bool)
    pair_scores: dict[PairKey, dict[str, list]] = {}
    text_outputs: list[str] | None = (
        [""] * num_images if cfg.scoring_mode == "text" else None
    )

    # --- Resume: pre-fill scores and skip already-processed images ---
    has_orig = "original_index" in df.columns
    orig_to_row: dict[int, int] = {}
    for i in range(num_images):
        orig_idx = int(df.iloc[i]["original_index"]) if has_orig else i
        orig_to_row[orig_idx] = i

    resumed_rows: set[int] = set()
    if resume_data:
        for orig_idx, data in resume_data.items():
            if orig_idx in orig_to_row:
                row_idx = orig_to_row[orig_idx]
                max_scores[row_idx] = data["score"]
                seen_mask[row_idx] = True
                if text_outputs is not None and "model_output" in data:
                    text_outputs[row_idx] = data["model_output"]
                if pair_label not in pair_scores:
                    pair_scores[pair_label] = {
                        "indices": [], "scores": [], "detections": [],
                    }
                pair_scores[pair_label]["indices"].append(row_idx)
                pair_scores[pair_label]["scores"].append(data["score"])
                pair_scores[pair_label]["detections"].append(
                    {"boxes": [], "scores": []},
                )
                resumed_rows.add(row_idx)
        logger.info(
            f"Resumed {len(resumed_rows)} / {num_images} images from checkpoint."
        )

    logger.info(
        f"shot_mode={cfg.shot_mode}  scoring_mode={cfg.scoring_mode}  "
        f"images={num_images}  remaining={num_images - len(resumed_rows)}",
    )

    yes_ids, no_ids = [], []
    if cfg.scoring_mode == "logits":
        yes_ids, no_ids = _get_yes_no_token_ids(processor.tokenizer)
        logger.info(f"Yes token IDs: {yes_ids}, No token IDs: {no_ids}")

    total_time = 0.0
    labels = df["defect"].astype(int).values
    report_interval_s = cfg.report_interval_minutes * 60
    last_report_time = time.time()
    processed = 0
    last_save_count = 0

    for idx, image_path, item_id in tqdm(
        assignments,
        desc=f"VLM ({cfg.shot_mode}/{cfg.scoring_mode})",
    ):
        if idx in resumed_rows:
            continue

        image = _prepare_model_input_image(
            image_path, item_id, cfg, ref_lookup,
        )

        t0 = time.time()

        if cfg.scoring_mode == "logits":
            score = _score_from_logits(
                model, processor, image, prompt_text, cfg,
                yes_ids, no_ids,
                enable_thinking=cfg.enable_thinking,
            )
        else:
            score, raw_reply = _score_from_text(
                model, processor, image, prompt_text, cfg,
                temperature=cfg.temperature,
                max_new_tokens=cfg.max_new_tokens,
                enable_thinking=cfg.enable_thinking,
            )
            assert text_outputs is not None
            text_outputs[idx] = raw_reply

        total_time += time.time() - t0
        max_scores[idx] = score
        seen_mask[idx] = True
        processed += 1

        if pair_label not in pair_scores:
            pair_scores[pair_label] = {
                "indices": [], "scores": [], "detections": [],
            }
        pair_scores[pair_label]["indices"].append(idx)
        pair_scores[pair_label]["scores"].append(score)
        pair_scores[pair_label]["detections"].append(
            {"boxes": [], "scores": []},
        )

        torch.cuda.empty_cache()

        # --- Periodic checkpoint ---
        if samples_per_save > 0 and checkpoint_path:
            if processed - last_save_count >= samples_per_save:
                n_saved = write_checkpoint_csv(
                    df, max_scores, labels, seen_mask,
                    cfg.data_dir, cfg.crop, checkpoint_path,
                    model_outputs=text_outputs,
                )
                logger.info(
                    f"  [Checkpoint] Saved {n_saved} rows to {checkpoint_path}"
                )
                last_save_count = processed

        # --- Periodic interim report ---
        now = time.time()
        if now - last_report_time >= report_interval_s:
            scored_mask = seen_mask[:num_images]
            scored_labels = labels[scored_mask]
            scored_scores = max_scores[scored_mask]
            n_pos = int(scored_labels.sum())
            n_neg = int(len(scored_labels) - n_pos)
            n_scored = n_pos + n_neg
            pos_pct = n_pos / n_scored * 100 if n_scored > 0 else 0.0
            random_ap = n_pos / n_scored if n_scored > 0 else float("nan")
            if n_pos > 0 and n_neg > 0:
                interim_ap = average_precision_score(
                    scored_labels, scored_scores,
                )
                logger.info(
                    f"\n  [Interim report — {processed}/{num_images} images, "
                    f"{total_time / 60:.1f} min elapsed]\n"
                    f"    AP = {interim_ap:.4f}  "
                    f"(random baseline AP = {random_ap:.4f})\n"
                    f"    pos={n_pos}, neg={n_neg}  "
                    f"({pos_pct:.1f}% positive)\n"
                )
            else:
                logger.info(
                    f"\n  [Interim report — {processed}/{num_images} images, "
                    f"{total_time / 60:.1f} min elapsed]\n"
                    f"    AP = N/A (only one class seen so far)\n"
                    f"    pos={n_pos}, neg={n_neg}  "
                    f"({pos_pct:.1f}% positive)\n"
                )
            last_report_time = now

    logger.info(f"Total inference time: {total_time:.2f}s")
    if processed > 0:
        logger.info(
            f"Average per image:    {total_time / processed * 1000:.1f}ms"
        )

    return max_scores, pair_scores, text_outputs
