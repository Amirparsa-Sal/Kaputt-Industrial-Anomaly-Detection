"""
ELO tournament inference — Steps 2 and 3.

Step 2: Pairwise matches among reference images only (cached per
        ``item_identifier`` to avoid redundant VLM calls).
Step 3: Pairwise matches between the query and each reference image.

Both steps update ELO ratings with the formula:
  E_A = 1 / (1 + 10^((R_B - R_A) / d))
  R_A = clip(R_A + k * (s - E_A), 0, 1)
  R_B = clip(R_B - k * (s - E_A), 0, 1)

Match result *s* comes from:
  - ``confidence`` mode: parsed ``CONFIDENCE: <float>`` from generated text.
  - ``wdl`` mode: ``RESULT: WIN`` → 1, ``DRAW`` → 0.5, ``LOSS`` → 0.
"""

import hashlib
import itertools
import logging
import os
import re
import time

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from src.common.data import PairKey, prepare_image, write_checkpoint_csv
from src.elo_tournament.config import EloTournamentConfig
from src.elo_tournament.zero_shot import (
    compute_zero_shot_queries,
    compute_zero_shot_references,
)
from src.tournament.inference import (
    build_tournament_reference_lookup,
    compose_tournament_grid,
    _grid_image_description_league,
    _parse_confidence,
    _separate_image_description_league,
)
from src.vlm.inference import (
    _apply_chat_template,
    ensure_pad_token_for_generation,
)

logger = logging.getLogger("vlm")


# ---------------------------------------------------------------------------
# ELO update
# ---------------------------------------------------------------------------

def elo_update(
    r_a: float, r_b: float, s: float, d: float, k: float,
) -> tuple[float, float]:
    """
    Update ELO ratings for a single match result.

    Parameters
    ----------
    r_a, r_b : current ratings of participants A and B.
    s : match outcome from A's perspective (1 = A wins, 0 = A loses).
    d : controls sensitivity of expected-probability curve.
    k : learning rate.

    Returns (new_r_a, new_r_b), both clipped to [0, 1].
    """
    e_a = 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / d))
    delta = k * (s - e_a)
    new_r_a = float(np.clip(r_a + delta, 0.0, 1.0))
    new_r_b = float(np.clip(r_b - delta, 0.0, 1.0))
    return new_r_a, new_r_b


# ---------------------------------------------------------------------------
# Match message building
# ---------------------------------------------------------------------------

def _build_match_messages(
    image_a: Image.Image,
    image_b: Image.Image,
    cfg: EloTournamentConfig,
) -> list[dict]:
    """Build chat messages for a pairwise comparison match."""
    if cfg.use_grid:
        image_desc = _grid_image_description_league()
    else:
        image_desc = _separate_image_description_league()

    user_text = cfg.match_user_prompt_template().format(
        image_description=image_desc,
    )

    if cfg.use_grid:
        grid = compose_tournament_grid([image_a, image_b], cfg.input_size)
        content: list[dict] = [
            {"type": "image", "image": grid},
            {"type": "text", "text": user_text},
        ]
    else:
        content = [
            {"type": "image", "image": image_a},
            {"type": "image", "image": image_b},
            {"type": "text", "text": user_text},
        ]

    return [
        {"role": "system", "content": cfg.match_system_prompt()},
        {"role": "user", "content": content},
    ]


# ---------------------------------------------------------------------------
# Text generation
# ---------------------------------------------------------------------------

def _generate_text(
    model, processor, messages: list[dict], cfg: EloTournamentConfig,
) -> str:
    """Run a single text-generation forward pass and return the reply."""
    ensure_pad_token_for_generation(processor, model)

    inputs = _apply_chat_template(
        processor, messages, cfg.enable_thinking,
    )
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
    return reply


# ---------------------------------------------------------------------------
# WDL parser
# ---------------------------------------------------------------------------

def _parse_wdl(text: str) -> float:
    """
    Parse ``RESULT: WIN/DRAW/LOSS`` from model output.

    Returns *s*: 1.0 for WIN, 0.5 for DRAW, 0.0 for LOSS.
    Falls back to keyword search, then 0.5 (draw).
    """
    match = re.search(r"RESULT:\s*(WIN|DRAW|LOSS)", text, re.IGNORECASE)
    if match:
        token = match.group(1).upper()
        return {"WIN": 1.0, "LOSS": 0.0}.get(token, 0.5)

    upper = text.upper()
    if "WIN" in upper:
        return 1.0
    if "LOSS" in upper or "LOSE" in upper:
        return 0.0
    return 0.5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_s(s: float, match_mode: str) -> str:
    """Human-readable match result for compact logging."""
    if match_mode == "wdl":
        tag = {1.0: "W", 0.0: "L"}.get(s, "D")
        return f"{s:.1f}({tag})"
    return f"{s:.3f}"


def _truncate(text: str, max_len: int = 60) -> str:
    """Collapse whitespace and truncate for log / CSV embedding."""
    one = " ".join(text.split())
    return one if len(one) <= max_len else one[: max_len - 3] + "..."


def _item_seed(item_id: str) -> int:
    """Deterministic seed derived from item_identifier (stable across runs)."""
    return int(hashlib.md5(item_id.encode()).hexdigest()[:8], 16)


# ---------------------------------------------------------------------------
# Play a single match
# ---------------------------------------------------------------------------

def _play_match(
    image_a: Image.Image,
    image_b: Image.Image,
    model,
    processor,
    cfg: EloTournamentConfig,
    rng: np.random.Generator,
) -> tuple[float, str]:
    """
    Play one pairwise match with randomised A/B presentation.

    Returns ``(s, raw_reply)`` where *s* is the confidence that the
    **original** image_a is more anomalous than image_b (position-bias
    corrected when the presentation order is swapped).
    """
    swapped = rng.random() < 0.5
    if swapped:
        show_a, show_b = image_b, image_a
    else:
        show_a, show_b = image_a, image_b

    messages = _build_match_messages(show_a, show_b, cfg)
    reply = _generate_text(model, processor, messages, cfg)

    if cfg.match_mode == "wdl":
        s = _parse_wdl(reply)
    else:
        s = _parse_confidence(reply)

    # Correct for the swap: s refers to shown_A vs shown_B.
    if swapped:
        s = 1.0 - s

    torch.cuda.empty_cache()
    return s, reply


# ---------------------------------------------------------------------------
# Step 2: Reference-only tournament
# ---------------------------------------------------------------------------

def _run_reference_tournament(
    ref_images: list[Image.Image],
    ref_elos: list[float],
    model,
    processor,
    cfg: EloTournamentConfig,
    rng: np.random.Generator,
) -> tuple[dict[float, list[float]], dict[float, str]]:
    """
    Round-robin pairwise matches among reference images.

    ELO ratings are tracked in parallel for every ``cfg.elo_k`` value.
    A single VLM call per pair produces the match result *s*; the ELO
    update is then applied independently for each *k*.

    Returns ``(k → updated_elos, k → compact_match_log)``.
    """
    n = len(ref_images)
    k_values = cfg.elo_k

    all_elos: dict[float, list[float]] = {
        k: list(ref_elos) for k in k_values
    }
    all_parts: dict[float, list[str]] = {k: [] for k in k_values}

    pairs = list(itertools.combinations(range(n), 2))
    rng.shuffle(pairs)

    for i, j in pairs:
        s, reply = _play_match(
            ref_images[i], ref_images[j], model, processor, cfg, rng,
        )
        s_str = _format_s(s, cfg.match_mode)
        logger.info(
            "  Ref match R%d vs R%d: s=%s  [%s]",
            i + 1, j + 1, s_str, _truncate(reply),
        )

        for k in k_values:
            all_elos[k][i], all_elos[k][j] = elo_update(
                all_elos[k][i], all_elos[k][j], s, cfg.elo_d, k,
            )
            all_parts[k].append(
                f"R{i+1}vR{j+1} s={s_str} "
                f"R{i+1}\u2192{all_elos[k][i]:.3f} "
                f"R{j+1}\u2192{all_elos[k][j]:.3f}"
            )

    logs = {k: ";".join(all_parts[k]) for k in k_values}
    return all_elos, logs


# ---------------------------------------------------------------------------
# Step 3: Query vs. references
# ---------------------------------------------------------------------------

def _run_query_vs_refs(
    query_image: Image.Image,
    ref_images: list[Image.Image],
    query_elo: float,
    ref_elos_per_k: dict[float, list[float]],
    model,
    processor,
    cfg: EloTournamentConfig,
    rng: np.random.Generator,
) -> tuple[dict[float, float], dict[float, str]]:
    """
    One match per reference: query vs. each ref (shuffled order).

    ELO ratings are tracked in parallel for every ``cfg.elo_k`` value.
    ``ref_elos_per_k`` provides the post-Step-2 reference ELOs per *k*.

    Returns ``(k → final_query_elo, k → compact_match_log)``.
    """
    k_values = cfg.elo_k
    q_elos: dict[float, float] = {k: query_elo for k in k_values}
    r_elos: dict[float, list[float]] = {
        k: list(ref_elos_per_k[k]) for k in k_values
    }
    all_parts: dict[float, list[str]] = {k: [] for k in k_values}

    indices = list(range(len(ref_images)))
    rng.shuffle(indices)

    for j in indices:
        s, reply = _play_match(
            query_image, ref_images[j], model, processor, cfg, rng,
        )
        s_str = _format_s(s, cfg.match_mode)
        logger.info(
            "  Query vs R%d: s=%s  [%s]",
            j + 1, s_str, _truncate(reply),
        )

        for k in k_values:
            q_elos[k], r_elos[k][j] = elo_update(
                q_elos[k], r_elos[k][j], s, cfg.elo_d, k,
            )
            all_parts[k].append(
                f"QvR{j+1} s={s_str} Q\u2192{q_elos[k]:.3f}"
            )

    logs = {k: ";".join(all_parts[k]) for k in k_values}
    return q_elos, logs


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _k_csv_path(exp_dir: str, k_val: float) -> str:
    """Per-k prediction CSV path inside the experiment directory."""
    return os.path.join(exp_dir, f"predictions_k_{k_val}.csv")


def run_elo_tournament_inference(
    df,
    model,
    processor,
    cfg: EloTournamentConfig,
    *,
    exp_dir: str,
    samples_per_save: int = 0,
    resume_per_k: dict[float, dict[int, dict]] | None = None,
) -> tuple[
    dict[float, np.ndarray],
    dict[float, dict[PairKey, dict[str, list]]],
    dict[float, list[str]],
]:
    """
    Run the full 3-step ELO tournament pipeline on every row of *df*.

    ELO updates are tracked independently for every value in
    ``cfg.elo_k``, reusing the same VLM match results.

    Returns ``(scores_per_k, pair_scores_per_k, text_per_k)`` where
    each dict is keyed by *k* value.
    """
    col = "query_crop" if cfg.crop else "query_image"
    pair_label: PairKey = ("elo_tournament", cfg.match_mode)
    k_values = cfg.elo_k

    if "item_identifier" not in df.columns:
        raise ValueError(
            "ELO tournament requires an 'item_identifier' column."
        )

    # ---- Step 0: reference lookup ----
    ref_lookup = build_tournament_reference_lookup(
        cfg.data_dir, cfg.split, cfg.crop, cfg.num_references,
    )

    # ---- Step 1: zero-shot scoring ----
    logger.info("\n" + "=" * 50)
    logger.info("Step 1 — Zero-shot scoring")
    logger.info("=" * 50)

    query_zs, _ = compute_zero_shot_queries(
        df, model, processor, cfg, exp_dir,
    )
    ref_zs, _ = compute_zero_shot_references(
        ref_lookup, model, processor, cfg, exp_dir,
    )

    logger.info(
        "Zero-shot complete: %d queries, %d unique references.",
        len(query_zs), len(ref_zs),
    )

    # ---- Prepare per-k output arrays ----
    num_images = len(df)
    seen_mask = np.zeros(num_images, dtype=bool)

    scores_per_k: dict[float, np.ndarray] = {
        k: np.zeros(num_images, dtype=np.float64) for k in k_values
    }
    pair_scores_per_k: dict[float, dict[PairKey, dict[str, list]]] = {
        k: {} for k in k_values
    }
    text_per_k: dict[float, list[str]] = {
        k: [""] * num_images for k in k_values
    }

    # ---- Resume from per-k checkpoints ----
    has_orig = "original_index" in df.columns
    orig_to_row: dict[int, int] = {}
    for i in range(num_images):
        orig_idx = int(df.iloc[i]["original_index"]) if has_orig else i
        orig_to_row[orig_idx] = i

    resumed_rows: set[int] = set()
    if resume_per_k:
        primary_k = k_values[0]
        primary_data = resume_per_k.get(primary_k, {})
        for orig_idx, data in primary_data.items():
            if orig_idx in orig_to_row:
                row_idx = orig_to_row[orig_idx]
                seen_mask[row_idx] = True
                resumed_rows.add(row_idx)

        for k_val in k_values:
            k_data = resume_per_k.get(k_val, {})
            for orig_idx, data in k_data.items():
                if orig_idx in orig_to_row:
                    row_idx = orig_to_row[orig_idx]
                    scores_per_k[k_val][row_idx] = data["score"]
                    if "model_output" in data:
                        text_per_k[k_val][row_idx] = data["model_output"]
                    if pair_label not in pair_scores_per_k[k_val]:
                        pair_scores_per_k[k_val][pair_label] = {
                            "indices": [], "scores": [], "detections": [],
                        }
                    pair_scores_per_k[k_val][pair_label]["indices"].append(
                        row_idx
                    )
                    pair_scores_per_k[k_val][pair_label]["scores"].append(
                        data["score"]
                    )
                    pair_scores_per_k[k_val][pair_label][
                        "detections"
                    ].append({"boxes": [], "scores": []})

        logger.info(
            "Resumed %d / %d images from per-k checkpoints.",
            len(resumed_rows), num_images,
        )

    # ---- Identify items that still need ref tournaments ----
    items_needing_tournament: set[str] = set()
    for idx in range(num_images):
        if idx not in resumed_rows:
            items_needing_tournament.add(
                str(df.iloc[idx]["item_identifier"])
            )

    # ---- Step 2: reference-only tournaments (cached, multi-k) ----
    logger.info("\n" + "=" * 50)
    logger.info("Step 2 — Reference-only tournaments")
    logger.info("=" * 50)
    logger.info("Tracking %d k-values: %s", len(k_values), k_values)

    # Cache stores k → updated_elos and k → match_log per item.
    ref_cache: dict[str, tuple[dict[float, list[float]], dict[float, str]]] = {}
    items_logged: set[str] = set()

    items_to_process = sorted(items_needing_tournament)
    for item_id in tqdm(items_to_process, desc="Ref tournaments"):
        ref_paths = ref_lookup.get(item_id, [])
        if len(ref_paths) < 2:
            init = [ref_zs.get(p, 0.5) for p in ref_paths]
            ref_cache[item_id] = (
                {k: list(init) for k in k_values},
                {k: "skip(n<2)" for k in k_values},
            )
            continue

        item_rng = np.random.default_rng(seed=_item_seed(item_id))

        ref_images = [
            prepare_image(rp, cfg.data_dir, cfg.mask, input_size=None)
            for rp in ref_paths
        ]
        init_elos = [ref_zs.get(p, 0.5) for p in ref_paths]

        logger.info(
            "\nRef tournament item=%s  init=%s",
            item_id,
            " ".join(f"R{i+1}={e:.3f}" for i, e in enumerate(init_elos)),
        )

        elos_per_k, logs_per_k = _run_reference_tournament(
            ref_images, init_elos, model, processor, cfg, item_rng,
        )
        ref_cache[item_id] = (elos_per_k, logs_per_k)

        for k in k_values:
            logger.info(
                "  Post-ref (k=%.4f): %s", k,
                " ".join(
                    f"R{i+1}={e:.3f}"
                    for i, e in enumerate(elos_per_k[k])
                ),
            )

    logger.info(
        "Reference tournaments done for %d item(s).", len(items_to_process),
    )

    # ---- Step 3: query vs. references (multi-k) ----
    logger.info("\n" + "=" * 50)
    logger.info("Step 3 — Query vs. reference tournaments")
    logger.info("=" * 50)

    total_time = 0.0
    has_labels = "defect" in df.columns
    labels = df["defect"].astype(int).values if has_labels else None
    processed = 0
    last_save_count = 0
    report_interval_s = cfg.report_interval_minutes * 60
    last_report_time = time.time()

    for idx in tqdm(range(num_images), desc="ELO query matches"):
        if idx in resumed_rows:
            continue

        item_id = str(df.iloc[idx]["item_identifier"])
        image_path = df.iloc[idx][col]
        query_elo = float(query_zs[idx])

        ref_paths = ref_lookup.get(item_id, [])
        if not ref_paths:
            for k in k_values:
                scores_per_k[k][idx] = query_elo
                text_per_k[k][idx] = (
                    f"ZS:Q={query_elo:.3f} | NoRefs | F={query_elo:.3f}"
                )
            seen_mask[idx] = True
            processed += 1
            continue

        cached_elos_per_k, ref_logs_per_k = ref_cache[item_id]
        ref_images = [
            prepare_image(rp, cfg.data_dir, cfg.mask, input_size=None)
            for rp in ref_paths
        ]

        orig_idx = int(df.iloc[idx]["original_index"]) if has_orig else idx
        query_rng = np.random.default_rng(seed=42 + orig_idx)

        t0 = time.time()

        logger.info(
            "\nQuery %d (orig=%d): item=%s Q=%.3f",
            idx, orig_idx, item_id, query_elo,
        )

        final_q_per_k, qry_logs_per_k = _run_query_vs_refs(
            query_image=prepare_image(
                image_path, cfg.data_dir, cfg.mask, input_size=None,
            ),
            ref_images=ref_images,
            query_elo=query_elo,
            ref_elos_per_k=cached_elos_per_k,
            model=model,
            processor=processor,
            cfg=cfg,
            rng=query_rng,
        )

        elapsed = time.time() - t0
        total_time += elapsed

        # ---- Build per-k compact model_output for CSV ----
        zs_refs = " ".join(
            f"R{i+1}={ref_zs.get(ref_paths[i], 0.5):.3f}"
            for i in range(len(ref_paths))
        )
        zs_part = f"ZS:Q={query_elo:.3f} {zs_refs}"

        for k in k_values:
            if item_id not in items_logged:
                ref_part = f"Ref:{ref_logs_per_k[k]}"
            else:
                elos_str = " ".join(
                    f"R{i+1}={e:.3f}"
                    for i, e in enumerate(cached_elos_per_k[k])
                )
                ref_part = f"Ref:[cached] {elos_str}"

            final_q = final_q_per_k[k]
            text_per_k[k][idx] = (
                f"{zs_part} | {ref_part} | "
                f"Qry:{qry_logs_per_k[k]} | F={final_q:.3f}"
            )
            scores_per_k[k][idx] = final_q

            if pair_label not in pair_scores_per_k[k]:
                pair_scores_per_k[k][pair_label] = {
                    "indices": [], "scores": [], "detections": [],
                }
            pair_scores_per_k[k][pair_label]["indices"].append(idx)
            pair_scores_per_k[k][pair_label]["scores"].append(final_q)
            pair_scores_per_k[k][pair_label]["detections"].append(
                {"boxes": [], "scores": []},
            )

        items_logged.add(item_id)
        seen_mask[idx] = True
        processed += 1

        logger.info(
            "  Final Q per k: %s  (%.1fs)",
            "  ".join(
                f"k={k}:{final_q_per_k[k]:.3f}" for k in k_values
            ),
            elapsed,
        )

        torch.cuda.empty_cache()

        # ---- Periodic checkpoint (save all per-k CSVs) ----
        if samples_per_save > 0:
            if processed - last_save_count >= samples_per_save:
                for k in k_values:
                    csv_path = _k_csv_path(exp_dir, k)
                    n_saved = write_checkpoint_csv(
                        df, scores_per_k[k], labels, seen_mask,
                        cfg.data_dir, cfg.crop, csv_path,
                        model_outputs=text_per_k[k],
                    )
                logger.info(
                    "  [Checkpoint] Saved %d rows to %d per-k CSVs",
                    n_saved, len(k_values),
                )
                last_save_count = processed

        # ---- Periodic progress report (uses first k) ----
        now = time.time()
        if now - last_report_time >= report_interval_s:
            if labels is not None:
                pk = k_values[0]
                scored_labels = labels[seen_mask]
                scored_scores = scores_per_k[pk][seen_mask]
                n_pos = int(scored_labels.sum())
                n_neg = int(len(scored_labels) - n_pos)
                if n_pos > 0 and n_neg > 0:
                    from sklearn.metrics import average_precision_score
                    interim_ap = average_precision_score(
                        scored_labels, scored_scores,
                    )
                    logger.info(
                        "\n  [Interim k=%.4f — %d/%d, %.1f min]  "
                        "AP=%.4f  pos=%d neg=%d\n",
                        pk, processed, num_images, total_time / 60,
                        interim_ap, n_pos, n_neg,
                    )
                else:
                    logger.info(
                        "\n  [Interim — %d/%d, %.1f min]  "
                        "AP=N/A  pos=%d neg=%d\n",
                        processed, num_images, total_time / 60,
                        n_pos, n_neg,
                    )
            else:
                logger.info(
                    "\n  [Interim — %d/%d, %.1f min]\n",
                    processed, num_images, total_time / 60,
                )
            last_report_time = now

    logger.info("Total query-match time: %.2fs", total_time)
    if processed > 0:
        logger.info(
            "Average per query: %.1fms", total_time / processed * 1000,
        )

    return scores_per_k, pair_scores_per_k, text_per_k
