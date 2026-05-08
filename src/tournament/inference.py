"""
Tournament-based VLM inference for anomaly detection.

Two strategies:
  - **Simple Ranking**: present all images to the model at once and ask it to
    rank them from most to least anomalous.  The query's rank determines
    its anomaly score via ``score = 1 - (rank - 1) / m``, with *rank* in
    ``{1, …, m + 1}`` (1 = most anomalous) and *m* the number of reference images.
  - **League**: pairwise comparisons with confidence-based scoring.  Two
    scoring modes: ``text`` (generate "CONFIDENCE: <float>") or ``logits``
    (P(Yes) from next-token yes/no logits — avoids discretisation bias).
    The winner accumulates +confidence, the loser +(1 − confidence).
    Two league types:
      * ``swiss`` — Swiss-system pairing over ⌈log₂(m+1)⌉ rounds.
      * ``complete`` — full round-robin: every pair plays exactly once.

Both strategies hide which image is the query: images are shuffled and
labelled generically (Image 1 … Image N, or Image A / Image B).
"""

import itertools
import logging
import math
import os
import re
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import average_precision_score
from tqdm import tqdm

from src.common.data import PairKey, prepare_image, write_checkpoint_csv
from src.tournament.config import TournamentConfig
from src.vlm.inference import (
    _apply_chat_template,
    _get_yes_no_token_ids,
    ensure_pad_token_for_generation,
)

logger = logging.getLogger("vlm")


# ---------------------------------------------------------------------------
# Reference lookup (variable count, no padding)
# ---------------------------------------------------------------------------

def build_tournament_reference_lookup(
    data_dir: str,
    split: str,
    crop: bool,
    num_references: int,
) -> dict[str, list[str]]:
    """
    Load ``reference-{split}.parquet`` and return up to *num_references*
    relative paths per ``item_identifier``.  No padding/cycling — if an
    item has fewer references than *num_references*, only the available
    ones are returned.
    """
    ref_path = os.path.join(data_dir, f"reference-{split}.parquet")
    if not os.path.isfile(ref_path):
        raise FileNotFoundError(
            f"Tournament mode requires {ref_path}."
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
    ref_count_summary: dict[int, int] = {}

    for item_id, group in ref_df.groupby("item_identifier", sort=False):
        paths = group[col].tolist()
        key = str(item_id)
        n = min(len(paths), num_references)
        paths = paths[:n]
        if not paths:
            raise ValueError(
                f"No reference images for item_identifier={key!r}."
            )
        lookup[key] = paths
        ref_count_summary[n] = ref_count_summary.get(n, 0) + 1

    for n_ref in sorted(ref_count_summary, reverse=True):
        logger.info(
            "Tournament refs (max %d): %d ref(s) → %d item(s)",
            num_references, n_ref, ref_count_summary[n_ref],
        )

    return lookup


# ---------------------------------------------------------------------------
# Grid composition
# ---------------------------------------------------------------------------

_GRID_LAYOUTS: dict[int, tuple[int, int]] = {
    2: (1, 2),
    3: (2, 2),
    4: (2, 2),
}


def compose_tournament_grid(
    images: list[Image.Image],
    output_size: int,
) -> Image.Image:
    """
    Compose *images* into a grid where each cell is *output_size* × *output_size*.

    Layouts: 2 → 1×2, 3 → 2×2 (one blank cell), 4 → 2×2.
    The resulting canvas is ``(cols * output_size, rows * output_size)``.
    """
    n = len(images)
    if n not in _GRID_LAYOUTS:
        raise ValueError(
            f"Tournament grid supports 2–4 images, got {n}."
        )
    rows, cols = _GRID_LAYOUTS[n]
    cell_w = output_size
    cell_h = output_size
    canvas = Image.new("RGB", (cols * cell_w, rows * cell_h), (128, 128, 128))

    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        resized = img.resize((cell_w, cell_h), Image.LANCZOS)
        canvas.paste(resized, (c * cell_w, r * cell_h))

    return canvas


# ---------------------------------------------------------------------------
# Image-position descriptions for prompts
# ---------------------------------------------------------------------------

_GRID_POSITIONS_2 = ["left", "right"]
_GRID_POSITIONS_4 = ["top-left", "top-right", "bottom-left", "bottom-right"]


def _grid_image_description_ranking(num_images: int) -> str:
    """Textual description of image positions in a ranking grid."""
    if num_images == 2:
        positions = _GRID_POSITIONS_2
    elif num_images in (3, 4):
        positions = _GRID_POSITIONS_4[:num_images]
    else:
        raise ValueError(f"Unsupported grid count: {num_images}")

    lines = [
        f"The image shows a grid of {num_images} product images."
    ]
    for i, pos in enumerate(positions, 1):
        lines.append(f"- Image {i}: {pos}")

    return "\n".join(lines)


def _separate_image_description_ranking(num_images: int) -> str:
    labels = ", ".join(f"Image {i}" for i in range(1, num_images + 1))
    return (
        f"You are shown {num_images} product images provided separately, "
        f"in order: {labels}."
    )


def _grid_image_description_league() -> str:
    return (
        "The image shows two product images side by side.\n"
        "- Image A: left\n"
        "- Image B: right"
    )


def _separate_image_description_league() -> str:
    return (
        "You are shown two product images. "
        "The first image is Image A, the second is Image B."
    )


# ---------------------------------------------------------------------------
# Message builders
# ---------------------------------------------------------------------------

def _build_ranking_messages(
    images: list[Image.Image],
    cfg: TournamentConfig,
) -> list[dict]:
    """
    Build chat messages for ranking inference.

    When ``cfg.use_grid`` is True a single grid image is attached;
    otherwise each image is a separate content block.
    """
    num_images = len(images)
    if cfg.use_grid:
        image_desc = _grid_image_description_ranking(num_images)
    else:
        image_desc = _separate_image_description_ranking(num_images)

    user_text = cfg.user_prompt_template().format(
        image_description=image_desc,
        num_images=num_images,
    )

    if cfg.use_grid:
        grid = compose_tournament_grid(images, cfg.input_size)
        content: list[dict] = [
            {"type": "image", "image": grid},
            {"type": "text", "text": user_text},
        ]
    else:
        content = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": user_text})

    return [
        {"role": "system", "content": cfg.system_prompt()},
        {"role": "user", "content": content},
    ]


def _build_league_messages(
    image_a: Image.Image,
    image_b: Image.Image,
    cfg: TournamentConfig,
) -> list[dict]:
    """Build chat messages for a single league pairwise comparison."""
    if cfg.use_grid:
        image_desc = _grid_image_description_league()
    else:
        image_desc = _separate_image_description_league()

    user_text = cfg.user_prompt_template().format(
        image_description=image_desc,
        num_images=2,
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
        {"role": "system", "content": cfg.system_prompt()},
        {"role": "user", "content": content},
    ]


# ---------------------------------------------------------------------------
# Text generation
# ---------------------------------------------------------------------------

def _generate_text(
    model,
    processor,
    messages: list[dict],
    cfg: TournamentConfig,
) -> str:
    """Run a single forward pass with text generation and return the reply."""
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
# Logits-based scoring for league pairwise comparisons
# ---------------------------------------------------------------------------

def _score_from_logits_league(
    model,
    processor,
    messages: list[dict],
    cfg: TournamentConfig,
    yes_ids: list[int],
    no_ids: list[int],
) -> float:
    """
    Compute P(Yes) from next-token logits for a league pairwise comparison.

    Performs one forward pass (no generation) and returns
    ``softmax(max_yes_logit, max_no_logit)[0]`` — the probability that the
    model would answer "Yes" to the question "Is Image A more anomalous
    than Image B?".  This value serves as the confidence score.
    """
    inputs = _apply_chat_template(
        processor, messages, cfg.enable_thinking,
    )
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
# Output parsers
# ---------------------------------------------------------------------------

def _parse_ranking(text: str, num_images: int) -> list[int] | None:
    """
    Extract a ranking from model output.

    Looks for ``RANKING: 3, 1, 4, 2`` (comma- or space-separated).
    Returns a list of 1-indexed image numbers from most-anomalous to
    least, or ``None`` on parse failure.
    """
    match = re.search(
        r"RANKING:\s*([\d,\s]+)", text, re.IGNORECASE,
    )
    if not match:
        nums = re.findall(r"\d+", text)
        if len(nums) == num_images:
            ranking = [int(n) for n in nums]
        else:
            return None
    else:
        ranking = [int(n) for n in re.findall(r"\d+", match.group(1))]

    if len(ranking) != num_images:
        return None

    expected = set(range(1, num_images + 1))
    if set(ranking) != expected:
        return None

    return ranking


def _participant_label(original_idx: int) -> str:
    """
    Stable display name before shuffle permutations.

    Index 0 is the query slot; indices ``1 … m`` are reference slots
    (``Ref`` numbering matches ``all_images[k]`` for ``k >= 1``).
    """
    if original_idx == 0:
        return "Query"
    return f"Ref{original_idx}"


def _collapse_text_for_logs(text: str, max_len: int = 240) -> str:
    """Flatten whitespace and truncate for CSV-safe single-line embedding."""
    one = " ".join(text.split())
    if len(one) <= max_len:
        return one
    return one[: max_len - 3] + "..."


def _parse_confidence(text: str) -> float:
    """
    Extract a confidence value from model output.

    Looks for ``CONFIDENCE: 0.73``.  Falls back to the first float in
    the text, and ultimately defaults to 0.5.
    """
    match = re.search(
        r"CONFIDENCE:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))",
        text,
        re.IGNORECASE,
    )
    if match:
        return min(max(float(match.group(1)), 0.0), 1.0)

    numeric = re.search(r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))", text)
    if numeric:
        return min(max(float(numeric.group(1)), 0.0), 1.0)

    return 0.5


# ---------------------------------------------------------------------------
# Strategy: Simple Ranking
# ---------------------------------------------------------------------------

def _run_simple_ranking_single(
    query_image: Image.Image,
    ref_images: list[Image.Image],
    model,
    processor,
    cfg: TournamentConfig,
    rng: np.random.Generator,
) -> tuple[float, str]:
    """
    Run one query through the simple-ranking strategy.

    Repeats ``cfg.repeat`` times with different shuffles and averages the
    resulting scores.

    Score formula (1-based rank in ``{1, …, m + 1}``):
        ``score = 1 − (rank − 1) / m``
    where *m* = number of reference images (so scores span ``[0, 1]``).

    Returns ``(averaged_score, human_readable_summary)``.
    """
    all_images = [query_image] + ref_images
    num_images = len(all_images)
    m = len(ref_images)
    scores: list[float] = []
    repeat_blocks: list[str] = []

    for rep_i in range(cfg.repeat):
        perm = rng.permutation(num_images)
        shuffled = [all_images[i] for i in perm]
        # 1-indexed label assigned to the query in this shuffle
        query_label = int(np.where(perm == 0)[0][0]) + 1

        messages = _build_ranking_messages(shuffled, cfg)
        reply = _generate_text(model, processor, messages, cfg)

        lines: list[str] = [
            f"Repeat {rep_i + 1}/{cfg.repeat}",
            "Generic label → participant (this shuffle):",
        ]
        for g in range(1, num_images + 1):
            orig = int(perm[g - 1])
            lines.append(f"  Image {g}: {_participant_label(orig)}")
        lines.append("")
        lines.append(f"Model output: {_collapse_text_for_logs(reply)}")

        ranking = _parse_ranking(reply, num_images)
        if ranking is None:
            logger.warning(
                "Could not parse ranking from model output — "
                "defaulting to score 0.5.  Output: %s",
                reply[:200],
            )
            scores.append(0.5)
            lines.append("")
            lines.append(
                "Parsed ranking: (failed — scored as 0.5 for this repeat)"
            )
            repeat_blocks.append("\n".join(lines))
            continue

        rank_1based = ranking.index(query_label) + 1
        score = 1.0 - (rank_1based - 1) / m
        scores.append(score)

        ordered_names = [
            _participant_label(int(perm[g - 1])) for g in ranking
        ]
        lines.append("")
        lines.append(
            "Final ranking (most anomalous first): "
            + ", ".join(ordered_names)
        )
        lines.append(
            f"Query rank (1 = most anomalous): {rank_1based} / {num_images}  "
            f"→ score = 1 - (rank-1)/{m} = {score:.6f}"
        )
        repeat_blocks.append("\n".join(lines))

    mean_score = float(np.mean(scores))
    blocks = "\n\n".join(repeat_blocks)
    if cfg.repeat > 1:
        blocks += (
            f"\n\n---\nMean score across {cfg.repeat} repeats: {mean_score:.6f}"
        )
    return mean_score, blocks


# ---------------------------------------------------------------------------
# Strategy: League (Swiss or Complete)
# ---------------------------------------------------------------------------

def _play_match(
    idx_a: int,
    idx_b: int,
    images: list[Image.Image],
    perm: np.ndarray,
    model,
    processor,
    cfg: TournamentConfig,
    rng: np.random.Generator,
    league_scores: list[float],
    match_lines: list[str],
    yes_ids: list[int] | None = None,
    no_ids: list[int] | None = None,
) -> None:
    """
    Play one pairwise match: show two images to the VLM, obtain a
    confidence score, update *league_scores*, and append a readable
    summary line to *match_lines*.

    When ``cfg.scoring_mode == "logits"``, P(Yes) is computed from
    next-token logits (no generation).  Otherwise the model generates
    text and the confidence is parsed from the reply.

    *perm[slot]* is the original participant index (0 = query, 1..m = refs)
    for the image at tournament index *slot*.
    """
    if rng.random() < 0.5:
        show_a, show_b = idx_a, idx_b
    else:
        show_a, show_b = idx_b, idx_a

    messages = _build_league_messages(
        images[show_a], images[show_b], cfg,
    )

    if cfg.scoring_mode == "logits":
        confidence = _score_from_logits_league(
            model, processor, messages, cfg, yes_ids, no_ids,
        )
        reply = f"(logits) P(Yes)={confidence:.6f}"
    else:
        reply = _generate_text(model, processor, messages, cfg)
        confidence = _parse_confidence(reply)

    league_scores[show_a] += confidence
    league_scores[show_b] += 1.0 - confidence

    first = _participant_label(int(perm[show_a]))
    second = _participant_label(int(perm[show_b]))
    k = len(match_lines) + 1
    match_lines.append(
        f"Match {k}: {first} {confidence:.4f} - {1.0 - confidence:.4f} "
        f"{second}  |  {_collapse_text_for_logs(reply)}"
    )

    torch.cuda.empty_cache()


def _league_final_ranking_block(
    perm: np.ndarray,
    league_scores: list[float],
) -> str:
    """Format participants by descending raw league points (ties by slot)."""
    order = sorted(
        range(len(league_scores)),
        key=lambda i: (-league_scores[i], i),
    )
    lines = ["Final ranking:"]
    for pos, slot in enumerate(order, 1):
        orig = int(perm[slot])
        lines.append(
            f"  {pos}. {_participant_label(orig)}  "
            f"(raw_points={league_scores[slot]:.4f})",
        )
    return "\n".join(lines)


def _run_league_single(
    query_image: Image.Image,
    ref_images: list[Image.Image],
    model,
    processor,
    cfg: TournamentConfig,
    rng: np.random.Generator,
    yes_ids: list[int] | None = None,
    no_ids: list[int] | None = None,
) -> tuple[float, str]:
    """
    Run one query through the league strategy.

    ``cfg.league_type`` selects the pairing scheme:

    * ``swiss`` — ⌈log₂(num_images)⌉ rounds; each round sorts by score
      and pairs adjacent participants.  Odd-one-out gets a bye (+0.5).
    * ``complete`` — full round-robin: every pair plays exactly once.

    In both cases the winner of each match gains +confidence and the
    loser gains +(1 − confidence).  When ``cfg.scoring_mode == "logits"``,
    the confidence is P(Yes) from next-token logits instead of parsed
    generated text.

    The query's accumulated score is min-max normalised to [0, 1] over
    all participants.

    Returns ``(normalised_score, human_readable_summary)``.
    """
    all_images = [query_image] + ref_images
    num_images = len(all_images)

    perm = rng.permutation(num_images)
    images = [all_images[i] for i in perm]
    query_idx = int(np.where(perm == 0)[0][0])

    league_scores = [0.0] * num_images
    match_lines: list[str] = []

    if cfg.league_type == "swiss":
        num_rounds = max(1, math.ceil(math.log2(num_images)))

        for _ in range(num_rounds):
            sorted_indices = sorted(
                range(num_images),
                key=lambda i: league_scores[i],
                reverse=True,
            )

            pairs: list[tuple[int, int]] = []
            for i in range(0, len(sorted_indices) - 1, 2):
                pairs.append((sorted_indices[i], sorted_indices[i + 1]))

            if len(sorted_indices) % 2 == 1:
                bye_idx = sorted_indices[-1]
                league_scores[bye_idx] += 0.5
                bye_name = _participant_label(int(perm[bye_idx]))
                match_lines.append(
                    f"Bye: {bye_name} +0.5 league points "
                    "(odd player out this round)",
                )

            for idx_a, idx_b in pairs:
                _play_match(
                    idx_a, idx_b, images, perm, model, processor,
                    cfg, rng, league_scores, match_lines,
                    yes_ids=yes_ids, no_ids=no_ids,
                )
    else:
        all_pairs = list(itertools.combinations(range(num_images), 2))
        rng.shuffle(all_pairs)
        for idx_a, idx_b in all_pairs:
            _play_match(
                idx_a, idx_b, images, perm, model, processor,
                cfg, rng, league_scores, match_lines,
                yes_ids=yes_ids, no_ids=no_ids,
            )

    # Min–max normalise the query's score to [0, 1].
    q_score = league_scores[query_idx]
    lo, hi = min(league_scores), max(league_scores)
    if hi > lo:
        normalised = (q_score - lo) / (hi - lo)
    else:
        normalised = 0.5

    body = "\n".join(match_lines)
    ranking_block = _league_final_ranking_block(perm, league_scores)
    return normalised, body + "\n\n" + ranking_block


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_tournament_inference(
    df: pd.DataFrame,
    model,
    processor,
    cfg: TournamentConfig,
    *,
    checkpoint_path: str | None = None,
    samples_per_save: int = 0,
    resume_data: dict[int, dict] | None = None,
) -> tuple[np.ndarray, dict[PairKey, dict[str, list]], list[str] | None]:
    """
    Run tournament inference on every row of *df*.

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

    Returns the same triple as ``vlm_inference.run_vlm_inference`` so that
    downstream metrics and plotting code can be reused:

    * ``max_scores``  — one anomaly score per image.
    * ``pair_scores`` — scores keyed by ``(strategy, "tournament")``.
    * ``text_outputs`` — per-sample readable summaries (prompt mapping,
      match lines, rankings, flattened model excerpts).
    """
    col = "query_crop" if cfg.crop else "query_image"
    pair_label: PairKey = (cfg.tournament_strategy, "tournament")

    if "item_identifier" not in df.columns:
        raise ValueError(
            "Tournament mode requires an 'item_identifier' column in the "
            "query dataframe."
        )

    ref_lookup = build_tournament_reference_lookup(
        cfg.data_dir, cfg.split, cfg.crop, cfg.num_references,
    )

    num_images = len(df)
    max_scores = np.zeros(num_images, dtype=np.float64)
    seen_mask = np.zeros(num_images, dtype=bool)
    pair_scores: dict[PairKey, dict[str, list]] = {}
    text_outputs: list[str] = [""] * num_images

    rng = np.random.default_rng(seed=42)

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
                if "model_output" in data:
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
            "Resumed %d / %d images from checkpoint.",
            len(resumed_rows), num_images,
        )

    # Estimate inference calls per sample for logging.
    if cfg.tournament_strategy == "simple_ranking":
        calls_per_sample = cfg.repeat
    else:
        avg_refs = np.mean([len(v) for v in ref_lookup.values()])
        avg_n = avg_refs + 1
        if cfg.league_type == "complete":
            calls_per_sample = int(avg_n * (avg_n - 1) / 2)
        else:
            avg_rounds = max(1, math.ceil(math.log2(avg_n)))
            calls_per_sample = int(avg_n / 2) * avg_rounds

    # Pre-compute yes/no token IDs for logits scoring mode.
    yes_ids: list[int] | None = None
    no_ids: list[int] | None = None
    if cfg.scoring_mode == "logits":
        yes_ids, no_ids = _get_yes_no_token_ids(processor.tokenizer)
        logger.info("Yes token IDs: %s, No token IDs: %s", yes_ids, no_ids)

    strategy_label = cfg.tournament_strategy
    if cfg.tournament_strategy == "league":
        strategy_label = f"league/{cfg.league_type}"

    logger.info(
        "Tournament: strategy=%s  scoring_mode=%s  num_references=%d  "
        "use_grid=%s  repeat=%d  images=%d  remaining=%d  "
        "~%d VLM call(s)/sample",
        strategy_label,
        cfg.scoring_mode,
        cfg.num_references,
        cfg.use_grid,
        cfg.repeat,
        num_images,
        num_images - len(resumed_rows),
        calls_per_sample,
    )

    total_time = 0.0
    labels = df["defect"].astype(int).values
    report_interval_s = cfg.report_interval_minutes * 60
    last_report_time = time.time()
    processed = 0
    last_save_count = 0

    for idx in tqdm(
        range(num_images),
        desc=f"Tournament ({cfg.tournament_strategy})",
    ):
        if idx in resumed_rows:
            continue

        image_path = df.iloc[idx][col]
        item_id = str(df.iloc[idx]["item_identifier"])

        query_image = prepare_image(
            image_path, cfg.data_dir, cfg.mask, input_size=None,
        )

        ref_paths = ref_lookup.get(item_id, [])
        if not ref_paths:
            logger.warning(
                "No references for item %s — defaulting to score 0.5.", item_id,
            )
            max_scores[idx] = 0.5
            text_outputs[idx] = (
                "Skipped: no reference images for this item "
                "(score defaulted to 0.5)."
            )
            seen_mask[idx] = True
            processed += 1
            continue

        ref_images = [
            prepare_image(rp, cfg.data_dir, cfg.mask, input_size=None)
            for rp in ref_paths
        ]

        t0 = time.time()

        if cfg.tournament_strategy == "simple_ranking":
            score, text_summary = _run_simple_ranking_single(
                query_image, ref_images, model, processor, cfg, rng,
            )
        else:
            score, text_summary = _run_league_single(
                query_image, ref_images, model, processor, cfg, rng,
                yes_ids=yes_ids, no_ids=no_ids,
            )

        elapsed = time.time() - t0
        total_time += elapsed

        max_scores[idx] = score
        seen_mask[idx] = True
        text_outputs[idx] = text_summary
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
                    "  [Checkpoint] Saved %d rows to %s",
                    n_saved, checkpoint_path,
                )
                last_save_count = processed

        # --- Periodic progress report ---
        now = time.time()
        if now - last_report_time >= report_interval_s:
            scored_labels = labels[seen_mask]
            scored_scores = max_scores[seen_mask]
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
                    "\n  [Interim report — %d/%d images, "
                    "%.1f min elapsed]\n"
                    "    AP = %.4f  (random baseline AP = %.4f)\n"
                    "    pos=%d, neg=%d  (%.1f%% positive)\n",
                    processed, num_images, total_time / 60,
                    interim_ap, random_ap, n_pos, n_neg, pos_pct,
                )
            else:
                logger.info(
                    "\n  [Interim report — %d/%d images, "
                    "%.1f min elapsed]\n"
                    "    AP = N/A (only one class seen so far)\n"
                    "    pos=%d, neg=%d  (%.1f%% positive)\n",
                    processed, num_images, total_time / 60,
                    n_pos, n_neg, pos_pct,
                )
            last_report_time = now

    logger.info("Total inference time: %.2fs", total_time)
    if processed > 0:
        logger.info(
            "Average per image:    %.1fms",
            total_time / processed * 1000,
        )

    return max_scores, pair_scores, text_outputs
