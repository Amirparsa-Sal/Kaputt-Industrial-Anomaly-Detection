"""
ELO Tournament Match Prompt Testing GUI.

A Gradio-based interface for interactively testing the pairwise comparison
prompt used in the ELO tournament pipeline (confidence mode). For each query
image the GUI shows its reference images, pre-computed zero-shot scores from
CSV files, ground-truth labels, and lets you run query-vs-reference matches
with editable prompts.

Usage:
    python gui/elo_match_gui.py \
        --config configs/elo_tournament/elo_tournament_base.yaml
    python gui/elo_match_gui.py \
        --config configs/elo_tournament/elo_tournament_base.yaml \
        --gpu 1 --port 7864
    python gui/elo_match_gui.py \
        --config configs/elo_tournament/elo_tournament_base.yaml \
        --override configs/elo_tournament/elo_tournament_wdl.yaml
"""

import argparse
import csv
import os
import re
import time
from dataclasses import dataclass

import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image

matplotlib.use("Agg")

LOGICAL_DEFECT_TYPES = ["missing_unit", "actuation"]
STRUCTURAL_DEFECT_TYPES = [
    "deformation", "deconstruction", "penetration", "superficial", "spillage",
]
ALL_DEFECT_CLASSES = STRUCTURAL_DEFECT_TYPES + LOGICAL_DEFECT_TYPES


# ---------------------------------------------------------------------------
# Persistent VLM model holder
# ---------------------------------------------------------------------------

@dataclass
class VLMHolder:
    """Keeps the Qwen VLM model and processor in GPU memory across requests."""

    model: object = None
    processor: object = None
    model_name: str = ""
    device: str = "cuda"

    def load(
        self,
        model_name: str,
        cache_dir: str | None,
        gpu_id: int = 0,
        load_in_4bit: bool = False,
        min_pixels: int = 200704,
        max_pixels: int = 401408,
    ):
        if self.model is not None and self.model_name == model_name:
            return

        from transformers import AutoProcessor

        try:
            from transformers import Qwen3_5ForConditionalGeneration
            ModelClass = Qwen3_5ForConditionalGeneration
            print("[VLM] Using Qwen3_5ForConditionalGeneration.")
        except ImportError:
            from transformers import AutoModelForImageTextToText
            ModelClass = AutoModelForImageTextToText
            print("[VLM] Falling back to AutoModelForImageTextToText.")

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        print(f"[VLM] Loading processor from '{model_name}' ...")
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=cache_dir,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )

        model_kwargs = dict(
            device_map=f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu",
            torch_dtype=dtype,
            trust_remote_code=True,
            cache_dir=cache_dir,
        )
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
            )
            model_kwargs.pop("torch_dtype", None)
            print("[VLM] 4-bit quantization enabled (NF4).")

        print(f"[VLM] Loading model '{model_name}' ...")
        self.model = ModelClass.from_pretrained(model_name, **model_kwargs)
        self.model.eval()

        _ensure_pad_token(self.processor, self.model)
        self.model_name = model_name
        print("[VLM] Model loaded and ready.")

    @property
    def ready(self) -> bool:
        return self.model is not None


_holder = VLMHolder()


# ---------------------------------------------------------------------------
# Token / pad helpers
# ---------------------------------------------------------------------------

def _ensure_pad_token(processor, model) -> None:
    """Align pad/eos token IDs so generate() doesn't emit warnings."""
    tok = getattr(processor, "tokenizer", None)
    if tok is None:
        return
    if getattr(tok, "pad_token", None) is None and getattr(tok, "eos_token", None):
        tok.pad_token = tok.eos_token
    if getattr(tok, "pad_token_id", None) is None and getattr(tok, "eos_token_id", None) is not None:
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


# ---------------------------------------------------------------------------
# Browser state
# ---------------------------------------------------------------------------

@dataclass
class BrowserState:
    """Mutable state shared by the Gradio callbacks."""

    df: pd.DataFrame | None = None
    current_idx: int = 0
    data_dir: str = ""
    ref_lookup: dict | None = None
    query_scores: dict | None = None   # parquet iloc key → predicted_score (CSV original_index)
    ref_scores: dict | None = None     # image_path → predicted_score


_state = BrowserState()


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_yaml_config(path: str) -> dict:
    with open(path) as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a YAML mapping, got {type(raw)}")
    return raw


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


# ---------------------------------------------------------------------------
# CSV loaders for zero-shot scores
# ---------------------------------------------------------------------------

def load_query_csv(csv_path: str) -> dict[int, float]:
    """
    Load zero_shot_queries.csv → {original_index: predicted_score}.

    ``original_index`` matches the row label stored by the training pipeline:
    the parquet row index at load time (see ``load_and_filter``), i.e. the
    same row you get from ``dataset_df.iloc[original_index]`` on the raw
    query parquet before filtering.
    """
    result: dict[int, float] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            orig_idx = int(row["original_index"])
            score = float(row["predicted_score"])
            result[orig_idx] = score
    return result


def load_reference_csv(csv_path: str) -> dict[str, float]:
    """Load zero_shot_references.csv → {image_path: predicted_score}."""
    result: dict[str, float] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            path = row["image_path"]
            score = float(row["predicted_score"])
            result[path] = score
    return result


# ---------------------------------------------------------------------------
# Data loading and filtering
# ---------------------------------------------------------------------------

def load_and_filter(
    data_dir: str,
    split: str,
    is_defect: str,
    major_defect: str,
    defect_type: str,
    defect_class: str,
) -> pd.DataFrame:
    parquet_path = os.path.join(data_dir, f"query-{split}.parquet")
    if not os.path.isfile(parquet_path):
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    # Same convention as src.common.data.load_and_filter_data: preserve the
    # parquet row position before any filtering. Zero-shot CSV ``original_index``
    # refers to ``dataset_df.iloc[original_index]`` on the raw query parquet.
    df = df.copy()
    df["original_index"] = df.index.to_numpy()

    if is_defect == "true":
        df = df[df["defect"]].copy()
    elif is_defect == "false":
        df = df[~df["defect"]].copy()

    if major_defect == "true":
        df = df[~df["defect"] | df["major_defect"]].copy()
    elif major_defect == "false":
        df = df[~df["defect"] | ~df["major_defect"]].copy()

    if defect_type != "any":
        primary = df["defect_types"].str.split(",").str[0].fillna("")
        if defect_type == "structural":
            keep = ~df["defect"] | ~primary.isin(LOGICAL_DEFECT_TYPES)
        elif defect_type == "logical":
            keep = ~df["defect"] | primary.isin(LOGICAL_DEFECT_TYPES)
        else:
            keep = pd.Series(True, index=df.index)
        df = df[keep].copy()

    if defect_class and defect_class != "any":
        primary = df["defect_types"].str.split(",").str[0].fillna("")
        df = df[primary == defect_class].copy()

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Reference lookup (reuses src.tournament.inference logic inline to avoid
# heavy import chains that pull in sklearn etc. at startup)
# ---------------------------------------------------------------------------

def _build_reference_lookup(
    data_dir: str, split: str, crop: bool, num_references: int,
) -> dict[str, list[str]]:
    """
    Load reference-{split}.parquet and return up to *num_references*
    relative paths per item_identifier.
    """
    ref_path = os.path.join(data_dir, f"reference-{split}.parquet")
    if not os.path.isfile(ref_path):
        raise FileNotFoundError(f"Reference parquet not found: {ref_path}")

    ref_df = pd.read_parquet(ref_path)
    col = "reference_crop" if crop else "reference_image"

    if "item_identifier" not in ref_df.columns:
        raise ValueError("Reference parquet must contain 'item_identifier'.")
    if col not in ref_df.columns:
        raise ValueError(f"Reference parquet must contain '{col}'.")

    if "index" in ref_df.columns:
        ref_df = ref_df.sort_values("index", kind="mergesort")

    lookup: dict[str, list[str]] = {}
    for item_id, group in ref_df.groupby("item_identifier", sort=False):
        paths = group[col].tolist()[:num_references]
        if paths:
            lookup[str(item_id)] = paths

    return lookup


# ---------------------------------------------------------------------------
# Image preparation
# ---------------------------------------------------------------------------

def prepare_image(
    image_path: str,
    base_dir: str,
    apply_mask: bool,
    input_size: int | None = None,
) -> Image.Image:
    full_path = os.path.join(base_dir, image_path)
    image = Image.open(full_path).convert("RGB")

    if apply_mask:
        mask_path = full_path.replace("image", "mask").replace(".jpg", ".png")
        if os.path.isfile(mask_path):
            gt_mask = np.array(Image.open(mask_path))
            image_np = np.array(image)
            image_np[(gt_mask <= 1)] = [0, 0, 0]
            image = Image.fromarray(image_np)

    if input_size is not None:
        image = image.resize((input_size, input_size), Image.LANCZOS)

    return image


def _compose_grid(images: list[Image.Image], output_size: int) -> Image.Image:
    """Compose two images into a 1x2 side-by-side grid."""
    cell_w = output_size
    cell_h = output_size
    canvas = Image.new("RGB", (2 * cell_w, cell_h), (128, 128, 128))
    for i, img in enumerate(images[:2]):
        resized = img.resize((cell_w, cell_h), Image.LANCZOS)
        canvas.paste(resized, (i * cell_w, 0))
    return canvas


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def _render_image_with_badge(
    pil_image: Image.Image,
    title: str,
    badge_text: str,
    badge_color: str,
    sub_badge: str | None = None,
    sub_color: str | None = None,
) -> np.ndarray:
    """Render a PIL image with a title and one or two overlay badges."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(pil_image)

    ax.text(
        0.5, 0.04, badge_text,
        transform=ax.transAxes, ha="center", va="bottom",
        fontsize=16, fontweight="bold", color="white",
        bbox=dict(
            facecolor=badge_color, alpha=0.85,
            boxstyle="round,pad=0.4", edgecolor="white", linewidth=1.5,
        ),
    )

    if sub_badge:
        ax.text(
            0.98, 0.98, sub_badge,
            transform=ax.transAxes, ha="right", va="top",
            fontsize=10, fontweight="bold", color="white",
            bbox=dict(
                facecolor=sub_color or "#444", alpha=0.75,
                boxstyle="round,pad=0.3",
            ),
        )

    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.axis("off")
    fig.tight_layout(pad=0.5)

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return buf


def _render_plain(pil_image: Image.Image, title: str = "") -> np.ndarray:
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(pil_image)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.axis("off")
    fig.tight_layout(pad=0.5)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return buf


def _score_color(score: float) -> str:
    if score < 0.3:
        return "#00b894"
    elif score < 0.6:
        return "#fdcb6e"
    return "#d63031"


# ---------------------------------------------------------------------------
# Match message building (mirrors src/elo_tournament/inference._build_match_messages)
# ---------------------------------------------------------------------------

def _grid_image_description() -> str:
    return (
        "The image shows two product images side by side.\n"
        "- Image A: left\n"
        "- Image B: right"
    )


def _separate_image_description() -> str:
    return (
        "You are shown two product images. "
        "The first image is Image A, the second is Image B."
    )


def _build_match_messages(
    image_a: Image.Image,
    image_b: Image.Image,
    system_prompt: str,
    user_prompt_template: str,
    use_grid: bool,
    input_size: int,
) -> tuple[list[dict], Image.Image | None]:
    """
    Build chat messages for a pairwise comparison match.
    Returns (messages, grid_image_or_None).
    """
    if use_grid:
        image_desc = _grid_image_description()
    else:
        image_desc = _separate_image_description()

    user_text = user_prompt_template.format(image_description=image_desc)

    grid_img = None
    if use_grid:
        grid_img = _compose_grid([image_a, image_b], input_size)
        content: list[dict] = [
            {"type": "image", "image": grid_img},
            {"type": "text", "text": user_text},
        ]
    else:
        content = [
            {"type": "image", "image": image_a},
            {"type": "image", "image": image_b},
            {"type": "text", "text": user_text},
        ]

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]

    return messages, grid_img


# ---------------------------------------------------------------------------
# Text generation
# ---------------------------------------------------------------------------

def _apply_chat_template(processor, messages):
    kwargs = dict(
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    )
    try:
        return processor.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return processor.apply_chat_template(messages, **kwargs)


def _generate_text(
    messages: list[dict],
    temperature: float,
    max_new_tokens: int,
) -> str:
    """Run a single text-generation forward pass and return the reply."""
    _ensure_pad_token(_holder.processor, _holder.model)

    inputs = _apply_chat_template(_holder.processor, messages)
    inputs = {k: v.to(_holder.model.device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    gen_kwargs: dict = {"max_new_tokens": max_new_tokens}
    if temperature > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
    else:
        gen_kwargs["do_sample"] = False

    tok = _holder.processor.tokenizer
    pad_id = getattr(tok, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tok, "eos_token_id", None)
    if pad_id is not None:
        gen_kwargs["pad_token_id"] = pad_id

    with torch.no_grad():
        generated_ids = _holder.model.generate(**inputs, **gen_kwargs)

    new_tokens = generated_ids[0][input_len:]
    reply = _holder.processor.tokenizer.decode(
        new_tokens, skip_special_tokens=True,
    ).strip()

    del inputs, generated_ids
    torch.cuda.empty_cache()

    return reply


def _parse_confidence(text: str) -> float:
    """Extract CONFIDENCE: <float> from model output, fallback to 0.5."""
    match = re.search(
        r"CONFIDENCE:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))",
        text, re.IGNORECASE,
    )
    if match:
        return min(max(float(match.group(1)), 0.0), 1.0)

    numeric = re.search(r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))", text)
    if numeric:
        return min(max(float(numeric.group(1)), 0.0), 1.0)

    return 0.5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_input_size(val: str) -> int | None:
    if not val or val.strip().lower() in ("none", "null", ""):
        return None
    try:
        return int(val.strip())
    except ValueError:
        return None


def _get_query_zs_score(idx: int) -> float | None:
    """Get the zero-shot score for the query image at dataframe row idx."""
    if _state.query_scores is None or _state.df is None:
        return None
    has_orig = "original_index" in _state.df.columns
    orig_idx = int(_state.df.iloc[idx]["original_index"]) if has_orig else idx
    return _state.query_scores.get(orig_idx)


def _get_ref_zs_score(ref_path: str) -> float | None:
    """Get the zero-shot score for a reference image path."""
    if _state.ref_scores is None:
        return None
    return _state.ref_scores.get(ref_path)


# ---------------------------------------------------------------------------
# GUI callbacks
# ---------------------------------------------------------------------------

def apply_filters(
    data_dir, split, is_defect, major_defect, defect_type, defect_class,
    crop, num_references,
    query_csv_path, ref_csv_path,
):
    """Load dataset, build reference lookup, and load CSV scores."""
    if not data_dir or not os.path.isdir(data_dir):
        return "Invalid data directory.", gr.update(), gr.update(), None, [], ""

    try:
        df = load_and_filter(
            data_dir, split, is_defect, major_defect, defect_type, defect_class,
        )
    except Exception as e:
        return f"Error loading data: {e}", gr.update(), gr.update(), None, [], ""

    try:
        ref_lookup = _build_reference_lookup(data_dir, split, crop, int(num_references))
    except Exception as e:
        return f"Error loading references: {e}", gr.update(), gr.update(), None, [], ""

    _state.data_dir = data_dir
    _state.current_idx = 0
    _state.ref_lookup = ref_lookup

    _state.query_scores = None
    _state.ref_scores = None

    if query_csv_path and os.path.isfile(query_csv_path):
        try:
            _state.query_scores = load_query_csv(query_csv_path)
        except Exception as e:
            print(f"[Warning] Could not load query CSV: {e}")

    if ref_csv_path and os.path.isfile(ref_csv_path):
        try:
            _state.ref_scores = load_reference_csv(ref_csv_path)
        except Exception as e:
            print(f"[Warning] Could not load reference CSV: {e}")

    # When a query CSV is provided, keep only rows whose ``original_index``
    # appears in the CSV (aligned with experiment checkpoints).
    if _state.query_scores:
        csv_indices = set(_state.query_scores.keys())
        df = df[df["original_index"].astype(int).isin(csv_indices)].copy()
        df = df.sort_values("original_index", kind="mergesort").reset_index(
            drop=True,
        )

    _state.df = df

    n = len(df)
    csv_info = ""
    if _state.query_scores:
        csv_info += f"Query CSV: {len(_state.query_scores)} scores → {n} matching images. "
    if _state.ref_scores:
        csv_info += f"Reference CSV: {len(_state.ref_scores)} scores loaded."

    status = f"Loaded **{n}** images (split: {split}). {csv_info}"

    if n == 0:
        return status, gr.update(minimum=0, maximum=1, value=0), gr.update(), None, [], ""

    slider_max = max(n - 1, 1)
    slider_update = gr.update(minimum=0, maximum=slider_max, value=0, step=1)
    query_img, ref_gallery, info = _get_current_display(0, crop, False, None)
    return status, slider_update, gr.update(value=f"1 / {n}"), query_img, ref_gallery, info


def navigate(idx, crop, mask, input_size_str):
    if _state.df is None or len(_state.df) == 0:
        return None, [], "", gr.update()

    idx = max(0, min(int(idx), len(_state.df) - 1))
    _state.current_idx = idx
    input_size = _parse_input_size(input_size_str)
    query_img, ref_gallery, info = _get_current_display(idx, crop, mask, input_size)
    counter = f"{idx + 1} / {len(_state.df)}"
    return query_img, ref_gallery, info, gr.update(value=counter)


def go_prev(crop, mask, input_size_str, current_slider):
    if _state.df is None or len(_state.df) == 0:
        return None, [], "", gr.update(), gr.update()

    idx = max(0, int(current_slider) - 1)
    _state.current_idx = idx
    input_size = _parse_input_size(input_size_str)
    query_img, ref_gallery, info = _get_current_display(idx, crop, mask, input_size)
    n = len(_state.df)
    counter = f"{idx + 1} / {n}"
    return query_img, ref_gallery, info, gr.update(value=idx), gr.update(value=counter)


def go_next(crop, mask, input_size_str, current_slider):
    if _state.df is None or len(_state.df) == 0:
        return None, [], "", gr.update(), gr.update()

    n = len(_state.df)
    idx = min(n - 1, int(current_slider) + 1)
    _state.current_idx = idx
    input_size = _parse_input_size(input_size_str)
    query_img, ref_gallery, info = _get_current_display(idx, crop, mask, input_size)
    counter = f"{idx + 1} / {n}"
    return query_img, ref_gallery, info, gr.update(value=idx), gr.update(value=counter)


def _get_current_display(
    idx: int, crop: bool, mask: bool, input_size: int | None,
) -> tuple[np.ndarray | None, list, str]:
    """Return (query_render, ref_gallery_items, info_markdown)."""
    if _state.df is None or len(_state.df) == 0:
        return None, [], ""

    row = _state.df.iloc[idx]
    col = "query_crop" if crop else "query_image"
    image_path = row.get(col, row.get("query_image", ""))
    item_id = str(row.get("item_identifier", ""))

    is_defect = bool(row.get("defect", False))
    defect_types = str(row.get("defect_types", ""))
    is_major = bool(row.get("major_defect", False))
    gt_label = "DEFECT" if is_defect else "NORMAL"

    zs_score = _get_query_zs_score(idx)
    zs_str = f"{zs_score:.4f}" if zs_score is not None else "N/A"

    info = f"### Query Image {idx + 1}\n\n"
    info += f"**Path:** `{image_path}`\n\n"
    info += f"**Item ID:** `{item_id}`\n\n"
    info += f"**Ground Truth:** {gt_label}"
    if is_defect:
        info += f" | **Type:** {defect_types} | **Major:** {'Yes' if is_major else 'No'}"
    info += f"\n\n**Zero-Shot Score:** `{zs_str}`\n"

    try:
        pil_img = prepare_image(image_path, _state.data_dir, mask, input_size)
        badge = f"GT: {gt_label} | ZS: {zs_str}"
        badge_col = "#d63031" if is_defect else "#00b894"
        query_render = _render_image_with_badge(
            pil_img, f"Query — Image {idx + 1}", badge, badge_col,
        )
    except Exception:
        query_render = None

    ref_gallery = []
    ref_paths = _state.ref_lookup.get(item_id, []) if _state.ref_lookup else []
    for i, rp in enumerate(ref_paths):
        try:
            ref_pil = prepare_image(rp, _state.data_dir, mask, input_size)
            ref_zs = _get_ref_zs_score(rp)
            ref_zs_str = f"ZS: {ref_zs:.4f}" if ref_zs is not None else "ZS: N/A"
            ref_render = _render_image_with_badge(
                ref_pil, f"Ref {i + 1}", ref_zs_str, "#0984e3",
            )
            ref_gallery.append((ref_render, f"Ref {i + 1} — {ref_zs_str}"))
        except Exception:
            pass

    if ref_paths:
        info += f"\n**References:** {len(ref_paths)} image(s) for item `{item_id}`\n"
    else:
        info += f"\n**References:** None found for item `{item_id}`\n"

    return query_render, ref_gallery, info


def run_matches(
    system_prompt, user_prompt_template,
    crop, mask, input_size_str, use_grid,
    temperature, max_new_tokens,
    current_slider,
):
    """Run query vs. each reference match and return results."""
    if _state.df is None or len(_state.df) == 0:
        return "", []

    if not _holder.ready:
        return "**Model not loaded.** Wait for startup to finish.", []

    idx = int(current_slider)
    row = _state.df.iloc[idx]
    col = "query_crop" if crop else "query_image"
    query_path = row.get(col, row.get("query_image", ""))
    item_id = str(row.get("item_identifier", ""))
    input_size = _parse_input_size(input_size_str)
    effective_input_size = input_size if input_size else 1024

    ref_paths = _state.ref_lookup.get(item_id, []) if _state.ref_lookup else []
    if not ref_paths:
        return "**No reference images** for this item.", []

    query_pil = prepare_image(query_path, _state.data_dir, mask, input_size)

    results_md = "## Match Results\n\n"
    results_md += f"**Query Image {idx + 1}** vs. {len(ref_paths)} reference(s)\n\n"

    gallery_items = []

    for i, rp in enumerate(ref_paths):
        ref_pil = prepare_image(rp, _state.data_dir, mask, input_size)

        t0 = time.time()
        messages, grid_img = _build_match_messages(
            query_pil, ref_pil,
            system_prompt, user_prompt_template,
            use_grid, effective_input_size,
        )
        reply = _generate_text(messages, temperature, int(max_new_tokens))
        confidence = _parse_confidence(reply)
        elapsed_ms = (time.time() - t0) * 1000

        results_md += f"---\n\n"
        results_md += f"### Match {i + 1}: Query (A) vs Ref {i + 1} (B)\n\n"
        results_md += f"**CONFIDENCE:** `{confidence:.4f}`\n\n"
        results_md += f"**Inference Time:** {elapsed_ms:.0f} ms\n\n"

        ref_zs = _get_ref_zs_score(rp)
        if ref_zs is not None:
            results_md += f"**Ref ZS Score:** `{ref_zs:.4f}`\n\n"

        results_md += f"**Raw Reply:**\n```\n{reply}\n```\n\n"

        interpretation = _interpret_confidence(confidence)
        results_md += f"**Interpretation:** {interpretation}\n\n"

        vis_img = grid_img if grid_img else query_pil
        vis_render = _render_image_with_badge(
            vis_img,
            f"Match {i + 1}: Q vs R{i + 1}",
            f"CONFIDENCE: {confidence:.4f}",
            _score_color(confidence),
            sub_badge=f"{elapsed_ms:.0f}ms",
            sub_color="#636e72",
        )
        gallery_items.append((vis_render, f"Q vs R{i+1}: {confidence:.4f}"))

    return results_md, gallery_items


def _interpret_confidence(c: float) -> str:
    if c >= 0.8:
        return "Strong evidence query (A) is more anomalous"
    elif c >= 0.6:
        return "Moderate evidence query (A) is more anomalous"
    elif c >= 0.4:
        return "Roughly equal — little anomaly difference"
    elif c >= 0.2:
        return "Moderate evidence reference (B) is more anomalous"
    else:
        return "Strong evidence reference (B) is more anomalous"


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
.main-title {
    text-align: center;
    background: linear-gradient(135deg, #0c3547 0%, #1a6b4a 50%, #2d8659 100%);
    color: white !important;
    padding: 20px 30px;
    border-radius: 12px;
    margin-bottom: 16px;
    font-size: 1.5em;
}
.main-title h1 { color: white !important; margin: 0; }
.main-title p { color: #a0e8c0 !important; margin: 4px 0 0 0; font-size: 0.6em; }
.nav-btn { min-width: 100px !important; }
.run-btn {
    background: linear-gradient(135deg, #00b894, #00cec9) !important;
    color: white !important;
    font-weight: bold !important;
    font-size: 1.1em !important;
    border: none !important;
    min-height: 48px !important;
}
.counter-display {
    text-align: center;
    font-size: 1.2em;
    font-weight: bold;
    padding: 8px;
}
.prompt-editor textarea {
    font-family: monospace !important;
    font-size: 0.85em !important;
}
"""


def build_ui(default_cfg: dict) -> gr.Blocks:

    sys_prompt_init = default_cfg.get(
        "system_prompt_match_confidence", "",
    )
    usr_prompt_init = default_cfg.get(
        "user_prompt_match_confidence", "",
    )

    with gr.Blocks(
        title="ELO Match Prompt Tester",
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(
            primary_hue="green",
            secondary_hue="teal",
            neutral_hue="slate",
        ),
    ) as app:

        gr.HTML(
            '<div class="main-title">'
            "<h1>ELO Tournament Match Prompt Tester</h1>"
            "<p>Test pairwise comparison prompts — query vs. references with zero-shot scores</p>"
            "</div>"
        )

        with gr.Row():
            # ================================================================
            # LEFT COLUMN — Controls
            # ================================================================
            with gr.Column(scale=1, min_width=400):

                with gr.Group():
                    gr.Markdown("### Data Filters")
                    data_dir = gr.Textbox(
                        label="Data Directory",
                        value=default_cfg.get("data_dir", ""),
                        placeholder="/path/to/kaputt1/",
                    )
                    split = gr.Dropdown(
                        label="Split",
                        choices=["train", "validation", "test"],
                        value=default_cfg.get("split", "train"),
                    )
                    with gr.Row():
                        is_defect = gr.Dropdown(
                            label="Is Defect",
                            choices=["any", "true", "false"],
                            value=str(default_cfg.get("is_defect", "any")).lower(),
                        )
                        major_defect = gr.Dropdown(
                            label="Major Defect",
                            choices=["any", "true", "false"],
                            value=str(default_cfg.get("major_defect", "any")).lower(),
                        )
                    with gr.Row():
                        defect_type = gr.Dropdown(
                            label="Defect Type",
                            choices=["any", "structural", "logical"],
                            value=default_cfg.get("defect_type", "any"),
                        )
                        defect_class = gr.Dropdown(
                            label="Defect Class",
                            choices=["any"] + ALL_DEFECT_CLASSES,
                            value="any",
                        )

                with gr.Group():
                    gr.Markdown("### Image Processing")
                    with gr.Row():
                        crop = gr.Checkbox(
                            label="Use Crop",
                            value=default_cfg.get("crop", True),
                        )
                        mask = gr.Checkbox(
                            label="Apply Mask",
                            value=default_cfg.get("mask", False),
                        )
                    input_size = gr.Textbox(
                        label="Input Size (px, blank = original)",
                        value=str(default_cfg.get("input_size", "") or ""),
                        placeholder="e.g. 1024",
                    )
                    num_references = gr.Slider(
                        label="Max References per Item",
                        minimum=1, maximum=3, step=1,
                        value=default_cfg.get("num_references", 3),
                    )

                with gr.Group():
                    gr.Markdown("### Zero-Shot CSV Files")
                    query_csv = gr.Textbox(
                        label="Query Scores CSV",
                        value=default_cfg.get("zero_shot_query_csv", "") or "",
                        placeholder="/path/to/zero_shot_queries.csv",
                    )
                    ref_csv = gr.Textbox(
                        label="Reference Scores CSV",
                        value=default_cfg.get("zero_shot_reference_csv", "") or "",
                        placeholder="/path/to/zero_shot_references.csv",
                    )

                filter_btn = gr.Button(
                    "Load Dataset & References",
                    variant="primary", size="lg",
                )
                filter_status = gr.Markdown("")

                with gr.Group():
                    gr.Markdown("### Match Settings")
                    use_grid = gr.Checkbox(
                        label="Use Grid (side-by-side)",
                        value=default_cfg.get("use_grid", True),
                    )
                    temperature = gr.Slider(
                        label="Temperature",
                        minimum=0.0, maximum=1.5, step=0.05,
                        value=default_cfg.get("temperature", 0.0),
                    )
                    max_new_tokens = gr.Slider(
                        label="Max New Tokens",
                        minimum=32, maximum=1024, step=32,
                        value=default_cfg.get("max_new_tokens", 256),
                    )

                with gr.Group():
                    gr.Markdown("### Match Prompts (editable)")
                    system_prompt = gr.Textbox(
                        label="System Prompt",
                        value=sys_prompt_init,
                        lines=6,
                        elem_classes=["prompt-editor"],
                    )
                    user_prompt = gr.Textbox(
                        label="User Prompt Template (use {image_description})",
                        value=usr_prompt_init,
                        lines=14,
                        elem_classes=["prompt-editor"],
                    )

                run_btn = gr.Button(
                    "Run Matches (Query vs. Each Reference)",
                    variant="primary", size="lg",
                    elem_classes=["run-btn"],
                )

            # ================================================================
            # RIGHT COLUMN — Display
            # ================================================================
            with gr.Column(scale=2):

                with gr.Row():
                    prev_btn = gr.Button(
                        "◀  Previous", size="sm", elem_classes=["nav-btn"],
                    )
                    counter = gr.Textbox(
                        value="0 / 0", show_label=False, interactive=False,
                        elem_classes=["counter-display"], text_align="center",
                    )
                    next_btn = gr.Button(
                        "Next  ▶", size="sm", elem_classes=["nav-btn"],
                    )
                image_slider = gr.Slider(
                    label="Image Index",
                    minimum=0, maximum=1, step=1, value=0,
                )

                with gr.Tabs():
                    with gr.Tab("Query & References"):
                        query_image = gr.Image(
                            label="Query Image",
                            type="numpy", interactive=False,
                        )
                        image_info = gr.Markdown("")
                        gr.Markdown("#### Reference Images")
                        ref_gallery = gr.Gallery(
                            label="References (with zero-shot scores)",
                            columns=3, rows=1, height="auto",
                        )

                    with gr.Tab("Match Results"):
                        match_results_md = gr.Markdown(
                            "*Run matches to see results here.*",
                        )
                        match_gallery = gr.Gallery(
                            label="Match Visualizations",
                            columns=3, rows=2, height="auto",
                        )

        # ================================================================
        # Event wiring
        # ================================================================

        filter_btn.click(
            fn=apply_filters,
            inputs=[
                data_dir, split, is_defect, major_defect, defect_type, defect_class,
                crop, num_references,
                query_csv, ref_csv,
            ],
            outputs=[
                filter_status, image_slider, counter,
                query_image, ref_gallery, image_info,
            ],
        )

        image_slider.change(
            fn=navigate,
            inputs=[image_slider, crop, mask, input_size],
            outputs=[query_image, ref_gallery, image_info, counter],
        )

        prev_btn.click(
            fn=go_prev,
            inputs=[crop, mask, input_size, image_slider],
            outputs=[query_image, ref_gallery, image_info, image_slider, counter],
        )

        next_btn.click(
            fn=go_next,
            inputs=[crop, mask, input_size, image_slider],
            outputs=[query_image, ref_gallery, image_info, image_slider, counter],
        )

        run_btn.click(
            fn=run_matches,
            inputs=[
                system_prompt, user_prompt,
                crop, mask, input_size, use_grid,
                temperature, max_new_tokens,
                image_slider,
            ],
            outputs=[match_results_md, match_gallery],
        )

    return app


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ELO Tournament Match Prompt Testing GUI",
    )
    parser.add_argument(
        "--config", type=str,
        default="configs/elo_tournament/elo_tournament_base.yaml",
        help="Path to the ELO tournament YAML config file.",
    )
    parser.add_argument(
        "--override", type=str, default=None,
        help="Path to an override YAML config.",
    )
    parser.add_argument(
        "--gpu", type=int, default=0,
        help="GPU device index (default: 0).",
    )
    parser.add_argument(
        "--port", type=int, default=7864,
        help="Port to serve the Gradio app on.",
    )
    parser.add_argument(
        "--share", action="store_true",
        help="Create a public Gradio share link.",
    )
    parser.add_argument(
        "--load-in-4bit", action="store_true",
        help="Load VLM in 4-bit quantization (requires bitsandbytes).",
    )
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    default_cfg: dict = {}
    if os.path.isfile(args.config):
        default_cfg = load_yaml_config(args.config)

    if args.override and os.path.isfile(args.override):
        overrides = load_yaml_config(args.override)
        default_cfg = _deep_merge(default_cfg, overrides)

    model_name = default_cfg.get("model_name", "Qwen/Qwen3.5-9B")
    cache_dir = default_cfg.get("cache_dir")
    gpu_id = default_cfg.get("gpu_id", args.gpu)
    load_4bit = args.load_in_4bit or default_cfg.get("load_in_4bit", False)
    min_pixels = default_cfg.get("min_pixels", 200704)
    max_pixels = default_cfg.get("max_pixels", 401408)

    _holder.device = "cuda" if torch.cuda.is_available() else "cpu"
    _holder.load(
        model_name, cache_dir,
        gpu_id=gpu_id,
        load_in_4bit=load_4bit,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )

    app = build_ui(default_cfg)
    app.launch(
        server_port=args.port,
        share=args.share,
        server_name="0.0.0.0",
    )


if __name__ == "__main__":
    main()
