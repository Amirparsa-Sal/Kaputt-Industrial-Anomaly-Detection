"""
Qwen VLM Interactive Inference GUI.

A Gradio-based interface for running single-image anomaly classification
using the Qwen3.5 Vision-Language Model. Supports both logit-based scoring
(P(Yes) from next-token logits) and text-based scoring (parsed SCORE: from
generated output). Dataset browsing, prompt editing, and side-by-side
visualization are provided.

Usage:
    python gui/vlm_inference_gui.py \
        --config configs/vlm/vlm_base.yaml
    python gui/vlm_inference_gui.py \
        --config configs/vlm/vlm_base.yaml --gpu 1
    python gui/vlm_inference_gui.py \
        --config configs/vlm/vlm_base.yaml --gpu 0 --port 7863
    python gui/vlm_inference_gui.py \
        --config configs/vlm/vlm_base.yaml \
        --override configs/vlm/vlm_zero_logits.yaml
"""

import argparse
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
# Persistent VLM model holder — loaded once, reused across all inference calls
# ---------------------------------------------------------------------------

@dataclass
class VLMHolder:
    """Keeps the Qwen VLM model and processor in GPU memory across requests."""

    model: object = None
    processor: object = None
    model_name: str = ""
    device: str = "cuda"
    yes_ids: list = None
    no_ids: list = None

    def load(
        self,
        model_name: str,
        cache_dir: str | None,
        gpu_id: int = 0,
        load_in_4bit: bool = False,
        min_pixels: int = 200704,
        max_pixels: int = 401408,
    ):
        """Load model only if it hasn't been loaded or the name changed."""
        if self.model is not None and self.model_name == model_name:
            return

        from transformers import AutoProcessor

        try:
            from transformers import Qwen3_5ForConditionalGeneration
            ModelClass = Qwen3_5ForConditionalGeneration
            print(f"[VLM] Using Qwen3_5ForConditionalGeneration.")
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

        self.yes_ids, self.no_ids = _get_yes_no_token_ids(
            self.processor.tokenizer,
        )
        print(f"[VLM] Yes IDs: {self.yes_ids}, No IDs: {self.no_ids}")

        self.model_name = model_name
        print("[VLM] Model loaded and ready.")

    @property
    def ready(self) -> bool:
        return self.model is not None


_holder = VLMHolder()


# ---------------------------------------------------------------------------
# Token helpers (mirrored from src/vlm/inference.py for standalone usage)
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


def _get_yes_no_token_ids(tokenizer) -> tuple[list[int], list[int]]:
    """Collect token IDs for common Yes/No surface forms."""
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


# ---------------------------------------------------------------------------
# Dataset browser state
# ---------------------------------------------------------------------------

@dataclass
class BrowserState:
    """Mutable state shared by the Gradio callbacks."""

    df: pd.DataFrame | None = None
    current_idx: int = 0
    data_dir: str = ""
    last_filter_key: str = ""


_state = BrowserState()


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_yaml_config(path: str) -> dict:
    """Load a YAML config file and return it as a dict."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a YAML mapping, got {type(raw)}")
    return raw


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*."""
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
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
    """Load the parquet file and apply the selected filters."""
    parquet_path = os.path.join(data_dir, f"query-{split}.parquet")
    if not os.path.isfile(parquet_path):
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)

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


def _filter_key(data_dir, split, is_defect, major_defect, defect_type, defect_class):
    return f"{data_dir}|{split}|{is_defect}|{major_defect}|{defect_type}|{defect_class}"


# ---------------------------------------------------------------------------
# Image preparation
# ---------------------------------------------------------------------------

def prepare_image(
    image_path: str,
    base_dir: str,
    apply_mask: bool,
    input_size: int | None = None,
) -> Image.Image:
    """Load an image, optionally mask out background, and resize."""
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


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def render_plain_image(pil_image: Image.Image, title: str = "") -> np.ndarray:
    """Render a PIL image with a title bar via matplotlib."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(pil_image)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.axis("off")
    fig.tight_layout(pad=0.5)

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return buf


def _render_score_overlay(
    pil_image: Image.Image,
    score: float,
    mode: str,
    title: str = "",
) -> np.ndarray:
    """Render the image with a large anomaly score overlay and a color bar."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(pil_image)

    color = _score_color(score)
    label = f"Anomaly Score: {score:.4f}"

    ax.text(
        0.5, 0.04, label,
        transform=ax.transAxes,
        ha="center", va="bottom",
        fontsize=22, fontweight="bold",
        color="white",
        bbox=dict(
            facecolor=color, alpha=0.85,
            boxstyle="round,pad=0.5",
            edgecolor="white", linewidth=2,
        ),
    )

    mode_label = "LOGITS" if mode == "logits" else "TEXT"
    ax.text(
        0.98, 0.98, mode_label,
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=11, fontweight="bold",
        color="white",
        bbox=dict(facecolor="#444444", alpha=0.7, boxstyle="round,pad=0.3"),
    )

    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.axis("off")
    fig.tight_layout(pad=0.5)

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return buf


def _score_color(score: float) -> str:
    """Map anomaly score [0, 1] to a green-yellow-red color hex string."""
    if score < 0.3:
        return "#00b894"
    elif score < 0.6:
        return "#fdcb6e"
    else:
        return "#d63031"


# ---------------------------------------------------------------------------
# VLM inference — logit mode
# ---------------------------------------------------------------------------

def _build_messages(
    image: Image.Image,
    system_prompt: str,
    user_prompt: str,
) -> list[dict]:
    """Build Qwen-style chat messages with an image attachment."""
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


def _apply_chat_template(processor, messages):
    """Tokenize chat messages via the processor's template."""
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


def score_from_logits(
    image: Image.Image,
    system_prompt: str,
    user_prompt: str,
) -> tuple[float, str]:
    """
    Compute anomaly probability from next-token logit scores.

    Returns (score, detail_string) where score is P(Yes) ∈ [0, 1].
    """
    messages = _build_messages(image, system_prompt, user_prompt)
    inputs = _apply_chat_template(_holder.processor, messages)
    inputs = {k: v.to(_holder.model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = _holder.model(**inputs)

    next_logits = outputs.logits[:, -1, :]

    yes_logit = next_logits[:, _holder.yes_ids].max(dim=-1).values
    no_logit = next_logits[:, _holder.no_ids].max(dim=-1).values
    probs = torch.softmax(
        torch.stack([yes_logit, no_logit], dim=-1), dim=-1,
    )

    p_yes = probs[0, 0].item()
    p_no = probs[0, 1].item()

    detail = (
        f"P(Yes) = {p_yes:.6f}\n"
        f"P(No)  = {p_no:.6f}\n"
        f"Yes logit = {yes_logit.item():.4f}\n"
        f"No  logit = {no_logit.item():.4f}"
    )

    del inputs, outputs
    torch.cuda.empty_cache()

    return p_yes, detail


# ---------------------------------------------------------------------------
# VLM inference — text mode
# ---------------------------------------------------------------------------

def score_from_text(
    image: Image.Image,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    max_new_tokens: int = 256,
) -> tuple[float, str, str]:
    """
    Generate text and parse a numeric confidence score in [0, 1].

    Returns (score, raw_reply, detail_string).
    """
    _ensure_pad_token(_holder.processor, _holder.model)

    messages = _build_messages(image, system_prompt, user_prompt)
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

    match = re.search(
        r"SCORE:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))", reply, re.IGNORECASE,
    )
    if match:
        score = float(match.group(1))
        score = min(max(score, 0.0), 1.0)
        detail = f"Parsed SCORE: {score:.4f} from model output"
        return score, reply, detail

    numeric_match = re.search(r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))", reply)
    if numeric_match:
        score = float(numeric_match.group(1))
        score = min(max(score, 0.0), 1.0)
        detail = f"Extracted numeric value: {score:.4f} (no SCORE: prefix found)"
        return score, reply, detail

    detail = "No numeric score found in model output — defaulting to 0.5"
    return 0.5, reply, detail


# ---------------------------------------------------------------------------
# Core GUI callbacks
# ---------------------------------------------------------------------------

def apply_filters(data_dir, split, is_defect, major_defect, defect_type, defect_class):
    """Load and filter the dataset, update the browser state."""
    if not data_dir or not os.path.isdir(data_dir):
        return "Invalid data directory.", gr.update(), gr.update(), None, None, ""

    key = _filter_key(data_dir, split, is_defect, major_defect, defect_type, defect_class)
    try:
        df = load_and_filter(data_dir, split, is_defect, major_defect, defect_type, defect_class)
    except Exception as e:
        return f"Error: {e}", gr.update(), gr.update(), None, None, ""

    _state.df = df
    _state.data_dir = data_dir
    _state.current_idx = 0
    _state.last_filter_key = key

    n = len(df)
    status = f"Loaded **{n}** images (split: {split})"

    if n == 0:
        return status, gr.update(minimum=0, maximum=1, value=0), gr.update(), None, None, ""

    slider_update = gr.update(minimum=0, maximum=n - 1, value=0, step=1)
    img_original, img_processed, info = _get_current_images(False, False, None)
    return status, slider_update, gr.update(value=f"1 / {n}"), img_original, img_processed, info


def navigate(idx, crop, mask, input_size_str):
    """Navigate to a specific image index."""
    if _state.df is None or len(_state.df) == 0:
        return None, None, "", gr.update()

    idx = int(idx)
    n = len(_state.df)
    idx = max(0, min(idx, n - 1))
    _state.current_idx = idx

    input_size = _parse_input_size(input_size_str)
    img_original, img_processed, info = _get_current_images(crop, mask, input_size)
    counter = f"{idx + 1} / {n}"
    return img_original, img_processed, info, gr.update(value=counter)


def go_prev(crop, mask, input_size_str, current_slider):
    """Move to the previous image."""
    if _state.df is None or len(_state.df) == 0:
        return None, None, "", gr.update(), gr.update()

    idx = max(0, int(current_slider) - 1)
    _state.current_idx = idx
    input_size = _parse_input_size(input_size_str)
    img_original, img_processed, info = _get_current_images(crop, mask, input_size)
    n = len(_state.df)
    counter = f"{idx + 1} / {n}"
    return img_original, img_processed, info, gr.update(value=idx), gr.update(value=counter)


def go_next(crop, mask, input_size_str, current_slider):
    """Move to the next image."""
    if _state.df is None or len(_state.df) == 0:
        return None, None, "", gr.update(), gr.update()

    n = len(_state.df)
    idx = min(n - 1, int(current_slider) + 1)
    _state.current_idx = idx
    input_size = _parse_input_size(input_size_str)
    img_original, img_processed, info = _get_current_images(crop, mask, input_size)
    counter = f"{idx + 1} / {n}"
    return img_original, img_processed, info, gr.update(value=idx), gr.update(value=counter)


def run_single_inference(
    scoring_mode,
    system_prompt,
    user_prompt,
    crop,
    mask,
    input_size_str,
    temperature,
    max_new_tokens,
    current_slider,
):
    """Run VLM inference on the currently displayed image."""
    if _state.df is None or len(_state.df) == 0:
        return None, "No dataset loaded."

    if not _holder.ready:
        return None, "Model not loaded. Wait for startup to finish."

    idx = int(current_slider)
    row = _state.df.iloc[idx]

    col = "query_crop" if crop else "query_image"
    image_path = row[col]
    input_size = _parse_input_size(input_size_str)
    pil_img = prepare_image(image_path, _state.data_dir, mask, input_size)

    t0 = time.time()

    if scoring_mode == "logits":
        score, detail = score_from_logits(pil_img, system_prompt, user_prompt)
        raw_reply = None
    else:
        score, raw_reply, detail = score_from_text(
            pil_img, system_prompt, user_prompt,
            temperature=temperature,
            max_new_tokens=int(max_new_tokens),
        )

    elapsed_ms = (time.time() - t0) * 1000

    result_img = _render_score_overlay(
        pil_img, score, scoring_mode,
        title=f"VLM Anomaly Score — Image {idx + 1}",
    )

    is_defect = bool(row.get("defect", False))
    gt_label = "DEFECT" if is_defect else "NORMAL"

    details_md = f"### VLM Inference Result\n\n"
    details_md += f"**Scoring Mode:** {scoring_mode.upper()}\n\n"
    details_md += f"**Anomaly Score:** `{score:.6f}`\n\n"
    details_md += f"**Ground Truth:** {gt_label}\n\n"
    details_md += f"**Inference Time:** {elapsed_ms:.0f} ms\n\n"
    details_md += f"---\n\n"
    details_md += f"**Score Details:**\n```\n{detail}\n```\n\n"

    if raw_reply is not None:
        details_md += f"---\n\n**Model Output:**\n```\n{raw_reply}\n```\n"

    return result_img, details_md


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_input_size(val: str) -> int | None:
    if not val or val.strip().lower() in ("none", "null", ""):
        return None
    try:
        return int(val.strip())
    except ValueError:
        return None


def _get_current_images(crop: bool, mask: bool, input_size: int | None):
    """Return (original_render, processed_render, info_markdown) for current index."""
    if _state.df is None or len(_state.df) == 0:
        return None, None, ""

    idx = _state.current_idx
    row = _state.df.iloc[idx]

    orig_path = row.get("query_image", "")
    full_orig = os.path.join(_state.data_dir, orig_path)

    is_defect = bool(row.get("defect", False))
    defect_types = str(row.get("defect_types", ""))
    is_major = bool(row.get("major_defect", False))

    info = f"### Image {idx + 1}\n\n"
    info += f"**Path:** `{orig_path}`\n\n"
    info += f"**Defect:** {'Yes' if is_defect else 'No'}"
    if is_defect:
        info += f" | **Type:** {defect_types} | **Major:** {'Yes' if is_major else 'No'}"
    info += "\n"

    try:
        original_img = Image.open(full_orig).convert("RGB")
        img_original = render_plain_image(original_img, "Original Image")
    except Exception:
        img_original = None

    col = "query_crop" if crop else "query_image"
    proc_path = row.get(col, orig_path)
    try:
        processed_img = prepare_image(proc_path, _state.data_dir, mask, input_size)
        label_parts = []
        if crop:
            label_parts.append("Cropped")
        if mask:
            label_parts.append("Masked")
        if input_size:
            label_parts.append(f"Resized to {input_size}")
        proc_title = " + ".join(label_parts) if label_parts else "Original (no processing)"
        img_processed = render_plain_image(processed_img, proc_title)
    except Exception:
        img_processed = None

    return img_original, img_processed, info


def _on_scoring_mode_change(scoring_mode, cfg_store):
    """Swap the system and user prompts when the scoring mode changes."""
    cfg = cfg_store or {}

    if scoring_mode == "logits":
        sys_prompt = cfg.get(
            "system_prompt_logits_zero_shot",
            "You are a quality inspector. Answer questions with only 'Yes' or 'No'.",
        )
        usr_prompt = cfg.get("user_prompt_logits_zero_shot", "")
        return (
            gr.update(value=sys_prompt),
            gr.update(value=usr_prompt),
            gr.update(visible=False),
            gr.update(visible=False),
        )
    else:
        sys_prompt = cfg.get(
            "system_prompt_text_zero_shot",
            "You are an expert quality control inspector analyzing products for defects.",
        )
        usr_prompt = cfg.get("user_prompt_text_zero_shot", "")
        return (
            gr.update(value=sys_prompt),
            gr.update(value=usr_prompt),
            gr.update(visible=True),
            gr.update(visible=True),
        )


# ---------------------------------------------------------------------------
# Gradio UI definition
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
.main-title {
    text-align: center;
    background: linear-gradient(135deg, #2d1b69 0%, #11998e 100%);
    color: white !important;
    padding: 20px 30px;
    border-radius: 12px;
    margin-bottom: 16px;
    font-size: 1.5em;
}
.main-title h1 { color: white !important; margin: 0; }
.main-title p { color: #c4f0e8 !important; margin: 4px 0 0 0; font-size: 0.65em; }
.filter-panel {
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    padding: 12px;
}
.nav-btn { min-width: 100px !important; }
.run-btn {
    background: linear-gradient(135deg, #6c5ce7, #a29bfe) !important;
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
    """Construct and return the Gradio Blocks application."""

    scoring_mode_init = default_cfg.get("scoring_mode", "logits")

    if scoring_mode_init == "logits":
        sys_prompt_init = default_cfg.get(
            "system_prompt_logits_zero_shot",
            "You are a quality inspector. Answer questions with only 'Yes' or 'No'.",
        )
        usr_prompt_init = default_cfg.get(
            "user_prompt_logits_zero_shot", "",
        )
    else:
        sys_prompt_init = default_cfg.get(
            "system_prompt_text_zero_shot",
            "You are an expert quality control inspector analyzing products for defects.",
        )
        usr_prompt_init = default_cfg.get(
            "user_prompt_text_zero_shot", "",
        )

    with gr.Blocks(
        title="Qwen VLM Inference Explorer",
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(
            primary_hue="violet",
            secondary_hue="teal",
            neutral_hue="slate",
        ),
    ) as app:

        # Hidden state to hold full config for prompt switching
        cfg_state = gr.State(default_cfg)

        # ---- Header ----
        gr.HTML(
            '<div class="main-title">'
            "<h1>Qwen VLM Inference Explorer</h1>"
            "<p>Interactive single-image anomaly classification — Logits &amp; Text scoring modes</p>"
            "</div>"
        )

        with gr.Row():
            # ================================================================
            # LEFT COLUMN — Controls
            # ================================================================
            with gr.Column(scale=1, min_width=380):

                # -- Data Filters --
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

                    filter_btn = gr.Button(
                        "Apply Filters & Load Dataset",
                        variant="primary",
                        size="lg",
                    )
                    filter_status = gr.Markdown("")

                # -- Image Processing --
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

                # -- Inference Settings --
                with gr.Group():
                    gr.Markdown("### VLM Inference")

                    scoring_mode = gr.Radio(
                        label="Scoring Mode",
                        choices=["logits", "text"],
                        value=scoring_mode_init,
                        info=(
                            "logits: P(Yes) from next-token logits (fast, no generation). "
                            "text: generate response and parse SCORE: value."
                        ),
                    )

                    system_prompt = gr.Textbox(
                        label="System Prompt",
                        value=sys_prompt_init,
                        lines=4,
                        elem_classes=["prompt-editor"],
                    )
                    user_prompt = gr.Textbox(
                        label="User Prompt",
                        value=usr_prompt_init,
                        lines=10,
                        elem_classes=["prompt-editor"],
                    )

                    temperature = gr.Slider(
                        label="Temperature (text mode)",
                        minimum=0.0, maximum=1.5, step=0.05,
                        value=default_cfg.get("temperature", 0.0),
                        visible=(scoring_mode_init == "text"),
                    )
                    max_new_tokens = gr.Slider(
                        label="Max New Tokens (text mode)",
                        minimum=32, maximum=1024, step=32,
                        value=default_cfg.get("max_new_tokens", 256),
                        visible=(scoring_mode_init == "text"),
                    )

                    run_btn = gr.Button(
                        "Run VLM Inference",
                        variant="primary",
                        size="lg",
                        elem_classes=["run-btn"],
                    )

            # ================================================================
            # RIGHT COLUMN — Image display & results
            # ================================================================
            with gr.Column(scale=2):

                # -- Navigation --
                with gr.Row():
                    prev_btn = gr.Button("◀  Previous", size="sm", elem_classes=["nav-btn"])
                    counter = gr.Textbox(
                        value="0 / 0",
                        show_label=False,
                        interactive=False,
                        elem_classes=["counter-display"],
                        text_align="center",
                    )
                    next_btn = gr.Button("Next  ▶", size="sm", elem_classes=["nav-btn"])

                image_slider = gr.Slider(
                    label="Image Index",
                    minimum=0, maximum=1, step=1, value=0,
                )

                # -- Images --
                with gr.Row():
                    img_original = gr.Image(
                        label="Original Image",
                        type="numpy",
                        interactive=False,
                    )
                    img_processed = gr.Image(
                        label="Processed / Result",
                        type="numpy",
                        interactive=False,
                    )

                # -- Info & Results --
                image_info = gr.Markdown("")
                inference_output = gr.Markdown("")

        # ================================================================
        # Event wiring
        # ================================================================

        filter_btn.click(
            fn=apply_filters,
            inputs=[data_dir, split, is_defect, major_defect, defect_type, defect_class],
            outputs=[filter_status, image_slider, counter, img_original, img_processed, image_info],
        )

        image_slider.change(
            fn=navigate,
            inputs=[image_slider, crop, mask, input_size],
            outputs=[img_original, img_processed, image_info, counter],
        )

        prev_btn.click(
            fn=go_prev,
            inputs=[crop, mask, input_size, image_slider],
            outputs=[img_original, img_processed, image_info, image_slider, counter],
        )

        next_btn.click(
            fn=go_next,
            inputs=[crop, mask, input_size, image_slider],
            outputs=[img_original, img_processed, image_info, image_slider, counter],
        )

        scoring_mode.change(
            fn=_on_scoring_mode_change,
            inputs=[scoring_mode, cfg_state],
            outputs=[system_prompt, user_prompt, temperature, max_new_tokens],
        )

        run_btn.click(
            fn=run_single_inference,
            inputs=[
                scoring_mode, system_prompt, user_prompt,
                crop, mask, input_size,
                temperature, max_new_tokens,
                image_slider,
            ],
            outputs=[img_processed, inference_output],
        )

    return app


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Qwen VLM Interactive Inference GUI",
    )
    parser.add_argument(
        "--config", type=str, default="configs/vlm/vlm_base.yaml",
        help="Path to the base YAML config file for defaults.",
    )
    parser.add_argument(
        "--override", type=str, default=None,
        help=(
            "Path to an override YAML config (e.g. vlm_zero_logits.yaml). "
            "Values in this file take precedence over the base config."
        ),
    )
    parser.add_argument(
        "--gpu", type=int, default=0,
        help="GPU device index to use (default: 0).",
    )
    parser.add_argument(
        "--port", type=int, default=7863,
        help="Port to serve the Gradio app on.",
    )
    parser.add_argument(
        "--share", action="store_true",
        help="Create a public Gradio share link.",
    )
    parser.add_argument(
        "--load-in-4bit", action="store_true",
        help="Load VLM in 4-bit quantization to save VRAM (requires bitsandbytes).",
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
