"""
SAM3 Interactive Inference GUI.

A modern Gradio-based interface for running single-image inference on the
SAM3 defect detection model. Supports dataset filtering, prompt customization,
image navigation, and side-by-side result visualization with bounding boxes.

Usage:
    python gui/inference_gui.py --config configs/sam3/base.yaml
    python gui/inference_gui.py --config configs/sam3/base.yaml --gpu 1
    python gui/inference_gui.py --config configs/sam3/base.yaml --gpu 0 --port 7861
"""

import argparse
import os
from dataclasses import dataclass

import gradio as gr
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image

matplotlib.use("Agg")

LOGICAL_DEFECT_TYPES = ["missing_unit", "actuation"]

STRUCTURAL_DEFECT_TYPES = [
    "deformation",
    "deconstruction",
    "penetration",
    "superficial",
    "spillage",
]
ALL_DEFECT_CLASSES = STRUCTURAL_DEFECT_TYPES + LOGICAL_DEFECT_TYPES


# ---------------------------------------------------------------------------
# Persistent model holder — loaded once, reused across all inference calls
# ---------------------------------------------------------------------------

@dataclass
class ModelHolder:
    """Keeps the SAM3 model and processor in GPU memory across requests."""

    model: object = None
    processor: object = None
    model_name: str = ""
    device: str = "cuda"

    def load(self, model_name: str, cache_dir: str | None, hf_token: str | None):
        """Load model only if it hasn't been loaded or the name changed."""
        if self.model is not None and self.model_name == model_name:
            return

        from huggingface_hub import login as hf_login
        from transformers import Sam3Model, Sam3Processor

        if hf_token:
            hf_login(token=hf_token)

        print(f"Loading model '{model_name}' on {self.device} ...")
        self.model = Sam3Model.from_pretrained(
            model_name, cache_dir=cache_dir
        ).to(self.device)
        self.processor = Sam3Processor.from_pretrained(
            model_name, cache_dir=cache_dir
        )
        self.model_name = model_name
        print("Model loaded.")

    @property
    def ready(self) -> bool:
        return self.model is not None


_holder = ModelHolder()


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
    prompts_map: dict | None = None


_state = BrowserState()


# ---------------------------------------------------------------------------
# Config loading (reuse logic from run_inference.py)
# ---------------------------------------------------------------------------

def load_yaml_config(path: str) -> dict:
    """Load a YAML config file and return it as a dict."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a YAML mapping, got {type(raw)}")
    return raw


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
    """
    Load the parquet file and apply the selected filters.

    Returns the filtered DataFrame (reset index).
    """
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
# Image preparation (mirrors run_inference.py)
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

def render_image_with_boxes(
    pil_image: Image.Image,
    boxes: list[list[float]],
    scores: list[float],
    title: str = "",
) -> np.ndarray:
    """
    Render a PIL image with bounding box overlays using matplotlib
    and return it as a numpy RGB array suitable for Gradio display.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(pil_image)

    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box
        rect = mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            fill=False, edgecolor="#00ff88", linewidth=2.5,
        )
        ax.add_patch(rect)
        ax.text(
            x1, max(y1 - 6, 0),
            f"{score:.3f}",
            color="white", fontsize=10, fontweight="bold",
            bbox=dict(facecolor="#222222", alpha=0.8, pad=2, edgecolor="#00ff88"),
        )

    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.axis("off")
    fig.tight_layout(pad=0.5)

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return buf


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
        return status, gr.update(maximum=0, value=0), gr.update(), None, None, ""

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


def suggest_prompt(current_slider):
    """Auto-fill the prompt field based on the current image's defect type and config prompts."""
    if _state.df is None or len(_state.df) == 0:
        return gr.update()

    idx = int(current_slider)
    row = _state.df.iloc[idx]
    defect_types_raw = str(row.get("defect_types", ""))
    primary = defect_types_raw.split(",")[0].strip() if defect_types_raw else ""

    if not primary or not bool(row.get("defect", False)):
        if _state.prompts_map:
            all_prompts = []
            for p_list in _state.prompts_map.values():
                if isinstance(p_list, list):
                    all_prompts.extend(p_list)
            return gr.update(value="; ".join(all_prompts) if all_prompts else "defect")
        return gr.update(value="defect")

    if _state.prompts_map and primary in _state.prompts_map:
        vals = _state.prompts_map[primary]
        if isinstance(vals, list):
            return gr.update(value="; ".join(vals))

    return gr.update(value=primary)


def run_single_inference(
    prompt_text, crop, mask, input_size_str,
    threshold, mask_threshold, current_slider,
):
    """Run SAM3 inference on the currently displayed image."""
    if _state.df is None or len(_state.df) == 0:
        return None, "No dataset loaded.", ""

    if not _holder.ready:
        return None, "Model not loaded. Wait for startup to finish.", ""

    idx = int(current_slider)
    row = _state.df.iloc[idx]

    col = "query_crop" if crop else "query_image"
    image_path = row[col]
    input_size = _parse_input_size(input_size_str)
    pil_img = prepare_image(image_path, _state.data_dir, mask, input_size)

    prompts = [p.strip() for p in prompt_text.split(";") if p.strip()]
    if not prompts:
        return None, "Please enter at least one prompt (separate multiples with `;`).", ""

    all_boxes = []
    all_scores = []
    prompt_details = []

    for prompt in prompts:
        inputs = _holder.processor(
            images=[pil_img],
            text=[prompt],
            input_boxes=None,
            input_boxes_labels=None,
            return_tensors="pt",
        ).to(_holder.device)

        with torch.no_grad():
            outputs = _holder.model(**inputs)

        results = _holder.processor.post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            mask_threshold=mask_threshold,
            target_sizes=inputs.get("original_sizes").tolist(),
        )

        result = results[0]
        if len(result["scores"]) > 0:
            boxes = result["boxes"].detach().float().cpu().tolist()
            scores = result["scores"].detach().float().cpu().tolist()
            all_boxes.extend(boxes)
            all_scores.extend(scores)
            max_score = max(scores)
            prompt_details.append(f'  **"{prompt}"** → {len(boxes)} detection(s), max score: {max_score:.4f}')
        else:
            prompt_details.append(f'  **"{prompt}"** → no detections')

    title = f"Detections ({len(all_boxes)} total)"
    result_img = render_image_with_boxes(pil_img, all_boxes, all_scores, title)

    details_md = "### Inference Results\n\n"
    details_md += "\n\n".join(prompt_details)
    details_md += f"\n\n---\n**Total detections:** {len(all_boxes)}"
    if all_scores:
        details_md += f" | **Max score:** {max(all_scores):.4f}"

    return result_img, details_md, ""


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


# ---------------------------------------------------------------------------
# Gradio UI definition
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
.main-title {
    text-align: center;
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    color: white !important;
    padding: 20px 30px;
    border-radius: 12px;
    margin-bottom: 16px;
    font-size: 1.5em;
}
.main-title h1 { color: white !important; margin: 0; }
.main-title p { color: #a0c4ff !important; margin: 4px 0 0 0; font-size: 0.65em; }
.filter-panel {
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    padding: 12px;
}
.nav-btn { min-width: 100px !important; }
.run-btn {
    background: linear-gradient(135deg, #00b894, #00cec9) !important;
    color: white !important;
    font-weight: bold !important;
    font-size: 1.1em !important;
    border: none !important;
}
.counter-display {
    text-align: center;
    font-size: 1.2em;
    font-weight: bold;
    padding: 8px;
}
"""


def build_ui(default_cfg: dict) -> gr.Blocks:
    """Construct and return the Gradio Blocks application."""

    with gr.Blocks(
        title="SAM3 Inference Explorer",
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(
            primary_hue="teal",
            secondary_hue="blue",
            neutral_hue="slate",
        ),
    ) as app:

        # ---- Header ----
        gr.HTML(
            '<div class="main-title">'
            "<h1>SAM3 Inference Explorer</h1>"
            "<p>Interactive single-image defect detection with bounding box visualization</p>"
            "</div>"
        )

        with gr.Row():
            # ================================================================
            # LEFT COLUMN — Controls
            # ================================================================
            with gr.Column(scale=1, min_width=340):

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
                    defect_type = gr.Dropdown(
                        label="Defect Type",
                        choices=["any", "structural", "logical"],
                        value=default_cfg.get("defect_type", "any"),
                    )
                    defect_class = gr.Dropdown(
                        label="Defect Class (exact)",
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
                    crop = gr.Checkbox(
                        label="Use Crop",
                        value=default_cfg.get("crop", False),
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
                    gr.Markdown("### Inference")
                    prompt_input = gr.Textbox(
                        label="Prompt(s)",
                        value=_default_prompt_string(default_cfg),
                        placeholder="damaged book; crushed box",
                        lines=2,
                        info="Separate multiple prompts with semicolons (;)",
                    )
                    suggest_btn = gr.Button(
                        "Auto-Suggest from Defect Type",
                        variant="secondary",
                        size="sm",
                    )
                    threshold = gr.Slider(
                        label="Detection Threshold",
                        minimum=0.0, maximum=1.0, step=0.05,
                        value=default_cfg.get("threshold", 0.5),
                    )
                    mask_threshold = gr.Slider(
                        label="Mask Threshold",
                        minimum=0.0, maximum=1.0, step=0.05,
                        value=default_cfg.get("mask_threshold", 0.5),
                    )
                    run_btn = gr.Button(
                        "Run Inference",
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
                    minimum=0, maximum=0, step=1, value=0,
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
                inference_error = gr.Textbox(visible=False)

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

        suggest_btn.click(
            fn=suggest_prompt,
            inputs=[image_slider],
            outputs=[prompt_input],
        )

        run_btn.click(
            fn=run_single_inference,
            inputs=[
                prompt_input, crop, mask, input_size,
                threshold, mask_threshold, image_slider,
            ],
            outputs=[img_processed, inference_output, inference_error],
        )

    return app


def _default_prompt_string(cfg: dict) -> str:
    """Build a sensible default prompt string from config prompts mapping."""
    prompts = cfg.get("prompts")
    if not prompts or not isinstance(prompts, dict):
        return "defect"
    first_key = next(iter(prompts))
    first_vals = prompts[first_key]
    if isinstance(first_vals, list):
        return "; ".join(first_vals)
    return str(first_vals)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SAM3 Interactive Inference GUI")
    parser.add_argument(
        "--config", type=str, default="configs/base.yaml",
        help="Path to the YAML config file for defaults.",
    )
    parser.add_argument(
        "--gpu", type=int, default=0,
        help="GPU device index to use (default: 0).",
    )
    parser.add_argument(
        "--port", type=int, default=7860,
        help="Port to serve the Gradio app on.",
    )
    parser.add_argument(
        "--share", action="store_true",
        help="Create a public Gradio share link.",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help=(
            "Hugging Face access token for gated models "
            "(optional; can also rely on ``huggingface-cli login``)."
        ),
    )
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    _holder.device = "cuda" if torch.cuda.is_available() else "cpu"

    default_cfg = {}
    if os.path.isfile(args.config):
        default_cfg = load_yaml_config(args.config)

    model_name = default_cfg.get("model_name", "facebook/sam3")
    cache_dir = default_cfg.get("cache_dir")
    hf_token = args.hf_token
    _holder.load(model_name, cache_dir, hf_token)

    _state.prompts_map = default_cfg.get("prompts")

    app = build_ui(default_cfg)
    app.launch(
        server_port=args.port,
        share=args.share,
        server_name="0.0.0.0",
    )


if __name__ == "__main__":
    main()
