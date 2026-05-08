"""
Agentic SAM3 Interactive Inference GUI.

An agentic approach to defect detection on Kaputt1 using:
  - Qwen2.5-VL as a Multimodal Vision-Language Model (MVLM) for reasoning
  - SAM3 as the open-vocabulary segmentation model

The agent iteratively:
  1. Identifies the object in the image via the MVLM
  2. Reasons about possible defects given Kaputt1 defect types
  3. Generates a SAM3 text prompt and runs segmentation
  4. Verifies the result and refines if needed (up to max_iters)

Usage:
    python gui/agentic_inference_gui.py --config configs/sam3/base.yaml
    python gui/agentic_inference_gui.py --config configs/sam3/base.yaml --gpu 0
    python gui/agentic_inference_gui.py --config configs/sam3/base.yaml --gpu 0 --port 7862
"""

import argparse
import json
import os
import time
from dataclasses import dataclass, field

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

# ---------------------------------------------------------------------------
# Domain constants
# ---------------------------------------------------------------------------

KAPUTT1_DEFECT_DESCRIPTION = (
    "The Kaputt1 dataset defines the following defect types for damaged items:\n"
    "- penetration: holes, tears, cuts in the packaging or product\n"
    "- deformation: dents, crushes, bent or warped surfaces\n"
    "- actuation: inappropriately opened containers (open box/bag/book)\n"
    "- deconstruction: items falling apart, collapsed or disassembled structures\n"
    "- spillage: leaked or spilled contents (liquid, powder, granules)\n"
    "- superficial: surface-level damage like dirt marks, scratches, scuffs\n"
    "- missing_unit: one or more expected units/parts are absent\n\n"
    "An item may exhibit multiple defect types simultaneously at the same or "
    "different spatial locations."
)

STRUCTURAL_DEFECT_TYPES = [
    "deformation", "deconstruction", "penetration", "superficial", "spillage",
]
LOGICAL_DEFECT_TYPES = ["missing_unit", "actuation"]
ALL_DEFECT_CLASSES = STRUCTURAL_DEFECT_TYPES + LOGICAL_DEFECT_TYPES

# ---------------------------------------------------------------------------
# Default prompts for the agentic pipeline (editable in the GUI)
# ---------------------------------------------------------------------------

DEFAULT_OBJECT_DETECTION_PROMPT = (
    "You are a precise visual analysis assistant specialized in product inspection.\n\n"
    "Look at this image carefully. Identify the primary object or product shown.\n"
    "Consider: What type of product/item is this? What is its category? "
    "What material is it made of? What is its expected intact condition?\n\n"
    "Reply with ONLY a JSON object:\n"
    '{"object_type": "<concise name>", "description": "<brief physical description>", '
    '"material": "<primary material>", "category": "<product category>"}\n\n'
    "No explanation, no markdown fences, just the JSON."
)

DEFAULT_DEFECT_ANALYSIS_PROMPT = (
    "You are an expert quality inspector analyzing products for defects.\n\n"
    "You are inspecting a {object_type} ({description}).\n\n"
    f"{KAPUTT1_DEFECT_DESCRIPTION}\n\n"
    "Given this specific object, which defect types from the list above are "
    "most plausible? For each plausible defect, provide a SHORT noun phrase "
    "(2-4 words) that SAM3 (an open-vocabulary segmentation model) can use to "
    "detect the defect visually.\n\n"
    "Prioritize defects that would be VISIBLE in the image. Focus on structural "
    "defects (penetration, deformation, deconstruction, superficial, spillage) "
    "over logical ones (missing_unit, actuation) unless the image clearly shows "
    "an opened container or missing parts.\n\n"
    "Reply with ONLY a JSON object:\n"
    '{{"defect_prompts": [{{"defect_type": "<type>", "sam_prompt": "<short phrase>"}},'
    " ...]}}\n\n"
    "Generate 2-5 prompts, ordered from most to least likely. "
    "No explanation, no markdown fences, just the JSON."
)

DEFAULT_VERIFICATION_PROMPT = (
    "You are a quality control verification assistant.\n\n"
    "A segmentation model was asked to find '{sam_prompt}' in an image of a "
    "{object_type}. The model returned {n_masks} detection(s) with confidence "
    "scores: {scores} and bounding boxes: {boxes}.\n\n"
    "Based on the image and the detection metadata, decide if this is a "
    "valid defect detection:\n"
    "1. Do the detections plausibly correspond to a real defect?\n"
    "2. Are the bounding box locations reasonable for this type of defect?\n"
    "   A box covering the entire image is NOT a valid defect detection.\n"
    "3. Are the confidence scores high enough to be trustworthy (>0.3)?\n\n"
    "Reply with ONLY a JSON object:\n"
    '{{"ok": true/false, "reason": "<brief explanation>"}}\n\n'
    "No explanation, no markdown fences, just the JSON."
)

DEFAULT_REFINEMENT_PROMPT = (
    "You are a vision assistant helping refine segmentation prompts for defect "
    "detection.\n\n"
    "We are inspecting a {object_type} for defects. The segmentation model "
    "did not produce a valid result for the prompt: '{failed_prompt}'.\n\n"
    "Previously tried prompts: {tried_prompts}\n\n"
    "Suggest a DIFFERENT noun phrase (2-4 words) that might detect a visible "
    "defect in this specific object. Think about:\n"
    "- Using more generic terms (e.g., 'damage' instead of 'crushed corner')\n"
    "- Describing the visual appearance rather than the defect name\n"
    "- Trying a different defect type entirely\n\n"
    "Do NOT repeat any previously tried prompt.\n\n"
    "Reply with ONLY a JSON object:\n"
    '{{"sam_prompt": "<short 2-4 word noun phrase>"}}\n\n'
    "No explanation, no markdown fences, just the JSON."
)

# ---------------------------------------------------------------------------
# Model holders — loaded once, reused across requests
# ---------------------------------------------------------------------------


@dataclass
class SAM3Holder:
    """Keeps the SAM3 model and processor in GPU memory."""

    model: object = None
    processor: object = None
    model_name: str = ""
    device: str = "cuda"

    def load(self, model_name: str, cache_dir: str | None, hf_token: str | None):
        if self.model is not None and self.model_name == model_name:
            return

        from huggingface_hub import login as hf_login
        from transformers import Sam3Model, Sam3Processor

        if hf_token:
            hf_login(token=hf_token)

        print(f"[SAM3] Loading '{model_name}' on {self.device} ...")
        self.processor = Sam3Processor.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = Sam3Model.from_pretrained(
            model_name, cache_dir=cache_dir,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self.model.eval()
        self.model_name = model_name
        print("[SAM3] Model loaded.")

    @property
    def ready(self) -> bool:
        return self.model is not None


@dataclass
class VLMHolder:
    """Keeps the Qwen2.5-VL model and processor in GPU memory."""

    model: object = None
    processor: object = None
    model_name: str = ""
    device: str = "cuda"

    def load(self, model_name: str, cache_dir: str | None, load_in_4bit: bool = False):
        if self.model is not None and self.model_name == model_name:
            return

        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        print(f"[VLM] Loading '{model_name}' ...")
        self.processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True, cache_dir=cache_dir,
            min_pixels=256 * 28 * 28,
            max_pixels=512 * 28 * 28,
        )

        model_kwargs = dict(
            device_map="auto",
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
            print("[VLM] Using 4-bit quantization to save ~8GB VRAM.")

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, **model_kwargs,
        )
        self.model.eval()
        self.model_name = model_name
        print("[VLM] Model loaded.")

    @property
    def ready(self) -> bool:
        return self.model is not None


_sam = SAM3Holder()
_vlm = VLMHolder()


# ---------------------------------------------------------------------------
# Browser state (shared across Gradio callbacks)
# ---------------------------------------------------------------------------

@dataclass
class BrowserState:
    df: pd.DataFrame | None = None
    current_idx: int = 0
    data_dir: str = ""
    last_filter_key: str = ""


_state = BrowserState()


# ---------------------------------------------------------------------------
# Agent trace — records each step for visualization
# ---------------------------------------------------------------------------

@dataclass
class AgentStep:
    """One step in the agentic reasoning trace."""

    round_idx: int
    action: str
    prompt_used: str = ""
    vlm_raw_reply: str = ""
    parsed_result: dict = field(default_factory=dict)
    sam_n_masks: int = 0
    sam_max_score: float = 0.0
    sam_boxes: list = field(default_factory=list)
    sam_scores: list = field(default_factory=list)
    result_image: np.ndarray | None = None
    elapsed_ms: float = 0.0


# ---------------------------------------------------------------------------
# Core VLM inference
# ---------------------------------------------------------------------------

def vlm_generate(image: Image.Image, messages: list, max_new_tokens: int = 512) -> str:
    """
    Run Qwen2.5-VL inference with chat-style messages.
    Returns the generated text reply.
    """
    torch.cuda.empty_cache()

    text_input = _vlm.processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = _vlm.processor(
        text=[text_input], images=[image], return_tensors="pt",
    )
    inputs = {k: v.to(_vlm.model.device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        generated_ids = _vlm.model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
        )

    new_tokens = generated_ids[0][input_len:]
    reply = _vlm.processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    del inputs, generated_ids, new_tokens
    torch.cuda.empty_cache()

    return reply


# ---------------------------------------------------------------------------
# Core SAM3 inference
# ---------------------------------------------------------------------------

def call_sam(
    image: Image.Image,
    text_prompt: str,
    threshold: float = 0.3,
    mask_threshold: float = 0.5,
) -> dict:
    """
    Run SAM3 segmentation with a text prompt.
    Returns dict with keys: masks (np), boxes (np), scores (np).
    """
    torch.cuda.empty_cache()

    inputs = _sam.processor(
        images=image, text=text_prompt, return_tensors="pt",
    ).to(_sam.device)

    with torch.no_grad():
        outputs = _sam.model(**inputs)

    results = _sam.processor.post_process_instance_segmentation(
        outputs,
        threshold=threshold,
        mask_threshold=mask_threshold,
        target_sizes=inputs.get("original_sizes").tolist(),
    )[0]

    n = len(results["scores"])
    if n == 0:
        result = {"masks": np.array([]), "boxes": np.array([]), "scores": np.array([])}
    else:
        result = {
            "masks": results["masks"].cpu().numpy(),
            "boxes": results["boxes"].cpu().to(torch.float32).numpy(),
            "scores": results["scores"].cpu().to(torch.float32).numpy(),
        }

    del inputs, outputs, results
    torch.cuda.empty_cache()

    return result


# ---------------------------------------------------------------------------
# JSON parsing helper
# ---------------------------------------------------------------------------

def _parse_json(raw: str) -> dict | None:
    """Attempt to extract a JSON object from a VLM reply."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.lstrip("`").lstrip("json").lstrip("\n")
        cleaned = cleaned.rstrip("`").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start:end])
            except json.JSONDecodeError:
                return None
    return None


# ---------------------------------------------------------------------------
# Agentic inference pipeline
# ---------------------------------------------------------------------------

def run_agentic_inference(
    pil_image: Image.Image,
    max_iters: int,
    seg_threshold: float,
    mask_threshold: float,
    object_detection_prompt: str,
    defect_analysis_prompt: str,
    verification_prompt: str,
    refinement_prompt: str,
) -> tuple[list[AgentStep], float | None, dict | None]:
    """
    Run the full agentic pipeline on a single image.

    Returns:
        steps: list of AgentStep objects (the full reasoning trace)
        presence_score: the maximum SAM3 score across all successful detections
        best_result: the SAM3 result dict for the best detection
    """
    steps: list[AgentStep] = []
    tried_prompts: list[str] = []
    best_score = 0.0
    best_result = None
    best_prompt = ""

    # ── Step 1: Object detection ──────────────────────────────────────────
    t0 = time.time()
    obj_messages = [
        {"role": "system", "content": object_detection_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": "Identify the object in this image."},
            ],
        },
    ]
    vlm_reply = vlm_generate(pil_image, obj_messages)
    elapsed = (time.time() - t0) * 1000

    parsed = _parse_json(vlm_reply)
    object_type = parsed.get("object_type", "unknown object") if parsed else "unknown object"
    obj_description = parsed.get("description", "") if parsed else ""

    step = AgentStep(
        round_idx=0,
        action="Object Detection",
        vlm_raw_reply=vlm_reply,
        parsed_result=parsed or {"raw": vlm_reply},
        elapsed_ms=elapsed,
    )
    steps.append(step)

    # ── Step 2: Defect analysis ───────────────────────────────────────────
    t0 = time.time()
    filled_defect_prompt = defect_analysis_prompt.format(
        object_type=object_type, description=obj_description,
    )
    defect_messages = [
        {"role": "system", "content": filled_defect_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {
                    "type": "text",
                    "text": (
                        f"This is a {object_type}. Analyze what defects could be "
                        "visible in this image and generate SAM3 prompts to detect them."
                    ),
                },
            ],
        },
    ]
    vlm_reply = vlm_generate(pil_image, defect_messages)
    elapsed = (time.time() - t0) * 1000

    parsed = _parse_json(vlm_reply)
    defect_prompts = []
    if parsed and "defect_prompts" in parsed:
        for item in parsed["defect_prompts"]:
            if isinstance(item, dict) and "sam_prompt" in item:
                defect_prompts.append(item)
    if not defect_prompts:
        defect_prompts = [
            {"defect_type": "generic", "sam_prompt": "damage"},
            {"defect_type": "generic", "sam_prompt": "defect"},
        ]

    step = AgentStep(
        round_idx=1,
        action="Defect Analysis",
        vlm_raw_reply=vlm_reply,
        parsed_result=parsed or {"raw": vlm_reply},
        elapsed_ms=elapsed,
    )
    steps.append(step)

    # ── Phase A: Try ALL original defect prompts first ─────────────────────
    # This ensures every defect type from the analysis gets a chance before
    # we spend iterations on refinements.
    iteration = 0
    original_prompts = [dp["sam_prompt"] for dp in defect_prompts]
    refinement_queue: list[str] = []
    verified = False

    for current_prompt in original_prompts:
        if iteration >= max_iters:
            break
        if current_prompt in tried_prompts:
            continue
        tried_prompts.append(current_prompt)
        iteration += 1

        step, sam_result = _try_sam_prompt(
            pil_image, current_prompt, iteration, len(steps),
            seg_threshold, mask_threshold,
        )

        if step.sam_n_masks > 0 and step.sam_max_score > best_score:
            best_score = step.sam_max_score
            best_result = sam_result

        if step.sam_n_masks > 0:
            verified = _verify_result(
                pil_image, step, current_prompt, object_type,
                verification_prompt,
            )
            if verified:
                steps.append(step)
                break
            # Verification failed — ask refinement for a better prompt
            new_prompt = _get_refinement(
                pil_image, step, current_prompt, object_type,
                tried_prompts, refinement_prompt,
            )
            if new_prompt and new_prompt not in tried_prompts:
                refinement_queue.append(new_prompt)
        else:
            new_prompt = _get_refinement(
                pil_image, step, current_prompt, object_type,
                tried_prompts, refinement_prompt,
            )
            if new_prompt and new_prompt not in tried_prompts:
                refinement_queue.append(new_prompt)

        steps.append(step)

    # ── Phase B: Refinement rounds with VLM-suggested prompts ────────────
    # Only reached if no prompt was verified in Phase A.
    while not verified and iteration < max_iters and refinement_queue:
        current_prompt = refinement_queue.pop(0)
        if current_prompt in tried_prompts:
            continue
        tried_prompts.append(current_prompt)
        iteration += 1

        step, sam_result = _try_sam_prompt(
            pil_image, current_prompt, iteration, len(steps),
            seg_threshold, mask_threshold,
        )

        if step.sam_n_masks > 0 and step.sam_max_score > best_score:
            best_score = step.sam_max_score
            best_result = sam_result

        if step.sam_n_masks > 0:
            verified = _verify_result(
                pil_image, step, current_prompt, object_type,
                verification_prompt,
            )
            if verified:
                steps.append(step)
                break
            new_prompt = _get_refinement(
                pil_image, step, current_prompt, object_type,
                tried_prompts, refinement_prompt,
            )
            if new_prompt and new_prompt not in tried_prompts:
                refinement_queue.append(new_prompt)
        else:
            new_prompt = _get_refinement(
                pil_image, step, current_prompt, object_type,
                tried_prompts, refinement_prompt,
            )
            if new_prompt and new_prompt not in tried_prompts:
                refinement_queue.append(new_prompt)

        steps.append(step)

    presence_score = best_score if best_score > 0 else None
    return steps, presence_score, best_result


def _is_valid_sam_prompt(prompt: str) -> bool:
    """Reject VLM outputs that are sentences/instructions instead of noun phrases."""
    if not prompt or len(prompt) > 60:
        return False
    if any(w in prompt.lower() for w in ("should", "must", "need", "try", "focus on", "instead")):
        return False
    if prompt.endswith(".") or prompt.count(" ") > 6:
        return False
    return True


def _try_sam_prompt(
    pil_image: Image.Image,
    current_prompt: str,
    iteration: int,
    step_idx: int,
    seg_threshold: float,
    mask_threshold: float,
) -> tuple[AgentStep, dict]:
    """Run SAM3 with a single prompt and return (AgentStep, raw_sam_result)."""
    t0 = time.time()
    sam_result = call_sam(
        pil_image, text_prompt=current_prompt,
        threshold=seg_threshold, mask_threshold=mask_threshold,
    )
    sam_elapsed = (time.time() - t0) * 1000

    n_masks = len(sam_result["scores"])
    max_score = float(sam_result["scores"].max()) if n_masks > 0 else 0.0
    boxes_list = sam_result["boxes"].tolist() if n_masks > 0 else []
    scores_list = sam_result["scores"].tolist() if n_masks > 0 else []

    result_img = None
    if n_masks > 0:
        result_img = _render_boxes_on_image(
            pil_image, boxes_list, scores_list,
            f"SAM3: '{current_prompt}' ({n_masks} det.)",
        )

    step = AgentStep(
        round_idx=step_idx,
        action=f"SAM3 Segment (iter {iteration})",
        prompt_used=current_prompt,
        sam_n_masks=n_masks,
        sam_max_score=max_score,
        sam_boxes=boxes_list,
        sam_scores=scores_list,
        result_image=result_img,
        elapsed_ms=sam_elapsed,
    )
    return step, sam_result


def _verify_result(
    pil_image: Image.Image,
    step: AgentStep,
    current_prompt: str,
    object_type: str,
    verification_prompt: str,
) -> bool:
    """Ask VLM to verify a SAM3 result. Returns True if the detection is valid."""
    t0 = time.time()
    filled_verify = verification_prompt.format(
        sam_prompt=current_prompt,
        object_type=object_type,
        n_masks=step.sam_n_masks,
        scores=step.sam_scores,
        boxes=step.sam_boxes,
    )
    verify_messages = [
        {"role": "system", "content": filled_verify},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {
                    "type": "text",
                    "text": "Is this segmentation result a valid defect detection?",
                },
            ],
        },
    ]
    vlm_reply = vlm_generate(pil_image, verify_messages, max_new_tokens=256)
    verify_elapsed = (time.time() - t0) * 1000

    parsed = _parse_json(vlm_reply)
    step.vlm_raw_reply = vlm_reply
    step.parsed_result = parsed or {"raw": vlm_reply}
    step.elapsed_ms += verify_elapsed

    if parsed and parsed.get("ok", True):
        step.action += " → Verified OK"
        return True

    step.action += " → Rejected"
    return False


def _get_refinement(
    pil_image: Image.Image,
    step: AgentStep,
    current_prompt: str,
    object_type: str,
    tried_prompts: list[str],
    refinement_prompt: str,
) -> str | None:
    """Ask VLM for a better prompt after SAM3 found nothing."""
    t0 = time.time()
    filled_refine = refinement_prompt.format(
        object_type=object_type,
        failed_prompt=current_prompt,
        tried_prompts=tried_prompts,
    )
    refine_messages = [
        {"role": "system", "content": filled_refine},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {
                    "type": "text",
                    "text": (
                        f"The prompt '{current_prompt}' found nothing. "
                        "Suggest a better prompt."
                    ),
                },
            ],
        },
    ]
    vlm_reply = vlm_generate(pil_image, refine_messages, max_new_tokens=128)
    refine_elapsed = (time.time() - t0) * 1000

    parsed = _parse_json(vlm_reply)
    step.vlm_raw_reply = vlm_reply
    step.parsed_result = parsed or {"raw": vlm_reply}
    step.elapsed_ms += refine_elapsed
    step.action += " → No detections"

    if parsed and "sam_prompt" in parsed:
        candidate = parsed["sam_prompt"]
        if _is_valid_sam_prompt(candidate):
            return candidate
    return None


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def _render_boxes_on_image(
    pil_image: Image.Image,
    boxes: list[list[float]],
    scores: list[float],
    title: str = "",
) -> np.ndarray:
    """Render bounding boxes on an image and return as numpy RGB array."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(pil_image)

    cmap = matplotlib.colormaps.get_cmap("rainbow").resampled(max(len(boxes), 1))
    for i, (box, score) in enumerate(zip(boxes, scores)):
        x1, y1, x2, y2 = box
        color = [float(c) for c in cmap(i)[:3]]
        rect = mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            fill=False, edgecolor=color, linewidth=2.5,
        )
        ax.add_patch(rect)
        ax.text(
            x1, max(y1 - 6, 0), f"{score:.3f}",
            color="white", fontsize=10, fontweight="bold",
            bbox=dict(facecolor="#222222", alpha=0.8, pad=2, edgecolor=color),
        )

    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.axis("off")
    fig.tight_layout(pad=0.5)

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return buf


def render_plain_image(pil_image: Image.Image, title: str = "") -> np.ndarray:
    """Render a PIL image with a title via matplotlib."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(pil_image)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.axis("off")
    fig.tight_layout(pad=0.5)

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return buf


def _render_final_result(
    pil_image: Image.Image, sam_result: dict, prompt: str,
) -> np.ndarray:
    """Render the final SAM3 result with masks and boxes side by side."""
    masks = sam_result["masks"]
    boxes = sam_result["boxes"]
    scores = sam_result["scores"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    axes[0].imshow(pil_image)
    axes[0].set_title(f"Detections  |  Best prompt: '{prompt}'", fontsize=11)
    axes[0].axis("off")
    cmap = matplotlib.colormaps.get_cmap("rainbow").resampled(max(len(masks), 1))
    for i, (box, score) in enumerate(zip(boxes, scores)):
        x1, y1, x2, y2 = box
        color = [float(c) for c in cmap(i)[:3]]
        rect = plt.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color, facecolor="none",
        )
        axes[0].add_patch(rect)
        axes[0].text(
            x1, y1 - 4, f"{score:.2f}", color=color, fontsize=9, fontweight="bold",
        )

    composite = pil_image.convert("RGBA")
    for i, mask in enumerate(masks):
        color = tuple(int(c * 255) for c in cmap(i)[:3])
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        overlay = Image.new("RGBA", composite.size, color + (0,))
        overlay.putalpha(mask_img.point(lambda v: int(v * 0.5)))
        composite = Image.alpha_composite(composite, overlay)

    axes[1].imshow(composite)
    axes[1].set_title(f"SAM3 masks  ({len(masks)} instance(s))", fontsize=11)
    axes[1].axis("off")

    plt.tight_layout()
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return buf


def _format_agent_trace(steps: list[AgentStep]) -> str:
    """Format the agent trace as rich Markdown for display."""
    lines = ["## Agent Reasoning Trace\n"]

    for step in steps:
        icon = "🔍" if "Object" in step.action else (
            "🧠" if "Analysis" in step.action else (
                "✅" if "Verified OK" in step.action else (
                    "❌" if "No detections" in step.action else "🔄"
                )
            )
        )
        lines.append(f"### {icon} Step {step.round_idx}: {step.action}")
        lines.append(f"*Time: {step.elapsed_ms:.0f}ms*\n")

        if step.prompt_used:
            lines.append(f"**SAM3 Prompt:** `{step.prompt_used}`\n")

        if step.sam_n_masks > 0:
            lines.append(
                f"**Detections:** {step.sam_n_masks} mask(s), "
                f"max score: **{step.sam_max_score:.4f}**\n"
            )
            lines.append(f"**Scores:** {[f'{s:.3f}' for s in step.sam_scores]}\n")

        if step.vlm_raw_reply:
            truncated = step.vlm_raw_reply[:500]
            if len(step.vlm_raw_reply) > 500:
                truncated += "..."
            lines.append(f"**VLM Reply:**\n```json\n{truncated}\n```\n")

        lines.append("---\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_yaml_config(path: str) -> dict:
    with open(path) as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a YAML mapping, got {type(raw)}")
    return raw


# ---------------------------------------------------------------------------
# Data loading and filtering
# ---------------------------------------------------------------------------

def load_and_filter(
    data_dir: str, split: str, is_defect: str, major_defect: str,
    defect_type: str, defect_class: str,
) -> pd.DataFrame:
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


# ---------------------------------------------------------------------------
# Image preparation
# ---------------------------------------------------------------------------

def prepare_image(
    image_path: str, base_dir: str, apply_mask: bool,
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


def _parse_input_size(val: str) -> int | None:
    if not val or val.strip().lower() in ("none", "null", ""):
        return None
    try:
        return int(val.strip())
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# GUI callbacks
# ---------------------------------------------------------------------------

def apply_filters(data_dir, split, is_defect, major_defect, defect_type, defect_class):
    if not data_dir or not os.path.isdir(data_dir):
        return "Invalid data directory.", gr.update(), gr.update(), None, ""

    try:
        df = load_and_filter(
            data_dir, split, is_defect, major_defect, defect_type, defect_class,
        )
    except Exception as e:
        return f"Error: {e}", gr.update(), gr.update(), None, ""

    _state.df = df
    _state.data_dir = data_dir
    _state.current_idx = 0

    n = len(df)
    status = f"Loaded **{n}** images (split: {split})"

    if n == 0:
        return status, gr.update(maximum=0, value=0), gr.update(), None, ""

    slider_update = gr.update(minimum=0, maximum=n - 1, value=0, step=1)
    img, info = _get_current_display(False, False, None)
    return status, slider_update, gr.update(value=f"1 / {n}"), img, info


def navigate(idx, crop, mask, input_size_str):
    if _state.df is None or len(_state.df) == 0:
        return None, "", gr.update()

    idx = max(0, min(int(idx), len(_state.df) - 1))
    _state.current_idx = idx
    input_size = _parse_input_size(input_size_str)
    img, info = _get_current_display(crop, mask, input_size)
    counter = f"{idx + 1} / {len(_state.df)}"
    return img, info, gr.update(value=counter)


def go_prev(crop, mask, input_size_str, current_slider):
    if _state.df is None or len(_state.df) == 0:
        return None, "", gr.update(), gr.update()

    idx = max(0, int(current_slider) - 1)
    _state.current_idx = idx
    input_size = _parse_input_size(input_size_str)
    img, info = _get_current_display(crop, mask, input_size)
    counter = f"{idx + 1} / {len(_state.df)}"
    return img, info, gr.update(value=idx), gr.update(value=counter)


def go_next(crop, mask, input_size_str, current_slider):
    if _state.df is None or len(_state.df) == 0:
        return None, "", gr.update(), gr.update()

    n = len(_state.df)
    idx = min(n - 1, int(current_slider) + 1)
    _state.current_idx = idx
    input_size = _parse_input_size(input_size_str)
    img, info = _get_current_display(crop, mask, input_size)
    counter = f"{idx + 1} / {n}"
    return img, info, gr.update(value=idx), gr.update(value=counter)


def _get_current_display(
    crop: bool, mask: bool, input_size: int | None,
) -> tuple[np.ndarray | None, str]:
    if _state.df is None or len(_state.df) == 0:
        return None, ""

    idx = _state.current_idx
    row = _state.df.iloc[idx]
    orig_path = row.get("query_image", "")
    is_defect = bool(row.get("defect", False))
    defect_types = str(row.get("defect_types", ""))
    is_major = bool(row.get("major_defect", False))

    info = f"### Image {idx + 1}\n\n"
    info += f"**Path:** `{orig_path}`\n\n"
    info += f"**Defect:** {'Yes' if is_defect else 'No'}"
    if is_defect:
        info += f" | **Type:** {defect_types} | **Major:** {'Yes' if is_major else 'No'}"
    info += "\n"

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
        proc_title = " + ".join(label_parts) if label_parts else "Input Image"
        img_arr = render_plain_image(processed_img, proc_title)
    except Exception:
        img_arr = None

    return img_arr, info


def run_agentic_gui(
    crop, mask, input_size_str, max_iters, seg_threshold, mask_threshold,
    obj_detect_prompt, defect_analysis_prompt_text,
    verification_prompt_text, refinement_prompt_text,
    current_slider,
):
    """Main callback: run the agentic pipeline on the current image."""
    if _state.df is None or len(_state.df) == 0:
        return None, "No dataset loaded.", None, ""

    if not _sam.ready:
        return None, "SAM3 model not loaded yet.", None, ""

    if not _vlm.ready:
        return None, "VLM (Qwen2.5-VL) not loaded yet.", None, ""

    idx = int(current_slider)
    row = _state.df.iloc[idx]
    col = "query_crop" if crop else "query_image"
    image_path = row[col]
    input_size = _parse_input_size(input_size_str)
    pil_img = prepare_image(image_path, _state.data_dir, mask, input_size)

    steps, presence_score, best_result = run_agentic_inference(
        pil_image=pil_img,
        max_iters=int(max_iters),
        seg_threshold=seg_threshold,
        mask_threshold=mask_threshold,
        object_detection_prompt=obj_detect_prompt,
        defect_analysis_prompt=defect_analysis_prompt_text,
        verification_prompt=verification_prompt_text,
        refinement_prompt=refinement_prompt_text,
    )

    trace_md = _format_agent_trace(steps)

    result_summary = "### Results\n\n"
    if presence_score is not None:
        result_summary += f"**Presence Score:** {presence_score:.4f}\n\n"
        result_summary += f"**Total iterations:** {sum(1 for s in steps if 'SAM3' in s.action)}\n"
    else:
        result_summary += "**No defect detected.** All prompts returned 0 detections.\n"
        result_summary += f"**Prompts tried:** {sum(1 for s in steps if 'SAM3' in s.action)}\n"

    result_img = None
    if best_result is not None and len(best_result["masks"]) > 0:
        best_prompt = ""
        for s in reversed(steps):
            if s.prompt_used and s.sam_n_masks > 0:
                best_prompt = s.prompt_used
                break
        result_img = _render_final_result(pil_img, best_result, best_prompt)

    step_gallery = []
    for s in steps:
        if s.result_image is not None:
            step_gallery.append((s.result_image, s.action))

    return result_img, result_summary, trace_md, step_gallery


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
.main-title {
    text-align: center;
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    color: white !important;
    padding: 20px 30px;
    border-radius: 12px;
    margin-bottom: 16px;
    font-size: 1.5em;
}
.main-title h1 { color: white !important; margin: 0; }
.main-title p { color: #b8c6db !important; margin: 4px 0 0 0; font-size: 0.6em; }
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
    with gr.Blocks(
        title="Agentic SAM3 — Defect Detection",
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(
            primary_hue="violet",
            secondary_hue="blue",
            neutral_hue="slate",
        ),
    ) as app:

        gr.HTML(
            '<div class="main-title">'
            "<h1>Agentic SAM3 Defect Detection</h1>"
            "<p>Iterative defect detection powered by Qwen2.5-VL reasoning + SAM3 segmentation</p>"
            "</div>"
        )

        with gr.Row():
            # ==============================================================
            # LEFT COLUMN — Controls
            # ==============================================================
            with gr.Column(scale=1, min_width=380):

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
                        "Apply Filters & Load Dataset", variant="primary", size="lg",
                    )
                    filter_status = gr.Markdown("")

                with gr.Group():
                    gr.Markdown("### Image Processing")
                    with gr.Row():
                        crop = gr.Checkbox(
                            label="Use Crop",
                            value=default_cfg.get("crop", False),
                        )
                        mask_toggle = gr.Checkbox(
                            label="Apply Mask",
                            value=default_cfg.get("mask", False),
                        )
                    input_size = gr.Textbox(
                        label="Input Size (px, blank = original)",
                        value=str(default_cfg.get("input_size", "") or ""),
                        placeholder="e.g. 1024",
                    )

                with gr.Group():
                    gr.Markdown("### Agentic Settings")
                    max_iters = gr.Slider(
                        label="Max Iterations",
                        minimum=1, maximum=20, step=1, value=5,
                        info="Maximum number of SAM3 segmentation attempts",
                    )
                    seg_threshold = gr.Slider(
                        label="Detection Threshold",
                        minimum=0.0, maximum=1.0, step=0.05, value=0.3,
                    )
                    mask_threshold_slider = gr.Slider(
                        label="Mask Threshold",
                        minimum=0.0, maximum=1.0, step=0.05,
                        value=default_cfg.get("mask_threshold", 0.5),
                    )
                    run_btn = gr.Button(
                        "Run Agentic Inference",
                        variant="primary", size="lg", elem_classes=["run-btn"],
                    )

            # ==============================================================
            # RIGHT COLUMN — Display
            # ==============================================================
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
                    label="Image Index", minimum=0, maximum=0, step=1, value=0,
                )

                with gr.Tabs():
                    with gr.Tab("Input Image"):
                        img_display = gr.Image(
                            label="Current Image", type="numpy", interactive=False,
                        )
                        image_info = gr.Markdown("")

                    with gr.Tab("Result"):
                        result_img = gr.Image(
                            label="Best Detection Result", type="numpy",
                            interactive=False,
                        )
                        result_summary = gr.Markdown("")

                    with gr.Tab("Agent Trace"):
                        agent_trace = gr.Markdown(
                            value="*Run agentic inference to see the reasoning trace.*",
                        )

                    with gr.Tab("Step Gallery"):
                        step_gallery = gr.Gallery(
                            label="Intermediate SAM3 Results",
                            columns=2, rows=3, height="auto",
                        )

                    with gr.Tab("Prompt Editor"):
                        gr.Markdown(
                            "Edit the prompts used at each stage of the agentic "
                            "pipeline. Use `{object_type}`, `{description}`, "
                            "`{sam_prompt}`, `{n_masks}`, `{scores}`, `{boxes}`, "
                            "`{failed_prompt}`, `{tried_prompts}` as placeholders."
                        )
                        obj_detect_prompt = gr.Textbox(
                            label="1) Object Detection (System Prompt)",
                            value=DEFAULT_OBJECT_DETECTION_PROMPT,
                            lines=8, elem_classes=["prompt-editor"],
                        )
                        defect_analysis_prompt_ui = gr.Textbox(
                            label="2) Defect Analysis (System Prompt)",
                            value=DEFAULT_DEFECT_ANALYSIS_PROMPT,
                            lines=12, elem_classes=["prompt-editor"],
                        )
                        verification_prompt_ui = gr.Textbox(
                            label="3) Verification (System Prompt)",
                            value=DEFAULT_VERIFICATION_PROMPT,
                            lines=8, elem_classes=["prompt-editor"],
                        )
                        refinement_prompt_ui = gr.Textbox(
                            label="4) Refinement (System Prompt)",
                            value=DEFAULT_REFINEMENT_PROMPT,
                            lines=8, elem_classes=["prompt-editor"],
                        )

        # ==================================================================
        # Event wiring
        # ==================================================================

        filter_btn.click(
            fn=apply_filters,
            inputs=[
                data_dir, split, is_defect, major_defect, defect_type, defect_class,
            ],
            outputs=[filter_status, image_slider, counter, img_display, image_info],
        )

        image_slider.change(
            fn=navigate,
            inputs=[image_slider, crop, mask_toggle, input_size],
            outputs=[img_display, image_info, counter],
        )

        prev_btn.click(
            fn=go_prev,
            inputs=[crop, mask_toggle, input_size, image_slider],
            outputs=[img_display, image_info, image_slider, counter],
        )

        next_btn.click(
            fn=go_next,
            inputs=[crop, mask_toggle, input_size, image_slider],
            outputs=[img_display, image_info, image_slider, counter],
        )

        run_btn.click(
            fn=run_agentic_gui,
            inputs=[
                crop, mask_toggle, input_size, max_iters,
                seg_threshold, mask_threshold_slider,
                obj_detect_prompt, defect_analysis_prompt_ui,
                verification_prompt_ui, refinement_prompt_ui,
                image_slider,
            ],
            outputs=[result_img, result_summary, agent_trace, step_gallery],
        )

    return app


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Agentic SAM3 Inference GUI")
    parser.add_argument(
        "--config", type=str, default=None,
        help="Optional YAML config file for default values.",
    )
    parser.add_argument(
        "--gpu", type=int, default=0,
        help="GPU device index to use (default: 0).",
    )
    parser.add_argument(
        "--port", type=int, default=7862,
        help="Port to serve the Gradio app on.",
    )
    parser.add_argument(
        "--share", action="store_true",
        help="Create a public Gradio share link.",
    )
    parser.add_argument(
        "--vlm-model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Qwen2.5-VL model ID from HuggingFace.",
    )
    parser.add_argument(
        "--sam-model", type=str, default=None,
        help="SAM3 model ID (default: facebook/sam3).",
    )
    parser.add_argument(
        "--cache-dir", type=str, default=None,
        help="HuggingFace cache directory for model weights.",
    )
    parser.add_argument(
        "--hf-token", type=str, default=None,
        help=(
            "Hugging Face access token for gated models (SAM3 and VLM); "
            "optional if already logged in via the Hugging Face CLI."
        ),
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Default data directory for the Kaputt1 dataset.",
    )
    parser.add_argument(
        "--load-in-4bit", action="store_true",
        help="Load VLM in 4-bit quantization to save ~8GB VRAM (requires bitsandbytes).",
    )
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _sam.device = device
    _vlm.device = device

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    default_cfg = {}
    if args.config and os.path.isfile(args.config):
        default_cfg = load_yaml_config(args.config)

    sam_model_name = args.sam_model or default_cfg.get("model_name", "facebook/sam3")
    cache_dir = args.cache_dir or default_cfg.get("cache_dir")
    if args.hf_token:
        from huggingface_hub import login as hf_login
        hf_login(token=args.hf_token)
    if args.data_dir:
        default_cfg["data_dir"] = args.data_dir

    _sam.load(sam_model_name, cache_dir, None)
    _vlm.load(args.vlm_model, cache_dir, load_in_4bit=args.load_in_4bit)

    app = build_ui(default_cfg)
    app.launch(
        server_port=args.port,
        share=args.share,
        server_name="0.0.0.0",
    )


if __name__ == "__main__":
    main()
