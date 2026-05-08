"""
Recompute AP and AUROC using per-prompt best thresholds or VLM verification.

After an experiment is run, this script loads the saved raw pair scores
and applies each prompt's individually optimal threshold as a score gate:
scores below the prompt's threshold are zeroed out.  This suppresses
weak false-positive detections per prompt before aggregating across
prompts, which can improve the overall ranking.

Modes:
    results  — use per-prompt best_threshold from results.json (default)
    roc      — find optimal threshold per prompt from ROC analysis
    vlm      — use a Vision-Language Model (Qwen2.5-VL) to verify each
               detection by re-running SAM3 and asking the VLM whether the
               detected region is a real defect; scores rejected by the VLM
               are zeroed out

Usage:
    python recompute_with_thresholds.py --exp-dir experiments/multi_prompt_no_mask
    python recompute_with_thresholds.py --exp-dir experiments/session_20260103_120000/multi_prompt_crop
    python recompute_with_thresholds.py --exp-dir experiments/multi_prompt_no_mask \
        --thresholds-from results
    python recompute_with_thresholds.py --exp-dir experiments/multi_prompt_no_mask \
        --thresholds-from roc
    python recompute_with_thresholds.py --exp-dir experiments/multi_prompt_no_mask \
        --thresholds-from vlm --gpu 0

Required files in <exp-dir>:
    - results.json      (per-prompt best thresholds + config)
    - pair_scores.json  (raw per-(prompt, image) scores saved by the main pipeline)
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from huggingface_hub import login as hf_login
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def load_experiment(exp_dir: str) -> tuple[dict, dict]:
    """Load results.json and pair_scores.json from an experiment directory."""
    results_path = os.path.join(exp_dir, "results.json")
    scores_path = os.path.join(exp_dir, "pair_scores.json")

    if not os.path.isfile(results_path):
        print(f"Error: {results_path} not found.", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(scores_path):
        print(
            f"Error: {scores_path} not found.\n"
            "Re-run the main experiment to generate this file "
            "(requires updated experiment.py).",
            file=sys.stderr,
        )
        sys.exit(1)

    with open(results_path) as f:
        results = json.load(f)
    with open(scores_path) as f:
        raw = json.load(f)

    return results, raw


def build_prompt_thresholds_from_results(results: dict) -> dict[str, float]:
    """
    Extract per-prompt best thresholds from the experiment results.json.
    These were computed during the original run by maximising precision
    over the configured threshold candidates.
    """
    thresholds = {}
    for entry in results.get("per_prompt", []):
        prompt = entry["prompt"]
        bt = entry.get("best_threshold")
        if bt is not None and not (isinstance(bt, float) and np.isnan(bt)):
            thresholds[prompt] = float(bt)
    return thresholds


def build_prompt_thresholds_from_roc(
    raw: dict,
    labels: np.ndarray,
) -> dict[str, float]:
    """
    Find the optimal threshold per prompt by maximising Youden's J statistic
    (TPR - FPR) on the ROC curve. This is threshold-list-independent and
    often finds a better operating point than the discrete candidate list.
    """
    prompt_data: dict[str, dict[str, list]] = {}
    for key_str, data in raw["pair_scores"].items():
        _dtype, prompt = json.loads(key_str)
        if prompt not in prompt_data:
            prompt_data[prompt] = {"indices": [], "scores": []}
        prompt_data[prompt]["indices"].extend(data["indices"])
        prompt_data[prompt]["scores"].extend(data["scores"])

    thresholds: dict[str, float] = {}
    for prompt, pdata in prompt_data.items():
        p_indices = np.array(pdata["indices"])
        p_scores = np.array(pdata["scores"])
        p_labels = labels[p_indices]

        n_pos = int(p_labels.sum())
        n_neg = int(len(p_labels) - n_pos)
        if n_pos == 0 or n_neg == 0:
            thresholds[prompt] = 0.5
            continue

        fpr, tpr, roc_t = roc_curve(p_labels, p_scores)
        j_scores = tpr - fpr
        best_idx = int(np.argmax(j_scores))
        thresholds[prompt] = float(roc_t[best_idx])

    return thresholds


VLM_VERIFICATION_PROMPT = (
    "You are a quality control inspector.\n\n"
    "This image shows a {object_type}. A defect detection model flagged this "
    "image as potentially containing: '{sam_prompt}'.\n"
    "The model found {n_masks} region(s) with confidence scores: {scores} "
    "and bounding boxes: {boxes}.\n\n"
    "Look at the image carefully. Is there actually a visible defect matching "
    "'{sam_prompt}' in the detected region(s)?\n"
    "A box covering the entire image is NOT a valid defect detection.\n\n"
    "Reply with ONLY a JSON object:\n"
    '{{\"ok\": true/false, \"reason\": \"<brief explanation>\"}}\n\n'
    "No explanation, no markdown fences, just the JSON."
)

VLM_OBJECT_DETECTION_PROMPT = (
    "You are a precise visual analysis assistant. Identify the primary object "
    "or product shown in this image.\n\n"
    "Reply with ONLY a JSON object:\n"
    '{\"object_type\": \"<concise name>\"}\n\n'
    "No explanation, no markdown fences, just the JSON."
)


def _load_vlm(vlm_model_id: str, cache_dir: str | None, load_in_4bit: bool = False):
    """Load Qwen2.5-VL model and processor."""
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"[VLM] Loading '{vlm_model_id}' ...")
    processor = AutoProcessor.from_pretrained(
        vlm_model_id, trust_remote_code=True, cache_dir=cache_dir,
        min_pixels=256 * 28 * 28,
        max_pixels=512 * 28 * 28,
    )

    model_kwargs = dict(
        device_map="auto", torch_dtype=dtype,
        trust_remote_code=True, cache_dir=cache_dir,
    )
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=dtype, bnb_4bit_quant_type="nf4",
        )
        model_kwargs.pop("torch_dtype", None)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        vlm_model_id, **model_kwargs,
    )
    model.eval()
    print("[VLM] Model loaded.")
    return model, processor


def _load_sam(sam_model_id: str, cache_dir: str | None):
    """Load SAM3 model and processor."""
    import torch
    from transformers import Sam3Model, Sam3Processor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"[SAM3] Loading '{sam_model_id}' ...")
    processor = Sam3Processor.from_pretrained(sam_model_id, cache_dir=cache_dir)
    model = Sam3Model.from_pretrained(
        sam_model_id, cache_dir=cache_dir, torch_dtype=dtype,
    ).to(device)
    model.eval()
    print("[SAM3] Model loaded.")
    return model, processor, device


def _vlm_generate(model, processor, image, messages, max_new_tokens=256):
    """Run a single VLM inference and return the text reply."""
    import torch

    torch.cuda.empty_cache()
    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = processor(text=[text_input], images=[image], return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
        )

    new_tokens = generated_ids[0][input_len:]
    reply = processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    del inputs, generated_ids, new_tokens
    torch.cuda.empty_cache()
    return reply


def _call_sam_for_boxes(model, processor, device, image, text_prompt, threshold=0.3):
    """Run SAM3 and return (boxes_list, scores_list, n_masks)."""
    import torch

    torch.cuda.empty_cache()
    inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_instance_segmentation(
        outputs, threshold=threshold, mask_threshold=0.5,
        target_sizes=inputs.get("original_sizes").tolist(),
    )[0]

    n = len(results["scores"])
    if n == 0:
        boxes, scores = [], []
    else:
        boxes = results["boxes"].cpu().to(torch.float32).tolist()
        scores = results["scores"].cpu().to(torch.float32).tolist()

    del inputs, outputs, results
    torch.cuda.empty_cache()
    return boxes, scores, n


def _parse_json_reply(raw: str) -> dict | None:
    """Extract a JSON object from a VLM reply."""
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


def recompute_scores_vlm(
    raw: dict,
    results: dict,
    vlm_model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    load_in_4bit: bool = False,
    min_score: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Recompute scores using VLM verification.

    For each (prompt, image) pair where the SAM3 score exceeds min_score:
    1. Load the image
    2. Re-run SAM3 to get bounding boxes
    3. Ask the VLM to verify if the detection is a real defect
    4. If the VLM rejects it, zero out the score

    Returns (labels, max_scores, vlm_stats).
    """
    from PIL import Image
    from tqdm import tqdm

    from src.common.data import load_and_filter_data, prepare_image
    from src.sam3.config import Config

    cfg_dict = results["config"]
    valid_keys = set(Config.__dataclass_fields__.keys())
    cfg_filtered = {k: v for k, v in cfg_dict.items() if k in valid_keys}
    cfg = Config(**cfg_filtered)

    sam_model, sam_processor, sam_device = _load_sam(
        cfg.model_name, cfg.cache_dir,
    )
    vlm_model, vlm_processor = _load_vlm(
        vlm_model_id, cfg.cache_dir, load_in_4bit,
    )

    df = load_and_filter_data(cfg)
    col = "query_crop" if cfg.crop else "query_image"
    image_paths = list(df[col])

    labels = np.array(raw["labels"])
    num_images = len(labels)
    max_scores = np.zeros(num_images, dtype=np.float64)

    candidates = []
    for key_str, data in raw["pair_scores"].items():
        _dtype, prompt = json.loads(key_str)
        for idx, score in zip(data["indices"], data["scores"]):
            if score >= min_score:
                candidates.append((idx, prompt, score))
            else:
                max_scores[idx] = max(max_scores[idx], score)

    unique_images = sorted(set(idx for idx, _, _ in candidates))
    print(f"\n[VLM] {len(candidates)} (image, prompt) pairs with score >= {min_score}")
    print(f"[VLM] {len(unique_images)} unique images to verify")

    object_type_cache: dict[int, str] = {}

    stats = {"total": len(candidates), "verified": 0, "rejected": 0, "errors": 0}

    for idx, prompt, original_score in tqdm(candidates, desc="VLM verification"):
        img_path = image_paths[idx]
        pil_img = prepare_image(img_path, cfg.data_dir, cfg.mask, cfg.input_size)

        if idx not in object_type_cache:
            obj_messages = [
                {"role": "system", "content": VLM_OBJECT_DETECTION_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_img},
                        {"type": "text", "text": "Identify the object in this image."},
                    ],
                },
            ]
            reply = _vlm_generate(vlm_model, vlm_processor, pil_img, obj_messages, 128)
            parsed = _parse_json_reply(reply)
            object_type_cache[idx] = (
                parsed.get("object_type", "product") if parsed else "product"
            )

        object_type = object_type_cache[idx]

        boxes, scores, n_masks = _call_sam_for_boxes(
            sam_model, sam_processor, sam_device, pil_img, prompt,
        )

        if n_masks == 0:
            stats["rejected"] += 1
            continue

        verify_prompt = VLM_VERIFICATION_PROMPT.format(
            object_type=object_type,
            sam_prompt=prompt,
            n_masks=n_masks,
            scores=[f"{s:.3f}" for s in scores],
            boxes=[[round(c, 1) for c in b] for b in boxes],
        )
        verify_messages = [
            {"role": "system", "content": verify_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text", "text": "Is this a valid defect detection?"},
                ],
            },
        ]
        reply = _vlm_generate(vlm_model, vlm_processor, pil_img, verify_messages)
        parsed = _parse_json_reply(reply)

        if parsed is None:
            stats["errors"] += 1
            max_scores[idx] = max(max_scores[idx], original_score)
            continue

        if parsed.get("ok", True):
            stats["verified"] += 1
            max_scores[idx] = max(max_scores[idx], original_score)
        else:
            stats["rejected"] += 1

    print(f"\n[VLM] Results: {stats['verified']} verified, "
          f"{stats['rejected']} rejected, {stats['errors']} parse errors")

    return labels, max_scores, stats


def recompute_scores(
    raw: dict,
    prompt_thresholds: dict[str, float],
    default_threshold: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply per-prompt thresholds to the raw pair scores and recompute
    the max-over-all-prompts score per image.

    For each (prompt, image) pair, if the score is below the prompt's
    threshold it is set to 0.0.  Then max_scores[image] = max across
    all filtered prompt scores for that image.

    Returns (labels, max_scores).
    """
    labels = np.array(raw["labels"])
    num_images = len(labels)
    max_scores = np.zeros(num_images, dtype=np.float64)

    for key_str, data in raw["pair_scores"].items():
        _dtype, prompt = json.loads(key_str)
        threshold = prompt_thresholds.get(prompt, default_threshold)
        for idx, score in zip(data["indices"], data["scores"]):
            filtered = score if score >= threshold else 0.0
            max_scores[idx] = max(max_scores[idx], filtered)

    return labels, max_scores


def print_comparison(
    labels: np.ndarray,
    original_ap: float,
    original_auroc: float,
    new_max_scores: np.ndarray,
    prompt_thresholds: dict[str, float],
) -> dict:
    """Print before/after comparison and return the new metrics."""
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)

    if n_pos > 0 and n_neg > 0:
        new_ap = average_precision_score(labels, new_max_scores)
        new_auroc = roc_auc_score(labels, new_max_scores)
    else:
        new_ap = float("nan")
        new_auroc = float("nan")

    num_detected = int((new_max_scores > 0).sum())

    print()
    print("=" * 60)
    print("Per-Prompt Thresholds Applied")
    print("=" * 60)
    for prompt, t in sorted(prompt_thresholds.items()):
        print(f"  {prompt:<40s}  threshold = {t:.4f}")
    print("=" * 60)

    print()
    print("=" * 60)
    print("Metrics Comparison")
    print("=" * 60)
    print(f"  {'Metric':<25s} {'Original':>12s} {'Recomputed':>12s} {'Delta':>12s}")
    print("-" * 65)

    def fmt(v: float) -> str:
        return f"{v:.6f}" if not np.isnan(v) else "N/A"

    def delta(old: float, new: float) -> str:
        if np.isnan(old) or np.isnan(new):
            return "N/A"
        d = new - old
        return f"{d:+.6f}"

    print(f"  {'Average Precision':<25s} {fmt(original_ap):>12s} "
          f"{fmt(new_ap):>12s} {delta(original_ap, new_ap):>12s}")
    print(f"  {'AUROC':<25s} {fmt(original_auroc):>12s} "
          f"{fmt(new_auroc):>12s} {delta(original_auroc, new_auroc):>12s}")
    print(f"  {'Detected (score > 0)':<25s} {'':>12s} {num_detected:>12d}")
    print("=" * 60)

    return {
        "average_precision": new_ap,
        "auroc": new_auroc,
        "prompt_thresholds": prompt_thresholds,
    }


def save_recomputed(exp_dir: str, new_metrics: dict) -> None:
    """Write the recomputed metrics to a separate JSON file."""
    out_path = os.path.join(exp_dir, "recomputed_metrics.json")
    serializable = dict(new_metrics)
    for key in ("average_precision", "auroc"):
        v = serializable[key]
        if isinstance(v, float) and np.isnan(v):
            serializable[key] = None
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nRecomputed metrics saved to {out_path}")


def print_vlm_comparison(
    labels: np.ndarray,
    original_ap: float,
    original_auroc: float,
    new_max_scores: np.ndarray,
    vlm_stats: dict,
) -> dict:
    """Print before/after comparison for VLM verification mode."""
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)

    if n_pos > 0 and n_neg > 0:
        new_ap = average_precision_score(labels, new_max_scores)
        new_auroc = roc_auc_score(labels, new_max_scores)
    else:
        new_ap = float("nan")
        new_auroc = float("nan")

    num_detected = int((new_max_scores > 0).sum())

    print()
    print("=" * 60)
    print("VLM Verification Results")
    print("=" * 60)
    print(f"  Total candidates:    {vlm_stats['total']}")
    print(f"  Verified (kept):     {vlm_stats['verified']}")
    print(f"  Rejected (zeroed):   {vlm_stats['rejected']}")
    print(f"  Parse errors (kept): {vlm_stats['errors']}")
    print("=" * 60)

    print()
    print("=" * 60)
    print("Metrics Comparison")
    print("=" * 60)
    print(f"  {'Metric':<25s} {'Original':>12s} {'VLM-filtered':>12s} {'Delta':>12s}")
    print("-" * 65)

    def fmt(v: float) -> str:
        return f"{v:.6f}" if not np.isnan(v) else "N/A"

    def delta(old: float, new: float) -> str:
        if np.isnan(old) or np.isnan(new):
            return "N/A"
        d = new - old
        return f"{d:+.6f}"

    print(f"  {'Average Precision':<25s} {fmt(original_ap):>12s} "
          f"{fmt(new_ap):>12s} {delta(original_ap, new_ap):>12s}")
    print(f"  {'AUROC':<25s} {fmt(original_auroc):>12s} "
          f"{fmt(new_auroc):>12s} {delta(original_auroc, new_auroc):>12s}")
    print(f"  {'Detected (score > 0)':<25s} {'':>12s} {num_detected:>12d}")
    print("=" * 60)

    return {
        "average_precision": new_ap,
        "auroc": new_auroc,
        "vlm_stats": vlm_stats,
        "mode": "vlm",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute AP/AUROC using per-prompt best thresholds or VLM verification."
    )
    parser.add_argument(
        "--exp-dir",
        type=str,
        required=True,
        help="Path to the experiment directory containing results.json "
             "and pair_scores.json.",
    )
    parser.add_argument(
        "--thresholds-from",
        type=str,
        choices=["results", "roc", "vlm"],
        default="results",
        help="How to filter scores: "
             "'results' uses best_threshold from results.json (default), "
             "'roc' finds the optimal point on each prompt's ROC curve, "
             "'vlm' uses Qwen2.5-VL to verify each detection.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save recomputed metrics to recomputed_metrics.json.",
    )
    parser.add_argument(
        "--gpu", type=int, default=0,
        help="GPU device index for VLM mode (default: 0).",
    )
    parser.add_argument(
        "--vlm-model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="VLM model ID for vlm mode (default: Qwen/Qwen2.5-VL-7B-Instruct).",
    )
    parser.add_argument(
        "--load-in-4bit", action="store_true",
        help="Load VLM in 4-bit quantization to save VRAM.",
    )
    parser.add_argument(
        "--min-score", type=float, default=0.1,
        help="Only verify detections with score >= this value (default: 0.1). "
             "Lower scores are kept as-is to save VLM calls.",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help=(
            "Hugging Face access token for gated/private models "
            "(only used with --thresholds-from vlm; optional if already "
            "logged in via the Hugging Face CLI)."
        ),
    )
    args = parser.parse_args()

    results, raw = load_experiment(args.exp_dir)

    original_ap = results.get("average_precision", float("nan"))
    original_auroc = results.get("auroc", float("nan"))
    if original_ap is None:
        original_ap = float("nan")
    if original_auroc is None:
        original_auroc = float("nan")

    if args.thresholds_from == "vlm":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

        if args.hf_token:
            hf_login(token=args.hf_token)

        labels, new_max_scores, vlm_stats = recompute_scores_vlm(
            raw, results,
            vlm_model_id=args.vlm_model,
            load_in_4bit=args.load_in_4bit,
            min_score=args.min_score,
        )

        new_metrics = print_vlm_comparison(
            labels, original_ap, original_auroc,
            new_max_scores, vlm_stats,
        )
    else:
        labels = np.array(raw["labels"])

        if args.thresholds_from == "results":
            prompt_thresholds = build_prompt_thresholds_from_results(results)
        else:
            prompt_thresholds = build_prompt_thresholds_from_roc(raw, labels)

        if not prompt_thresholds:
            print("No per-prompt thresholds found. Nothing to recompute.",
                  file=sys.stderr)
            sys.exit(1)

        labels, new_max_scores = recompute_scores(raw, prompt_thresholds)

        new_metrics = print_comparison(
            labels, original_ap, original_auroc,
            new_max_scores, prompt_thresholds,
        )

    if args.save:
        save_recomputed(args.exp_dir, new_metrics)


if __name__ == "__main__":
    main()
