"""Shared utilities: data loading, metrics, and plotting."""

from src.common.data import (
    LOGICAL_DEFECT_TYPES,
    PairKey,
    build_all_prompt_pairs,
    build_reference_path_lookup,
    compose_vlm_few_shot_grid,
    load_and_filter_data,
    prepare_image,
    write_inference_predictions_csv,
)
from src.common.metrics import (
    compute_auroc,
    compute_binary_metrics,
    compute_per_pair_metrics,
    compute_per_prompt_metrics,
    compute_type_max_scores,
    find_best_threshold,
    predict_classes,
)
