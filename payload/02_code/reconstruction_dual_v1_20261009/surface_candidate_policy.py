"""Admit surface candidates independently using the existing geometric support gates."""

import math


def select_surface_candidate(fused, visual, broad, native_selector):
    """Allow a supported broad candidate even when the strict candidate is ineligible.

    A global triangle gain cannot represent a local observed-surface improvement.
    A small gain is admitted only when metric sparse support increases, while
    retaining the original coherence, support, and maximum expansion gates.
    """
    _, evidence = native_selector(fused, visual, broad)
    candidates = (fused, visual, broad)
    for candidate in candidates:
        for key in ("dominant_component_area_fraction", "sparse_support_fraction_within_5cm"):
            value = candidate[key]
            if not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Invalid finite surface support fraction: " + key)
        if candidate["triangle_count"] <= 0 or candidate["component_count"] <= 0:
            raise ValueError("Surface candidate counts must be positive")
    thresholds = evidence["thresholds"]
    gain = evidence["broad_triangle_gain_fraction"]
    sparse_gain = broad["sparse_support_fraction_within_5cm"] - visual["sparse_support_fraction_within_5cm"]
    expansion_bounded = 0 <= gain <= thresholds["maximum_broad_triangle_gain_fraction"]
    observed_support_gain = sparse_gain > 0
    broad_adds_observed_surface = expansion_bounded and (
        evidence["broad_adds_supported_surface"] or observed_support_gain)
    selected = "registered_lidar_fused"
    if evidence["fused_fragmented"]:
        if evidence["broad_coherent"] and evidence["broad_supported"] and broad_adds_observed_surface:
            selected = "pgsr_depth_broad"
        elif evidence["visual_coherent"] and evidence["visual_supported"]:
            selected = "pgsr_depth_only"
    evidence.update({
        "method": "independent candidate admission with bounded observed sparse-support gain",
        "policy_version": "surface_candidate_admission_v2",
        "legacy_selected_mode": evidence["selected_mode"],
        "selected_mode": selected,
        "strict_visual_eligibility_required_for_broad": False,
        "broad_sparse_support_gain_fraction": sparse_gain,
        "broad_expansion_within_original_maximum": expansion_bounded,
        "broad_adds_observed_surface": broad_adds_observed_surface,
        "geometry_GT_accuracy_verified": False,
    })
    return selected, evidence
