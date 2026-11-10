"""Describe only the coarse positions actually bound to bundle adjustment."""

from collections import Counter


CURRENT_ALIGNMENT_SOURCE = "current_coarse_camera_positions_not_GT"
CURRENT_METRIC_SOURCE = "current_coarse_pose_meters_not_GT"
LEGACY_ALIGNMENT_SOURCE = "recorded_raw_camera_positions_not_GT"


def unique_coarse_lookup(rows):
    lookup = {}
    for row in rows:
        name = row["image_name"]
        if not isinstance(name, str) or not name or name in lookup:
            raise ValueError("Coarse position names must be nonempty and unique")
        lookup[name] = row
    return lookup


def _label(row, key):
    value = row.get(key)
    return value if isinstance(value, str) and value.strip() else "unconfirmed"


def summarize_used_priors(lookup, prior_rows):
    names = [row["image_name"] for row in prior_rows]
    if not names or len(names) != len(set(names)):
        raise ValueError("Bound position prior names must be nonempty and unique")
    if any(name not in lookup for name in names):
        raise ValueError("Every bound position prior needs its actual coarse source row")
    used = [lookup[name] for name in names]
    roles = dict(sorted(Counter(_label(row, "source_role") for row in used).items()))
    producers = dict(sorted(Counter(_label(row, "pose_producer") for row in used).items()))
    roots = dict(sorted(Counter(_label(row, "coarse_source_run_root") for row in used).items()))
    producer = next(iter(producers)) if len(producers) == 1 else "mixed"
    return {"supported_position_prior_count": len(names),
        "source_role_counts": roles, "pose_producer_counts": producers,
        "coarse_source_run_root_counts": roots, "coarse_pose_producer": producer,
        "time_offset_verified": False, "GT_used": False,
        "summary_scope": "only_actual_bound_position_prior_rows"}


def validate_alignment_source(alignment):
    if alignment.get("old_alignment_reused") is not False:
        raise ValueError("A newly computed coarse alignment is required")
    source = alignment.get("source")
    if source == LEGACY_ALIGNMENT_SOURCE:
        return
    if source != CURRENT_ALIGNMENT_SOURCE:
        raise ValueError("The alignment has no supported coarse position source")
    provenance = alignment.get("coarse_prior_provenance", {})
    count = provenance.get("supported_position_prior_count")
    if type(count) is not int or count <= 0 or provenance.get("GT_used") is not False:
        raise ValueError("Current coarse alignment lacks a valid bound-prior provenance")
    for key in ("source_role_counts", "pose_producer_counts", "coarse_source_run_root_counts"):
        counts = provenance.get(key, {})
        if (not counts or any(not isinstance(name, str) or not name or type(n) is not int or n <= 0
                              for name, n in counts.items()) or sum(counts.values()) != count):
            raise ValueError("Current coarse alignment provenance counts are inconsistent")
