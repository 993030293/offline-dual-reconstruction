"""Accept the complete measured matcher only after an actual unified route run."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"
V2 = ROOT/"02_code/gaussian_pipeline_optimization_v2_20261005"
sys.path[:0] = [str(V1), str(V2)]
from runtime import atomic_json, data_identity
from compare_actual_localization import compare


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("campaign", "entry", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    result = read(args.campaign/"RESULT.json")
    native = args.campaign/"trial0/native/distance_entry/reference_entry/query_entry/entry"
    entry = args.entry/"entry/distance_entry/reference_entry/query_entry/entry"
    exact = compare(native, entry, native/"NEIGHBOR_TRACE.json", entry/"NEIGHBOR_TRACE.json")
    trace = read(args.entry/"entry/INTEGER_SIFT_TRACE.json")
    checks = {
        "three_full_cost_pairs_and_all24_48_original_guards": all(result["checks"].values()),
        "full_native_padding_ties_range_and_batched_oracles": all(result["lifecycle"]["five_padding_tie_and_full_range_native_oracles"]),
        "non_native_component_rejected_and_worker_closed": result["lifecycle"]["non_native_float_component_rejected"]
            and result["lifecycle"]["owned_worker_complete_shutdown"],
        "actual_unified_entry_matches_all_native_outputs": exact["all_protected"],
        "current_measured_sources_unchanged": result["sources"] == [data_identity(Path(r["path"])) for r in result["sources"]],
        "actual_unified_worker_closed_and_within_original_budget": trace["trace"]["returncode"] == 0
            and trace["trace"]["shutdown"]["peak_CUDA_reserved_bytes"] <= 4*1024**3
    }
    args.output.mkdir(parents=True, exist_ok=False)
    accepted = {"status": "accepted_one_new_mechanism_current_cached_route_complete" if all(checks.values()) else "failed_verification",
        "mechanism": "R90_BATCHED_EXACT_INTEGER_SIFT_DISTANCE", "additional_accepted_count": int(all(checks.values())),
        "checks": checks, "median_seconds": result["median_seconds"], "whole_route_reduction": result["reduction"],
        "scope": "actual24_extra_frame_localization_of_current_fixed1016_map_with_Q28_Q47_Q48_Q51",
        "startup_preparation_IPC_and_shutdown_in_actual_whole_cost": True,
        "all_correspondences_distances_geometric_supports_and_poses_preserved": True,
        "GPU_reserved_bytes": trace["trace"]["shutdown"]["peak_CUDA_reserved_bytes"],
        "new_map_GT_or_quality_gain_claim": False, "independent_review": False,
        "campaign": str(args.campaign), "entry": str(args.entry), "assessor": data_identity(Path(__file__))}
    atomic_json(args.output/"ACCEPTANCE.json", accepted)
    print({"status": accepted["status"], "checks": checks, "additional_accepted_count": accepted["additional_accepted_count"]})
    if not all(checks.values()):
        raise ValueError("The current full localization route or original matcher protection failed")


if __name__ == "__main__":
    main()
