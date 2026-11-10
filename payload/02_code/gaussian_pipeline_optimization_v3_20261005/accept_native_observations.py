"""Accept a single observation-index family after complete native and external guards."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("campaign", "oracle", "external", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    proof, oracle, external = [read(p/"RESULT.json") for p in (args.campaign, args.oracle, args.external)]
    records = [*proof["sources"], *oracle["sources"], *external["sources_inputs"]]
    traces, invariants = [], []
    for trial in range(3):
        pair = args.campaign/f"trial{trial}"
        invariants.append(read(pair/"INVARIANTS.json"))
        for mode in ("baseline", "index"):
            target = pair/mode
            trace = read(target/"INDEX_TRACE.json")
            traces.append(trace)
            records += trace["sources"]
            snapshot = read(target/"entry/SNAPSHOT_TRACE.json")
            if not snapshot["no_held_handles_after_context"]:
                raise ValueError("The measured original input locks were not closed")
    checks = {"three_complete_original24_48_poses_pixels_and_features": len(invariants) == 3 and all(v["all_protected"] for v in invariants),
        "all_measured_local_cost_guards": all(proof["checks"].values()),
        "CPU_geometry_with_both_native_snapshots_reduces10percent": proof["reductions"]["complete_CPU_geometry_seconds"] >= .1,
        "whole_actual_entry_has_no_regression": proof["reductions"]["whole_seconds"] >= 0,
        "all_nine_heterogeneous_geometry_scale_camera_oracles": all(oracle["checks"].values()),
        "all_eight_mutations_and_truncated_payload_rejected": all(oracle["failures"].values()),
        "all_three_non_box_actual_component_pairs_protected": all(external["checks"].values()),
        "all_sources_and_external_inputs_still_same": all(r == data_identity(Path(r["path"])) for r in records),
        "full_private_map_snapshots_and_all48_native_indices": all(t["trace"]["calls"] == 48
            and (not t["trace"]["enabled"] or t["trace"]["index"]["complete_native_state_unchanged"]) for t in traces)}
    args.output.mkdir(parents=True, exist_ok=False)
    result = {"status": "accepted_one_new_mechanism" if all(checks.values()) else "failed_verification",
        "mechanism": "R94_NATIVE_IMMUTABLE_OBSERVATION_INDEX", "family": "previously_unaccepted_Q29_observation_index_counted_once",
        "additional_accepted_count": int(all(checks.values())), "checks": checks,
        "median_seconds": proof["median_seconds"], "reductions": proof["reductions"],
        "external_median_seconds": external["median_seconds"], "external_reductions": external["reductions"],
        "scope": "one_owned_immutable_COLMAP4.1.1_map_with_predeclared_native_name_and_resource_contract",
        "all_snapshot_build_and_complete_end_verification_included": True,
        "local_entry_and_real_non_box_matching_component": True,
        "whole_pipeline_generalization_new_object_quality_or_accuracy_claim": False, "independent_review": False,
        "campaign": str(args.campaign), "oracle": str(args.oracle), "external": str(args.external),
        "assessor": data_identity(Path(__file__))}
    atomic_json(args.output/"ACCEPTANCE.json", result)
    print({k: result[k] for k in ("status", "checks", "additional_accepted_count", "median_seconds", "external_median_seconds")})
    if not all(checks.values()):
        raise ValueError("A complete native observation index acceptance condition failed")


if __name__ == "__main__":
    main()
