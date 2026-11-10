"""Accept actual immutable input identity reuse with the original match guards."""

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
    for name in ("campaign", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    proof = read(args.campaign/"RESULT.json")
    oracle = read(Path(proof["oracle"])/"RESULT.json")
    checks = {"all_three_cost_pairs_original_correspondences_and_poses": all(proof["checks"].values()),
        "owned_Windows_and_WSL_byte_mutation_alias_and_SQLite_oracles": all(oracle["checks"].values()),
        "input_identity_stage_has_original10percent_gain": proof["reductions"]["identity_phase_seconds"] >= .1,
        "actual_whole_entry_also_exceeds10percent": proof["reductions"]["whole_seconds"] >= .1,
        "complete_hash_and_file_identity_not_metadata_cache": all(not p["snapshot"]["metadata_only_hash_cache"]
            and p["snapshot"]["hash_read_from_owned_handle_not_racy_path"]
            and p["snapshot"]["every_hit_reopens_and_checks_volume_file_id_and_size"]
            and p["snapshot"]["full_hashes"] > 0 and p["snapshot"]["hits"] > 0 for p in proof["pairs"]),
        "all_owned_locks_closed": all(p["all_handles_closed"] for p in proof["pairs"]),
        "all_measured_sources_and_oracle_sources_unchanged": all(r == data_identity(Path(r["path"]))
            for records in (proof["sources"], oracle["sources"]) for r in records)}
    args.output.mkdir(parents=True, exist_ok=False)
    result = {"status": "accepted_one_new_mechanism" if all(checks.values()) else "failed_verification",
        "mechanism": "R92_HELD_IMMUTABLE_INPUT_COMPLETE_IDENTITY_REUSE", "additional_accepted_count": int(all(checks.values())),
        "checks": checks, "median_seconds": proof["median_seconds"], "reductions": proof["reductions"],
        "scope": "Windows_cached_Box2_current1016_map_all24_extra_queries_with_accepted_R90_Q28_Q47_Q48_Q51",
        "component_invariant_independent_of_object_shape": True, "external_component_pending": True,
        "quality_gain_or_whole_pipeline_generalization_claim": False, "independent_review": False,
        "campaign": str(args.campaign), "assessor": data_identity(Path(__file__))}
    atomic_json(args.output/"ACCEPTANCE.json", result)
    print({k: result[k] for k in ("status", "checks", "additional_accepted_count", "median_seconds")})
    if not all(checks.values()):
        raise ValueError("An immutable snapshot acceptance condition failed")


if __name__ == "__main__":
    main()
