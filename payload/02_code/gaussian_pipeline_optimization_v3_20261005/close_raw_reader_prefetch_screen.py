"""Review the negative RAW component without claiming a full initial run."""

import argparse
import json
from pathlib import Path
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).parent
OWN = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/raw_reader_scorer_dag_001"
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity
from gpu_query_node import remaining


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    began = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.resolve().relative_to(OWN)
    args.output.mkdir(parents=True, exist_ok=False)
    generic, pilot = read(OWN / "g0/RESULT.json"), read(OWN / "p0/RESULT.json")
    contract, close = read(OWN / "p0/CONTRACT.json"), read(OWN / "p0/RESOURCE_CLOSE.json")
    guard = read(OWN / "PILOT_GUARD_001.json")
    leases = [read(OWN / name) for name in ("GENERIC_LEASE_001.json", "PILOT_LEASE_001.json")]
    identities = {}
    for row in generic["sources"] + pilot["sources"] + [guard["source"], data_identity(Path(__file__))]:
        previous = identities.setdefault(row["path"], row)
        if previous != row:
            raise ValueError("An executed RAW source has two different saved identities")
    current = [data_identity(Path(row["path"])) for row in identities.values()]
    snapshots = []
    for number, row in enumerate(pilot["sources"]):
        saved = OWN / "p0/source_snapshot" / (str(number) + "_" + Path(row["path"]).name)
        value = data_identity(saved)
        snapshots.append(value["bytes"] == row["bytes"] and value["sha256"] == row["sha256"])
    instances = list(guard["owned_process_identities"]) + [generic["own_process"],
        {"pid": close["actual_PID"], "creation_time": close["actual_creation_time"]}]
    for lease in leases:
        for prefix in ("guard", "child"):
            instances.append({"pid": lease[prefix + "_pid"], "creation_time": lease[prefix + "_creation_time"]})
    unique = {(row["pid"], row["creation_time"]): row for row in instances}
    live = remaining(list(unique.values()))
    raw = Path(contract["original_raw_path"]).stat()
    checks = {"original14_generic_checks": len(generic["checks"]) == 14 and all(generic["checks"].values()),
        "all_three_original128_full_quality_metadata_cache_and_order_exact": len(pilot["pairs"]) == 3
            and all(all(pair["checks"].values()) for pair in pilot["pairs"]),
        "only_new_reader_changed_common_R141_and_original4workers16queue": pilot["common_R141_workspace_both_sides"]
            and all(row["detail"]["submitted"] == 128 and row["detail"]["queue_capacity"] == 16 and row["detail"]["cache_hits"] == 0
                for pair in pilot["pairs"] for row in pair["modes"].values()),
        "all_three_actual_reader_queues_and_thread_owned_planes_closed": all(
            pair["modes"]["workspace"]["reader"]["calls"][0]["source_next_calls"] == 129
            and pair["modes"]["workspace"]["reader"]["calls"][0]["delivered"] == 128
            and pair["modes"]["workspace"]["reader"]["calls"][0]["queued_references_after_close"] == 0
            and not pair["modes"]["workspace"]["reader"]["calls"][0]["reader_alive_after_join"]
            and all(row["workspace"]["workspace"]["closed"] and row["workspace"]["workspace"]["owned_plane_bytes_after_close"] == 0
                for row in pair["modes"].values()) for pair in pilot["pairs"]),
        "preregistered_negative_component_stop_and_no_full_run": pilot["status"] == "component_screen_negative_below5percent"
            and pilot["reduction"] < .05 and pilot["formal_pairs_executed"] == 0,
        "current_complete_scientific_sources_and_exclusion_identity_unchanged": current == list(identities.values()),
        "all12_executed_pilot_source_snapshots": len(snapshots) == 12 and all(snapshots),
        "original_RAW_size_and_mtime_unchanged_not_full_SHA_claim": (raw.st_size, raw.st_mtime_ns)
            == (contract["original_raw_size"], contract["original_raw_mtime_ns"]),
        "actual_saved_Windows_instances_exited": not live,
        "prior_generic_pilot_leases_released": all(lease["returncode"] == 0 and lease.get("lease_released_utc") for lease in leases),
        "original_CV_thread_setting_and_bounded_actual_process_restored": close["OpenCV_threads_restored"]
            and guard["status"] == "completed" and guard["peak_tree_RSS_bytes"] <= 8 * 1024**3}
    if not all(checks.values()):
        raise AssertionError("A negative RAW scope, source, owner or original contract guard failed")
    result = {"status": "negative_RAW128_component_current_source_and_process_self_review_passed",
        "checks": checks, "review_kind": "separate_root_self_review_no_subagents", "independent_review": False,
        "new_optimization_count": 0, "parent_additional_count": 22,
        "complete_component_medians_seconds": pilot["medians"], "complete_component_reduction": pilot["reduction"],
        "generic_and_pilot_actual_computational_envelopes_seconds": sum(lease["child_elapsed_seconds"] for lease in leases),
        "executed_source_input_identities": len(identities), "executed_source_snapshots": len(snapshots),
        "actual_Windows_instances_checked": len(unique), "actual_live_instances": live,
        "executed_source_inputs": list(identities.values()), "observed_Windows_instances": list(unique.values()),
        "actual_pilot_Windows_tree_peak_RSS_bytes": guard["peak_tree_RSS_bytes"],
        "own_process": {"pid": psutil.Process().pid, "creation_time": psutil.Process().create_time()},
        "current_full7338_entry_formal_whole_cost_actual_scoring_errors_recovery_external_training_GT_preview_production": "not_run_after_negative_component",
        "full_wrapper_created_but_not_executed_or_validated": str(CODE / "run_raw_reader_prefetch.py"),
        "review_own_lease_process_exit_and_native_text_publication_pending_separate_observation": True,
        "review_before_last_publication_seconds": time.perf_counter() - began}
    atomic_json(args.output / "RESULT.json", result)
    print(json.dumps({key: result[key] for key in ("status", "checks", "executed_source_input_identities", "executed_source_snapshots", "actual_Windows_instances_checked")}), flush=True)


if __name__ == "__main__":
    main()
