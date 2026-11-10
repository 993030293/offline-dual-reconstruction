"""Bind the stopped SH diagnostic to original sources and actual process closure."""

import argparse
import datetime
import json
from pathlib import Path
import sys

import psutil

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    target = args.output.resolve()
    target.relative_to(ROOT / "04_outputs")
    failure = read(target / "FAILURE.json")
    protocol = read(target / "PROTOCOL.json")
    progress = read(target / "PROGRESS.json")
    process = read(target.with_name(target.name + "_process.json"))
    lease = read(target.with_name(target.name + "_lease.json"))
    source_checks = []
    for original in protocol["sources_inputs"]:
        path = Path(original["path"].replace("/mnt/c/", "C:/"))
        actual = data_identity(path)
        source_checks.append({"path": str(path), "bytes_and_sha_exact": all(actual[key] == original[key] for key in ("bytes", "sha256"))})
    identities = list(process["owned_process_identities"])
    identities.extend({"pid": lease[name + "_pid"], "creation_time": lease[name + "_creation_time"]} for name in ("guard", "child"))
    living = []
    for identity in identities:
        try:
            instance = psutil.Process(identity["pid"])
            if instance.create_time() == identity["creation_time"] and instance.is_running():
                living.append(identity)
        except psutil.NoSuchProcess:
            pass
    early = failure["results"][0]
    result = {"status": "failed_verification_closed", "updated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "candidate": "R106_old_Q66_unaccepted_family", "generic_cases": len(read(target / "GENERIC.json")["rows"]),
        "checks": {"all_original_sources_inputs_unchanged": all(row["bytes_and_sha_exact"] for row in source_checks),
            "all_recorded_windows_instances_exited": not living,
            "nonzero_original_guard_failure_preserved": process["status"] == "failed_child" and process["returncode"] != 0,
            "lease_drained_and_released": lease["status"] == "child_exited" and bool(lease.get("lease_released_utc")),
            "three_complete_early_pairs": len(early["pairs"]) == 3},
        "source_checks": source_checks, "same_instance_alive": living,
        "early_median_seconds": early["median_seconds"], "early_stage_reduction": early["stage_reduction"],
        "last_original_guard": progress["guards"][-1], "mature_views_executed_before_stop": len(progress["guards"]),
        "mature_pairs_complete": False, "original_gradient_failure_cause": "unconfirmed",
        "native_repeat_R95_issue_does_not_waive_R106_gate": True,
        "training_recovery_external_new_object_runs": "not_executed", "counted": False,
        "diagnostic_windows_tree_RSS_bytes": process["peak_tree_RSS_bytes"],
        "windows_RSS_does_not_include_WSL": True,
        "diagnostic_elapsed_seconds": process["elapsed_seconds"], "resource_wait_separate_seconds": lease["wait_seconds"]}
    atomic_json(target / "POSTRUN_REVIEW.json", result)
    print(json.dumps({key: result[key] for key in ("status", "checks", "early_stage_reduction", "mature_views_executed_before_stop", "counted")}), flush=True)
    if not all(result["checks"].values()):
        raise ValueError("The stopped diagnostic still has a source or process closure uncertainty")


if __name__ == "__main__":
    main()
