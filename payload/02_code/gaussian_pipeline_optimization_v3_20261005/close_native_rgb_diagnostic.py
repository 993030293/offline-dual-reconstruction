"""Close a negative component diagnostic without promoting it to a full benchmark."""

import time

STARTED = time.perf_counter()

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys

import psutil

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parent
OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/native_rgb_decode_001"
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def gone(row):
    try:
        return psutil.Process(row["pid"]).create_time() != row["creation_time"]
    except psutil.NoSuchProcess:
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--linux", action="store_true")
    args = parser.parse_args()
    destination = OUT / "r0"
    if args.linux:
        contract = read(OUT / "p0/CONTRACT.json")
        identities = contract["sources_inputs"]
        copies = sorted((OUT / "p0/source_copies").iterdir())
        copied = [{"path": str(copy), "exact": copy.read_bytes() == Path(identity["path"]).read_bytes()}
                  for copy, identity in zip(copies, identities[:7])]
        process = read(OUT / "p0/PROCESS_IDENTITY.json")
        checks = {"all19_current_input_source_identities": len(identities) == 19
                  and identities == [data_identity(row["path"]) for row in identities],
                  "all7_saved_sources_exact": len(copied) == 7 and all(row["exact"] for row in copied),
                  "actual_prior_Linux_instance_gone": gone(process),
                  "prior_Linux_guard_completed": read(OUT / "p0/WSL_RESOURCE.json")["status"] == "completed"}
        row = {"status": "passed" if all(checks.values()) else "failed", "checks": checks,
               "copies": copied, "prior_process": process, "new_optimization_count": 0}
        atomic_json(destination / "LINUX_REVIEW.json", row)
        if not all(checks.values()):
            raise AssertionError("The negative diagnostic source or Linux closure failed")
        return
    destination.mkdir(parents=True, exist_ok=False)
    owner = psutil.Process()
    atomic_json(destination / "PROCESS_IDENTITY.json", {"pid": owner.pid, "creation_time": owner.create_time()})
    lease = read(OUT / "PROFILE_LEASE_001.json")
    guard = read(OUT / "PROFILE_GUARD_001.json")
    processes = guard["owned_process_identities"] + [
        {"pid": lease["guard_pid"], "creation_time": lease["guard_creation_time"]},
        {"pid": lease["child_pid"], "creation_time": lease["child_creation_time"]}]
    unique = {(row["pid"], row["creation_time"]): row for row in processes}
    windows = [{**row, "gone": gone(row)} for row in unique.values()]
    linux_file = "/mnt/c/" + Path(__file__).as_posix()[3:]
    command = ["wsl", "-d", "E26-C2-LIV-20260907-134802", "--", "/opt/e26_thin_surface_v1/venv/bin/python",
               "-B", linux_file, "--linux"]
    subprocess.run(command, check=True, timeout=120)
    linux = read(destination / "LINUX_REVIEW.json")
    diagnostic = read(OUT / "p0/RESULT.json")
    medians = {mode: statistics.median(row["complete12_preparation_wall_seconds"][mode]
               for row in diagnostic["rows"]) for mode in ("pillow", "old_opencv", "direct_rgb")}
    checks = {"all_original_diagnostic_checks_passed": all(diagnostic["checks"].values()),
              "all_current_sources_and_saved_copies_and_Linux_closed": all(linux["checks"].values()),
              "all_prior_Windows_instances_gone": all(row["gone"] for row in windows),
              "original_guard_and_lease_closed": guard["status"] == "completed" and lease["returncode"] == 0
                  and bool(lease.get("lease_released_utc")),
              "direct_RGB_has_no_local_preparation_benefit": medians["direct_rgb"] > medians["pillow"],
              "no_full_Scene_or_training_claim": diagnostic["full1016_Scene_generic_errors_external_recovery_and_training"] == "not_run"
                  and diagnostic["new_optimization_count"] == 0}
    row = {"status": "closed_negative_bounded_component" if all(checks.values()) else "failed_review",
           "checks": checks, "raw_diagnostic_medians_seconds": medians,
           "direct_RGB_component_slowdown": medians["direct_rgb"] / medians["pillow"] - 1,
           "old_nonoverlapping_PB03_first_fees_seconds": 162.98888750001788,
           "current_diagnostic_full_exited_lease_seconds": lease["child_elapsed_seconds"],
           "review_before_publication_seconds": time.perf_counter() - STARTED,
           "windows": windows, "Linux_review": str(destination / "LINUX_REVIEW.json"),
           "root_single_chat_self_review": True, "new_optimization_count": 0,
           "formal_three_pair_complete_Scene_generic_errors_external_recovery": "not_run_after_local_screen",
           "own_review_exit_requires_caller_check": True}
    atomic_json(destination / "REVIEW.json", row)
    print(json.dumps({key: row[key] for key in ("status", "checks", "raw_diagnostic_medians_seconds", "direct_RGB_component_slowdown")}), flush=True)
    if not all(checks.values()):
        raise AssertionError("The negative diagnostic closure failed")


if __name__ == "__main__":
    main()
