"""Independently reread the bounded negative screen and close its owned resources."""

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity

WSL = "E26-C2-LIV-20260907-134802"
LINUX_PYTHON = "/opt/e26_thin_surface_v1/venv/bin/python"


def local_path(value):
    path = str(value)
    if path.startswith("/mnt/c/"):
        return Path("C:/" + path[len("/mnt/c/"):])
    return Path(path)


def same_instance(pid, created):
    try:
        return psutil.Process(pid).is_running() and abs(psutil.Process(pid).create_time()-created) < 1e-6
    except psutil.NoSuchProcess:
        return False


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    own = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/registered_camera_copy_001"
    args.output.resolve().relative_to(own.resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    self_process = psutil.Process()
    atomic_json(args.output / "PROCESS_IDENTITY.json", {"scope": "Windows", "pid": self_process.pid,
        "creation_time": self_process.create_time()})
    generic = json.loads((own / "g0/RESULT.json").read_text())
    result = json.loads((own / "p0/RESULT.json").read_text())
    protocol = json.loads((own / "p0/PROTOCOL.json").read_text())
    stage = {mode: statistics.median(t["render_copy_loss_backward_seconds"][mode]
        for t in result["trials"]) for mode in ("native", "registered")}
    reduction = 1-stage["registered"]/stage["native"]
    leases = [json.loads((own / name).read_text()) for name in ("G0_LEASE.json", "P0_LEASE.json")]
    windows = [{"pid": record[key], "creation_time": record[created],
                "same_instance_running": same_instance(record[key], record[created])}
        for record in leases for key, created in (("guard_pid", "guard_creation_time"),
                                                  ("child_pid", "child_creation_time"))]
    linux = [json.loads((own / name / "PROCESS_IDENTITY.json").read_text()) for name in ("g0", "p0")]
    script = ("import json,psutil; rows=" + repr(linux) + "; out=[]\n"
        "for row in rows:\n"
        "    try: live=psutil.Process(row['pid']).is_running() and abs(psutil.Process(row['pid']).create_time()-row['creation_time'])<1e-6\n"
        "    except psutil.NoSuchProcess: live=False\n"
        "    out.append(dict(row,same_instance_running=live))\n"
        "print(json.dumps(out))")
    response = subprocess.run(["wsl.exe", "-d", WSL, "--", LINUX_PYTHON, "-c", script],
        check=True, text=True, encoding="utf-8", capture_output=True, timeout=30)
    linux = json.loads(response.stdout)
    current = []
    for identity in protocol["sources_inputs"] + generic["sources"]:
        actual = data_identity(local_path(identity["path"]))
        current.append({"recorded_path": identity["path"], "current": actual,
                        "unchanged": actual["sha256"] == identity["sha256"] and actual["bytes"] == identity["bytes"]})
    snapshots = args.output / "executed_sources"
    snapshots.mkdir()
    for identity in protocol["sources_inputs"] + generic["sources"]:
        path = local_path(identity["path"])
        if path.suffix == ".py":
            target = snapshots / path.name
            payload = path.read_bytes()
            if target.exists() and target.read_bytes() != payload:
                raise ValueError("Different executed sources unexpectedly share a snapshot filename")
            target.write_bytes(payload)
    state_path = ROOT / "research/2026-10-05-gaussian-optimization-round3-50/CURRENT_STATE.json"
    state_text = state_path.read_text(encoding="utf-8")
    pairs = json.loads(state_text, object_pairs_hook=lambda value: value)
    state = json.loads(state_text)
    checks = {
        "all19_real_CUDA_generic_true": len(generic["checks"]) == 19 and all(generic["checks"].values()),
        "three_interleaved_pairs_and_fixed_modes": len(result["trials"]) == 3 and all(
            list(t["render_copy_loss_backward_seconds"]) == (["native", "registered"] if i%2 == 0 else ["registered", "native"])
            for i, t in enumerate(result["trials"])),
        "all_three_original_input_pixel_loss_guards": all(t["all24_original_guards"] == {"registered": True} for t in result["trials"]),
        "all_six_216_copy_consumers_closed": all(t["trace"][m]["copies"] == 216 and t["trace"][m]["closed"]
            for t in result["trials"] for m in ("native", "registered")),
        "all72_original_storage_registrations_unregistered": all(t["trace"]["registered"]["registration"]["registrations"] == 24
            and t["trace"]["registered"]["registration"]["unregistrations"] == 24
            and t["trace"]["registered"]["registration"]["closed"] for t in result["trials"]),
        "no_duplicate_CPU_staging_and_single_GPU_slot": all(t["trace"]["registered"]["duplicate_CPU_staging_bytes"] == 0
            and t["trace"]["registered"]["maximum_slots"] == 1 and t["trace"]["registered"]["device_buffer_bytes"] == 6041280
            for t in result["trials"]),
        "independent_recalculation_matches_negative_result": stage == result["median_seconds"] and reduction == result["results"]["registered"]["reduction"]
            and reduction < protocol["minimum_complete_makespan_reduction"] == .05,
        "current_measured_sources_inputs_unchanged": all(row["unchanged"] for row in current) and result["sources_inputs_unchanged"],
        "both_experiment_leases_successfully_drained": all(row.get("returncode") == 0 and row.get("lease_released_utc") for row in leases),
        "four_prior_Windows_instances_exited": len(windows) == 4 and all(not row["same_instance_running"] for row in windows),
        "two_actual_Linux_instances_exited": len(linux) == 2 and all(not row["same_instance_running"] for row in linux),
        "unique_state_keys_and_parent22_active_no_experiment": all(n == 1 for n in Counter(k for k,v in pairs).values())
            and state["additional_accepted_count"] == 22 and state["active_candidates"] == [],
        "bounded_stage_not_counted_as_full_training": result["counted"] is False and result["three_seed_training_executed"] is False
            and protocol["steps_each_side"] == 216 and protocol["bounded_stage_not_full_training"] is True,
    }
    receipt = {"status": "negative_screen_self_review_passed" if all(checks.values()) else "negative_screen_self_review_failed",
        "review_kind": "separate_root_self_review_no_subagent", "checks": checks, "current_identities": current,
        "source_snapshots": [str(p) for p in snapshots.iterdir()], "Windows_instances": windows, "Linux_instances": linux,
        "stage_seconds": stage, "raw_stage_reduction": reduction, "counted": False,
        "experiment_leases": leases, "known_new_measured_experiment_seconds": sum(row["child_elapsed_seconds"] for row in leases),
        "self_review_current_lease_and_actual_exit_pending": True, "seconds": time.perf_counter()-started}
    atomic_json(args.output / "RESULT.json", receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k not in ("current_identities","experiment_leases")}), flush=True)
    if not all(checks.values()):
        raise AssertionError("The bounded negative screen failed an independent root reread check")


if __name__ == "__main__":
    main()
