"""Close a measured candidate deferred by its remaining validation cost budget."""

import time

STARTED = time.perf_counter()

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import psutil

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/parallel_zstd_transport_001"
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def gone(row):
    try:
        return psutil.Process(row["pid"]).create_time() != row["creation_time"]
    except psutil.NoSuchProcess:
        return True


def as_windows(path):
    if not path.startswith("/mnt/c/"):
        raise ValueError("A frozen identity is outside the registered C drive")
    return Path("C:/" + path[len("/mnt/c/"):])


def main():
    if (OUT / "DEFERRED_CLOSURE_001.json").exists():
        raise FileExistsError("Never overwrite a deferred closure")
    screen, generic = read(OUT / "p0/RESULT.json"), read(OUT / "g1/RESULT.json")
    current, copies = {}, []
    for folder, value in [(OUT / "p0", screen), (OUT / "g1", generic)]:
        for i, r in enumerate(value["sources"]):
            current[r["path"]] = r
            raw = (folder / f"SOURCE_{i:02d}_{Path(r['path']).name}").read_bytes()
            copies.append(len(raw) == r["bytes"] and hashlib.sha256(raw).hexdigest() == r["sha256"])
    current.update({r["path"]: r for r in screen["inputs"]})
    current_checks = [all(data_identity(as_windows(r["path"]))[k] == r[k] for k in ["bytes", "sha256"]) for r in current.values()]
    windows, linux, leases = {}, {}, []
    for group in ["SCREEN", "GENERIC"]:
        lease = read(OUT / (group + "_LEASE_001.json"))
        win = read(OUT / (group + "_WINDOWS_GUARD_001.json"))
        lin = read(OUT / (group + "_LINUX_GUARD_001.json"))
        leases.append(lease)
        for r in win["owned_process_identities"] + [dict(pid=lease[k + "_pid"], creation_time=lease[k + "_creation_time"]) for k in ["guard", "child"]]:
            windows[(r["pid"], r["creation_time"])] = dict(r, gone=gone(r))
        for r in lin["owned_process_identities"]:
            linux[(r["pid"], r["creation_time"])] = r
        if not (win["status"] == lin["status"] == "completed" and win["returncode"] == lin["returncode"] == lease["returncode"] == 0):
            raise AssertionError("A recorded measured task is not terminal")
    code = "import json,psutil; rows=json.loads(" + repr(json.dumps(list(linux.values()))) + "); out=[]\n"
    code += "for r in rows:\n alive=False\n try:\n  p=psutil.Process(r['pid']); alive=p.create_time()==r['creation_time'] and p.status()!=psutil.STATUS_ZOMBIE\n except psutil.NoSuchProcess: pass\n out.append(dict(r,alive=alive))\nprint(json.dumps(out))"
    live = json.loads(subprocess.check_output(["wsl", "-d", "E26-C2-LIV-20260907-134802", "--",
        "/opt/e26_thin_surface_v1/venv/bin/python", "-B", "-c", code], text=True, timeout=30))
    source_checks = [v for k, v in screen["checks"].items() if "actual_bounded_parallel_write_and_join" not in k]
    recovery = [read(OUT.parent / "zstd_recovery_001" / name) for name in ["resume_process.json", "evaluation_process.json"]]
    historical_seconds = sum(r["elapsed_seconds"] for r in recovery)
    old_fees = read(OUT.parent / "lz4_transport_001/NEGATIVE_CLOSURE_001.json")["all_measured_nonoverlapping_screen_and_closure_seconds"]
    raw = {mode: screen["complete_three_source_sums_seconds"][mode]["complete_encode_seconds"] for mode in ["serial", "pipeline"]}
    saving = raw["serial"] - raw["pipeline"]
    checks = {
        "all_original_actual_science_speed_and_whole_screen_guards": all(source_checks),
        "lazy_worker_correction_and_eighteen_regression_checks": len(generic["checks"]) == 18 and all(generic["checks"].values()),
        "all_current_nine_sources_inputs_and_eight_recorded_source_copies": len(current_checks) == 9 and all(current_checks) and len(copies) == 8 and all(copies),
        "actual_Windows_instances_gone": all(r["gone"] for r in windows.values()),
        "actual_Linux_child_instances_gone": all(not r["alive"] for r in live),
        "both_measured_group_leases_released": all(bool(r.get("lease_released_utc")) for r in leases),
        "historical_recovery_reference_is_actual_terminal_not_new_execution": all(r["status"] == "completed" and r["returncode"] == 0 for r in recovery),
        "no_formal_or_new_native_recovery_or_acceptance_claim": screen["new_optimization_count"] == generic["new_optimization_count"] == 0
            and not (OUT / "ACCEPTANCE.json").exists(),
    }
    close_seconds = time.perf_counter() - STARTED
    known_fees = old_fees + sum(r["child_elapsed_seconds"] for r in leases) + close_seconds
    limit = 6 * (saving / .05 - raw["serial"])
    projected = 1 - (raw["pipeline"] + (known_fees + historical_seconds) / 6) / (raw["serial"] + (known_fees + historical_seconds) / 6)
    checks["historical_cost_based_projection_below5percent_requires_deferral"] = projected < .05
    result = {
        "status": "closed_deferred_cost_ranking_count0" if all(checks.values()) else "failed_close",
        "checks": checks, "new_optimization_count": 0,
        "actual_single_pass_encoding_reduction": screen["encoding_reduction"],
        "actual_single_pass_whole_transport_reduction": screen["whole_reduction"],
        "actual_complete_encoding_single_pass_seconds": raw,
        "all_known_measured_first_screen_generic_and_closure_fees_seconds": known_fees,
        "included_previous_R136_measured_exploration_seconds": old_fees,
        "closing_seconds_before_publication": close_seconds,
        "hypothetical_max_total_shared_fees_to_keep5percent_at_screen_times": limit,
        "hypothetical_remaining_fee_budget_seconds": limit - known_fees,
        "historical_actual_same_prefix_recovery_evaluation_seconds_not_new_run": historical_seconds,
        "projected_reduction_if_same_historical_recovery_cost_not_actual_formal_result": projected,
        "projection_excludes_unexecuted_formal_fault_and_final_review_costs": True,
        "single_pass_not_formal_median_or_cost_acceptance": True,
        "formal_full_faults_actual_new_training_recovery_external_GT_preview_production": "not_run",
        "source_review": dict(current_identity_count=len(current), saved_source_copy_count=len(copies)),
        "Windows": list(windows.values()), "Linux": live,
        "root_single_chat_self_review_no_subagent": True,
        "closing_script_own_pid": psutil.Process().pid,
        "closing_script_own_creation_time": psutil.Process().create_time(),
        "own_closing_script_exit_requires_outer_check": True,
        "small_final_text_publication_after_close_not_claimed_included": True,
    }
    atomic_json(OUT / "DEFERRED_CLOSURE_001.json", result)
    if not all(checks.values()):
        raise AssertionError("A deferred screen evidence, cost or closure gate failed")
    print(json.dumps({k: result[k] for k in ["status", "checks", "all_known_measured_first_screen_generic_and_closure_fees_seconds", "hypothetical_remaining_fee_budget_seconds", "projected_reduction_if_same_historical_recovery_cost_not_actual_formal_result"]}), flush=True)


if __name__ == "__main__":
    main()
