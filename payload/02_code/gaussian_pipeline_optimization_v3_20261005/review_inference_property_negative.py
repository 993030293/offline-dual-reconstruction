"""Review complete property-cache failures, costs, original inputs, and owned exits."""

import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import psutil

STARTED = time.perf_counter()
ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parent
OWN = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/aw001"
sys.path.insert(0, str(CODE))
import run_inference_property_task as native


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def mounted(path):
    value = str(path).replace("\\", "/")
    return Path("C:/" + value[7:]) if value.startswith("/mnt/c/") else Path(path)


def main():
    output = OWN / "r0"
    output.mkdir(exist_ok=False)
    owner = psutil.Process()
    native.atomic_json(output / "PROCESS_IDENTITY.json", {"scope": "Windows", "pid": owner.pid, "creation_time": owner.create_time()})
    checks, pair_rows = {}, []
    for label in ("p0", "p1", "q0", "q1", "q2"):
        pair = read(OWN / label / "RESULT.json")
        compared, reductions = native.compare(OWN / label / "control", OWN / label / "cache")
        if compared != pair["checks"] or reductions != pair["actual_CUDA_property_element_work_reductions"]:
            raise AssertionError("A saved full native pair verdict differs from actual complete arrays")
        control_contract = read(OWN / label / "control/pair/evaluation/EVALUATION_CONTRACT.json")
        cache_contract = read(OWN / label / "cache/pair/evaluation/EVALUATION_CONTRACT.json")
        if control_contract != cache_contract:
            raise AssertionError("A complete evaluator calibration or image contract differs")
        pair_rows.append({"label": label, "status": pair["status"], "checks": compared, "work": reductions,
            "raw_whole_ratio": pair["raw_whole_ratio"], "sides": pair["sides"],
            "eligible_for_promotion": False, "reason": "pre_reservation_debug" if label.startswith("p") else
                "failed_whole_guard" if label == "q1" else "post_failure_accidental_diagnostic" if label == "q2" else "first_corrected_screen_only"})
    checks["five_actual_pairs_all240_complete_RGB_arrays_metrics_K_and_JPEG_exact"] = len(pair_rows) == 5
    checks["q1_original_raw_whole_guard_failed_and_preserved"] = (read(OWN / "q1/RESULT.json")["status"] == "failed_whole_screen"
        and read(OWN / "q1/RESULT.json")["raw_whole_ratio"] > 1.1 and read(OWN / "LEASE_Q1.json")["returncode"] == 1)
    checks["q2_accidental_launch_after_failed_q1_not_used_as_replacement"] = (read(OWN / "LEASE_Q2.json")["created_utc"]
        > read(OWN / "LEASE_Q1.json")["lease_released_utc"])
    checks["g0_old_generic_and_g1_fixed14_native_checks_preserved"] = (len(read(OWN / "g0/RESULT.json")["checks"]) == 12
        and all(read(OWN / "g0/RESULT.json")["checks"].values()) and len(read(OWN / "g1/RESULT.json")["checks"]) == 14
        and all(read(OWN / "g1/RESULT.json")["checks"].values()))
    checks["actual_capacity_failure_and_full_entry_reservation_regression_preserved"] = (read(OWN / "b0/RESULT.json")["status"]
        == "reproduced_missing_full_entry_reservation" and read(OWN / "g1/RESULT.json")["checks"]["two_model_interleaving_reserves_complete_entry_before_native_execution"])
    checks["both_actual17th_render_errors_preserved_and_closed"] = all(all(read(OWN / label / "RESULT.json")["checks"].values())
        and len(read(OWN / label / "cache/FAILURE.json")["render_rows"]) == 16 for label in ("f0", "f1"))
    checks["corrected_f1_then_actual_full_q1_recovery_science_passed"] = (read(OWN / "LEASE_Q1.json")["created_utc"]
        > read(OWN / "LEASE_F1.json")["lease_released_utc"] and all(read(OWN / "q1/RESULT.json")["checks"].values()))
    checks["old360_colour_math_component_only_not_current_external_model_or_GT"] = (read(OWN / "x0/RESULT.json")["status"] == "external360_component_passed"
        and len(read(OWN / "x0/linux/RESULT.json")["rows"]) == 360 and not (OWN / "x1").exists())
    all_originals = native.identities()
    expected = read(OWN / "q0/CONTRACT.json")["inputs"]
    checks["all31_original_full_input_sources_unchanged"] = all_originals == expected
    current_sources = {}
    for label in ("g1", "q0", "q1", "q2", "f1"):
        record = read(OWN / label / "RESULT.json")
        for row in record["sources"]:
            actual = native.data_identity(mounted(row["path"]))
            if actual["sha256"] != row["sha256"] or actual["bytes"] != row["bytes"]:
                raise AssertionError("A current fixed-version source differs from actual execution")
            current_sources[str(mounted(row["path"]))] = actual
    checks["all_corrected_execution_sources_match_current_version"] = True
    copies_checked = []
    for label in ("g0", "g1", "p0", "p1", "q0", "q1", "q2", "f0", "f1", "x0"):
        result = read(OWN / label / "RESULT.json")
        for index, row in enumerate(result["sources"]):
            copied = OWN / label / "source_copies" / (f"{index:02d}_" + mounted(row["path"]).name)
            actual = native.data_identity(copied)
            if actual["sha256"] != row["sha256"] or actual["bytes"] != row["bytes"]:
                raise AssertionError("An executed old or current source snapshot was overwritten")
            copies_checked.append(str(copied))
    checks["all_old_and_current_text_source_snapshots_preserved"] = bool(copies_checked)
    windows, linux_records, leases = {}, {}, []
    for path in OWN.rglob("PROCESS_IDENTITY.json"):
        if path.parent == output:
            continue
        row = read(path)
        group = linux_records if row.get("scope") == "WSL" else windows
        group[(row["pid"], row["creation_time"])] = row
    guard_paths = list(OWN.rglob("WINDOWS_GUARD.json")) + [OWN / "B0_GUARD.json"]
    for path in guard_paths:
        guard = read(path)
        if guard["status"] not in ("completed", "failed_child") or guard["elapsed_seconds"] > guard["maximum_seconds"] or guard["peak_tree_RSS_bytes"] > guard["maximum_RSS_bytes"]:
            raise AssertionError("An actual Windows task remains open or exceeded its budget")
        for row in guard["owned_process_identities"]:
            windows[(row["pid"], row["creation_time"])] = row
    for path in OWN.glob("LEASE_*.json"):
        if path.name == "LEASE_R0.json":
            continue
        lease = read(path)
        if lease["status"] != "child_exited" or not lease.get("lease_released_utc"):
            raise AssertionError("A recorded diagnostic lease remains live")
        for role in ("child", "guard"):
            row = {"pid": lease[role + "_pid"], "creation_time": lease[role + "_creation_time"]}
            windows[(row["pid"], row["creation_time"])] = row
        leases.append({"path": str(path), "seconds": lease["child_elapsed_seconds"], "returncode": lease["returncode"]})
    for row in windows.values():
        try:
            if psutil.Process(row["pid"]).create_time() == row["creation_time"]:
                raise AssertionError("An actual recorded owned Windows instance is still running")
        except psutil.NoSuchProcess:
            pass
    checks["all_saved_diagnostic_Windows_instances_exited"] = bool(windows)
    linux_input = output / "LINUX_RECORDS.json"
    native.atomic_json(linux_input, list(linux_records.values()))
    command = "import json,psutil,sys; records=json.load(open(sys.argv[1])); rows=[dict(r,alive=psutil.pid_exists(r['pid']) and abs(psutil.Process(r['pid']).create_time()-r['creation_time'])<.01 and psutil.Process(r['pid']).status()!='zombie') for r in records]; print(json.dumps(rows)); sys.exit(any(r['alive'] for r in rows))"
    completed = subprocess.run(["wsl", "-d", "E26-C2-LIV-20260907-134802", "--", "/opt/e26_thin_surface_v1/venv/bin/python", "-B", "-c", command,
        native.linux(linux_input)], capture_output=True, text=True, check=True, timeout=30)
    actual_linux = json.loads(completed.stdout)
    checks["all_saved_actual_Linux_PID_creation_instances_exited"] = len(actual_linux) == len(linux_records) and not any(row["alive"] for row in actual_linux)
    gpu_rows = []
    for path in OWN.rglob("GPU_MEMORY.json"):
        row = read(path)
        if row["errors"] or not row["NVML_closed"] or row["minimum_free_bytes"] < 2 * 1024**3:
            raise AssertionError("A recorded GPU resource or monitor lifecycle failed")
        gpu_rows.append({"path": str(path), "minimum_free_bytes": row["minimum_free_bytes"], "sample_count": len(row["samples"])})
    checks["all_observed_GPU_headroom_and_closed_NVML_guards_passed"] = bool(gpu_rows)
    wsl_rows = []
    for path in OWN.rglob("WSL_RESOURCE.json"):
        row = read(path)
        if row["elapsed_seconds"] > row["maximum_seconds"] or row["peak_tree_RSS_bytes"] > row["maximum_RSS_bytes"]:
            raise AssertionError("A native Linux task exceeded its original resource bounds")
        wsl_rows.append({"path": str(path), "status": row["status"], "peak_tree_RSS_bytes": row["peak_tree_RSS_bytes"]})
    checks["all_actual_Linux_RSS_and_time_budgets_passed"] = bool(wsl_rows)
    checks["owned_disk_scope_and_original_free_reserve"] = (sum(path.stat().st_size for path in OWN.rglob("*") if path.is_file()) <= 30 * 1024**3
        and shutil.disk_usage(OWN).free >= 20 * 1024**3)
    parent = read(ROOT / "research/2026-10-05-gaussian-optimization-round3-50/CURRENT_STATE.json")
    checks["original_parent24_unique_acceptances_preserved_no_R139_promotion"] = (parent["additional_accepted_count"] == 24
        and len(parent["accepted_mechanisms"]) == 24 and len(parent["acceptance_receipts"]) == 24
        and len(set(parent["accepted_mechanisms"])) == 24 and not any(name.startswith("R139") for name in parent["accepted_mechanisms"]))
    formal_sides = sum(sum(side["complete_whole_seconds"] for side in read(OWN / label / "RESULT.json")["sides"].values()) for label in ("q0", "q1"))
    all_elapsed = sum(row["seconds"] for row in leases)
    known_fee = 29.589292199956855 + all_elapsed - formal_sides
    result = {"status": "closed_negative_whole14point66_and_procedural_q2_diagnostic_only" if all(checks.values()) else "failed_review",
        "checks": checks, "pairs": pair_rows, "known_all_diagnostic_lease_child_seconds_before_review_exit": all_elapsed,
        "known_fee_lower_bound_before_review_exit": known_fee,
        "fee_scope": "old_R13929point589292_and_all_new_recorded_envelopes_minus_q0_q1_actual_sides_once_q2_accidental_whole_remains_fee_no_formal_three_claim",
        "unmetered_scope": "text_research_tool_startup_and_final_publication_not_assumed_zero",
        "source_input_records": list(current_sources.values()) + all_originals, "snapshots": copies_checked,
        "actual_Windows_exited": list(windows.values()), "actual_Linux_exited": actual_linux,
        "leases": leases, "GPU_records": gpu_rows, "Linux_resource_records": wsl_rows,
        "review_pid": owner.pid, "review_creation_time": owner.create_time(), "review_seconds_before_exit": time.perf_counter() - STARTED,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "new_optimization_count": 0,
        "review_kind": "separate_root_self_review_no_subagents", "whole_speed_5percent_accepted": False,
        "not_run": ["corrected_x1_external360", "protocol_valid_three_passed_formal_pairs", "new_training", "GT", "unseen_object_chain", "production_or_preview_deployment"]}
    native.atomic_json(output / "RESULT.json", result)
    print(json.dumps({"status": result["status"], "checks": checks, "known_fee_before_review": known_fee}), flush=True)
    if not all(checks.values()):
        raise AssertionError("The separate complete failure review found an unresolved issue")


if __name__ == "__main__":
    main()
