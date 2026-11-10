"""Separately review complete outputs, source lineage, costs, and actual closure."""

import time

STARTED = time.perf_counter()

import json
import msvcrt
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

import psutil

from run_cpu_dependency_restore import compare, read, ROOT, CODE, atomic_json, data_identity

OWN = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/nrc001"
DOC = ROOT / "research/2026-10-05-gaussian-optimization-round3-50"


def windows(path):
    value = str(path)
    return Path("C:/" + value[7:]) if value.startswith("/mnt/c/") else Path(value)


def closed(identity):
    try:
        return psutil.Process(identity["pid"]).create_time() != identity["creation_time"]
    except psutil.NoSuchProcess:
        return True


def calculate_cost(extra_seconds=0.):
    formal = read(OWN / "c0/RESULT.json")
    old = read(ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/gsf001/bootstrap_dag_001/FINAL_SCREEN_PUBLICATION_001.json")
    leases = {name: read(OWN / (name + "_lease.json")) for name in ("p0", "c0", "r0")}
    formal_seconds = sum(row["whole_complete_entry_seconds"] for row in formal["rows"])
    residual = leases["c0"]["child_elapsed_seconds"] - formal_seconds
    if residual < 0:
        raise AssertionError("The actual formal envelope cannot be double charged")
    components = {
        "old_R142_measured_family_and_known_last_publication_seconds": old["known_measured_family_computational_envelopes_seconds"]
            + old["publication_before_last_serialization_seconds"],
        "actual_native_restore_diagnostic_lease_seconds": leases["p0"]["child_elapsed_seconds"],
        "actual_formal_controller_residual_not_six_side_recharge_seconds": residual,
        "actual_generic_faults_and_complete_recovery_lease_seconds": leases["r0"]["child_elapsed_seconds"],
        "known_native_source_read_and_metadata_command_extra_allowance_not_actual_seconds": 15.,
        "root_review_and_publisher_actual_so_far_seconds": extra_seconds,
        "post_exit_native_text_and_counter_publication_reserved_seconds": 5.,
    }
    total = sum(components.values())
    surcharge = total / 6
    medians = {mode: statistics.median(row["whole_complete_entry_seconds"] for row in formal["rows"] if row["mode"] == mode)
        for mode in ("incumbent", "bootstrap")}
    raw = medians["bootstrap"] / medians["incumbent"]
    adjusted = (medians["bootstrap"] + surcharge) / (medians["incumbent"] + surcharge)
    before_V6 = components["old_R142_measured_family_and_known_last_publication_seconds"] + components["actual_native_restore_diagnostic_lease_seconds"] + residual
    V5_upper = 1 - (medians["bootstrap"] + before_V6 / 6) / (medians["incumbent"] + before_V6 / 6)
    return {"components": components, "known_extra_execution_seconds_with_declared_allowances": total,
        "six_side_common_seconds_surcharge": surcharge, "formal_raw_median_seconds": medians,
        "raw_complete_entry_reduction": 1 - raw, "net_complete_entry_reduction_under_registered_six_side_model": 1 - adjusted,
        "raw_whole_median_ratio": raw, "adjusted_whole_median_ratio": adjusted,
        "each_original_whole_ratios": [pair["whole_ratio"] for pair in formal["pairs"]],
        "old_V5_net3percent_upper_before_V6_completion": V5_upper,
        "declared_allowances_are_not_full_measured_research_costs": True,
        "unknown_text_research_and_old_unrecorded_time_not_zero_no_research_ROI": True}


def main():
    if (OWN / "SELF_REVIEW_V6_001.json").exists():
        raise FileExistsError("The actual root review must use a unique receipt")
    review_source = data_identity(Path(__file__).resolve())
    shutil.copyfile(__file__, OWN / "EXECUTED_V6_REVIEW_SOURCE.py")
    parent = read(DOC / "CURRENT_STATE.json")
    formal, recovery, generic = (read(OWN / name) for name in ("c0/RESULT.json", "r0/RESULT.json", "r0/GENERIC_RESULT.json"))
    folders = [OWN / "p0", *[OWN / "c0" / f"t{trial}" / mode for trial in range(3) for mode in ("incumbent", "bootstrap")],
        OWN / "r0/checksum_error", OWN / "r0/fresh_complete_recovery"]
    identities, copies = {}, 0
    for folder in folders:
        contract = read(folder / "CONTRACT.json")
        for row in contract["sources"] + contract["inputs"]:
            if row["path"] in identities and identities[row["path"]] != row:
                raise AssertionError("A source or complete input differs across actual executions")
            identities[row["path"]] = row
        for index, row in enumerate(contract["sources"]):
            saved = folder / "source_copies" / (f"{index:02d}_" + Path(row["path"]).name)
            if data_identity(saved)["sha256"] != row["sha256"]:
                raise AssertionError("An actual executed source copy differs")
            copies += 1
    for row in read(OWN / "r0/CONTRACT.json")["sources"]:
        if data_identity(OWN / "r0/source_copies" / Path(row["path"]).name)["sha256"] != row["sha256"]:
            raise AssertionError("An executed completion controller copy differs")
        identities[row["path"]] = row
        copies += 1
    for row in read(OWN / "c0/CONTRACT.json")["sources"]:
        identities[row["path"]] = row
    current = [row for row in identities.values() if not row["path"].startswith("/opt/")]
    if any(data_identity(windows(row["path"]))["sha256"] != row["sha256"] for row in current):
        raise AssertionError("A complete original source or input changed")
    linux_ids = [{"pid": value["pid"], "creation_time": value["creation_time"]}
        for value in (read(path) for path in sorted(OWN.rglob("WSL_RESOURCE.json")))]
    native_sources = [row for row in identities.values() if row["path"].startswith("/opt/")]
    probe = ("import json,psutil,sys; sys.path.insert(0,'/mnt/c/lidar-camera-fusion-mask-robust-downsampling/02_code/gaussian_pipeline_optimization_v1_20261005'); from runtime import data_identity\n"
        "rows=" + repr(linux_ids) + "; sources=" + repr(native_sources) + "; out=[]\n"
        "for row in rows:\n"
        " try: done=psutil.Process(row['pid']).create_time()!=row['creation_time']\n"
        " except psutil.NoSuchProcess: done=True\n"
        " out.append(dict(row,instance_closed=done))\n"
        "print(json.dumps({'instances':out,'native_sources_exact':all(data_identity(row['path'])==row for row in sources)}))")
    completed = subprocess.run(["wsl", "-d", "E26-C2-LIV-20260907-134802", "--",
        "/opt/e26_thin_surface_v1/venv/bin/python", "-B", "-c", probe], check=True, text=True, capture_output=True, timeout=30)
    native = json.loads(completed.stdout)
    windows_ids = {}
    for path in sorted(OWN.rglob("*_GUARD.json")):
        for row in read(path).get("owned_process_identities", []):
            windows_ids[(row["pid"], row["creation_time"])] = row
    leases = [read(OWN / (name + "_lease.json")) for name in ("p0", "c0", "r0")]
    for lease in leases:
        for prefix in ("guard", "child"):
            row = {"pid": lease[prefix + "_pid"], "creation_time": lease[prefix + "_creation_time"]}
            windows_ids[(row["pid"], row["creation_time"])] = row
    lock = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/coordination_001/exclusive_benchmark.lock"
    with lock.open("r+b", buffering=0) as stream:
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
    normal = [read(folder / "RESULT.json") for folder in folders if folder.name != "checksum_error"]
    fault = [read(OWN / "r0" / name / "FAULT_RESULT.json") for name in ("checksum_error", "actual_cancel")]
    originals = (OWN / "r0/checksum_error/checkpoint_001000.pth").read_bytes()
    control = (OWN / "p0/checkpoint_001000.pth").read_bytes()
    one_byte_corruption = len(originals) == len(control) == 97079305 and originals[0] != control[0] and originals[1:] == control[1:]
    bootstrap_normals = [row for row in normal if row.get("mode") == "bootstrap"]
    resource_paths = [*OWN.rglob("WSL_RESOURCE.json"), *OWN.rglob("CPU_BOOTSTRAP_RESOURCE.json"), *OWN.rglob("*_GUARD.json")]
    resources = [read(path) for path in resource_paths]
    cost = calculate_cost(time.perf_counter() - STARTED)
    checks = {
        "all_three_fresh_complete1016_outputs_and_scene_files_recompared": all(all(compare(OWN / "c0" / f"t{trial}" / "incumbent",
            OWN / "c0" / f"t{trial}" / "bootstrap").values()) for trial in range(3)),
        "all_eight_normal_native_original_model_RNG_sampling_and_input_checks": len(normal) == 8 and all(all(row["checks"].values()) for row in normal),
        "both_real_checksum_and_SIGINT_failures_propagated_and_closed": len(fault) == 2 and all(all(row["checks"].values()) for row in fault),
        "corrupt_private_copy_exactly_one_byte_original_input_untouched": one_byte_corruption,
        "actual_signal_has_live_owned_worker": read(OWN / "r0/actual_cancel/ACTUAL_SIGNAL.json")["signal"] == 2
            and bool(read(OWN / "r0/actual_cancel/ACTUAL_SIGNAL.json")["live_owned_workers"]),
        "fresh_complete_recovery_original_controls_exact": all(compare(OWN / "c0/t0/incumbent", OWN / "r0/fresh_complete_recovery").values()),
        "all_seven_generic_interfaces_not_mislabelled_native_or_external": len(generic["checks"]) == 7 and all(generic["checks"].values())
            and generic["fixtures_not_native_Torch_or_external_object_evidence"],
        "four_actual_native_CPU_bootstrap_workers_non_GPU_and_joined": len(bootstrap_normals) == 4 and all(row["CPU_dependency"]["owned_worker_joined"]
            and row["CPU_dependency"]["native_import_calls"] == 1 and not row["CPU_dependency"]["actual_import"]["CUDA_context_initialized"]
            and row["CPU_dependency"]["actual_import"]["thread_ident"] != row["CPU_dependency"]["started_parent_thread_ident"]
            for row in bootstrap_normals),
        "all_current_sources_complete_original_inputs_and_executed_copies_exact": native["native_sources_exact"],
        "all_actual_Windows_instances_closed": all(closed(row) for row in windows_ids.values()),
        "all_ten_actual_Linux_instances_closed": len(linux_ids) == 10 and all(row["instance_closed"] for row in native["instances"]),
        "three_original_leases_released_and_native_OS_lock_reacquired": all(lease["returncode"] == 0 and lease.get("lease_released_utc") for lease in leases),
        "parent31_unique_target50_active_and_no_old_R142_acceptance": parent["additional_accepted_count"] == 31
            and len(parent["accepted_mechanisms"]) == len(set(parent["accepted_mechanisms"])) == 31
            and parent["additional_target"] == 50 and parent["status"].startswith("active")
            and not any(row.startswith("R142") for row in parent["accepted_mechanisms"]),
        "V5_old_failure_preserved_V6_uniform2percent_cost_passed": cost["old_V5_net3percent_upper_before_V6_completion"] < .03
            and cost["net_complete_entry_reduction_under_registered_six_side_model"] >= .02,
        "raw_and_adjusted_whole_protections_unchanged": cost["raw_whole_median_ratio"] <= 1.01
            and cost["adjusted_whole_median_ratio"] <= 1.01 and all(value <= 1.1 for value in cost["each_original_whole_ratios"]),
        "original_RSS_CUDA_time_output_and_disk_budgets": all(row["peak_tree_RSS_bytes"] <= row["maximum_RSS_bytes"]
            and row["elapsed_seconds"] <= row["maximum_seconds"] for row in resources)
            and all(row["peak_CUDA_reserved_bytes"] <= row["CUDA_budget_bytes"] for row in normal)
            and sum(path.stat().st_size for path in OWN.rglob("*") if path.is_file()) <= 2 * 1024**3
            and shutil.disk_usage(OWN).free >= 20 * 1024**3,
    }
    receipt = {"status": "root_V6_self_review_passed" if all(checks.values()) else "failed_root_V6_review", "checks": checks,
        "cost": cost, "source_and_complete_input_identities": len(identities), "executed_source_copies": copies,
        "Windows_instances": list(windows_ids.values()), "Linux_instances": native["instances"],
        "root_review_is_independent_agent": False, "new_optimization_count": 0,
        "actual_executed_review_source": review_source,
        "reviewer": {"pid": psutil.Process().pid, "creation_time": psutil.Process().create_time()},
        "actual_review_seconds_before_publication": time.perf_counter() - STARTED}
    atomic_json(OWN / "SELF_REVIEW_V6_001.json", receipt)
    atomic_json(OWN / "V5_COST_SCREEN_001.json", {"status": "V5_net3percent_failed_preserved", "new_optimization_count": 0,
        "before_V6_completion_upper_bound": cost["old_V5_net3percent_upper_before_V6_completion"],
        "subsequent_user_authorized_V6_reassessment_is_separate": True})
    if not all(checks.values()):
        raise AssertionError({key: value for key, value in checks.items() if not value})
    print(json.dumps({"status": receipt["status"], "checks": len(checks), "net_V6_reduction": cost[
        "net_complete_entry_reduction_under_registered_six_side_model"]}), flush=True)


if __name__ == "__main__":
    main()
