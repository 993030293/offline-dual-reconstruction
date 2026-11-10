"""Reconcile actual PLY evidence, process exits, sources, copies and complete fees."""

import time

STARTED = time.perf_counter()

import argparse
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

import psutil

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parent
BASE = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005"
OUT = BASE / "hc001"
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def windows(path):
    value = str(path)
    return Path("C:/" + value[7:]) if value.startswith("/mnt/c/") else Path(value)


def gone(row):
    try:
        return psutil.Process(row["pid"]).create_time() != row["creation_time"]
    except psutil.NoSuchProcess:
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.resolve().relative_to(OUT.resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    campaign = read(OUT / "c0/RESULT.json")
    generic = read(OUT / "g0/HOST_GENERIC_RESULT.json")
    recovery = read(OUT / "r0/RESULT.json")
    profile = read(BASE / "dp001/p0/RESULT.json")
    contracts = [BASE / "dp001/p0/CONTRACT.json", OUT / "g0/CONTRACT.json", OUT / "c0/CONTRACT.json"]
    contracts += [
        OUT / "c0" / str(index) / mode / "CONTRACT.json" for index in range(3) for mode in ("cpu", "host")]
    identities = {}
    copies = []
    native_copies = []
    for path in contracts:
        rows = read(path)["sources_inputs"]
        for row in rows:
            identities[(str(windows(row["path"])), row["sha256"])] = row
        folder = path.parent / "source_copies"
        if folder.is_dir():
            for copy in sorted(folder.iterdir()):
                index = int(copy.name.split("_", 1)[0])
                source_path = rows[index]["path"]
                if source_path.startswith("/") and not source_path.startswith("/mnt/c/"):
                    copies.append({"copy": str(copy), "source": source_path, "exact": None})
                    native_copies.append({"copy": "/mnt/c/" + str(copy.relative_to(Path("C:/"))).replace("\\", "/"),
                        "index": len(copies) - 1, "source": source_path})
                    continue
                original = windows(rows[index]["path"])
                copies.append({"copy": str(copy), "source": str(original),
                    "exact": copy.read_bytes() == original.read_bytes()})
    for row in recovery["sources"] + recovery["original_fallback_result"]["sources"]:
        identities[(str(windows(row["path"])), row["sha256"])] = row
    identities_checked = []
    native_identities = []
    for row in identities.values():
        if row["path"].startswith("/") and not row["path"].startswith("/mnt/c/"):
            native_identities.append(row)
            continue
        current = data_identity(windows(row["path"]))
        identities_checked.append({**row, "current_exact": current["sha256"] == row["sha256"] and current["bytes"] == row["bytes"]})
    leases = [BASE / "dp001/PROFILE_LEASE_001.json", OUT / "GENERIC_LEASE_001.json",
        OUT / "CAMPAIGN_LEASE_001.json", OUT / "RECOVERY_LEASE_001.json"]
    failed_reviews = [OUT / "REVIEW_LEASE_001.json", OUT / "REVIEW_LEASE_002.json"]
    guards = [BASE / "dp001/PROFILE_GUARD_001.json", OUT / "GENERIC_GUARD_001.json",
        OUT / "CAMPAIGN_GUARD_001.json", OUT / "RECOVERY_GUARD_001.json",
        OUT / "REVIEW_GUARD_001.json", OUT / "REVIEW_GUARD_002.json"] + sorted((OUT / "c0").glob("*_GUARD.json"))
    win = {}
    for path in guards:
        for row in read(path).get("owned_process_identities", []):
            win[(row["pid"], row["creation_time"])] = row
    for path in leases + failed_reviews:
        row = read(path)
        for pid, birth in ((row["guard_pid"], row["guard_creation_time"]), (row["child_pid"], row["child_creation_time"])):
            win[(pid, birth)] = {"pid": pid, "creation_time": birth}
    host_resources = [read(path) for path in sorted(OUT.rglob("HOST_IO_RESOURCE.json"))]
    for row in host_resources:
        win[(row["pid"], row["creation_time"])] = {"pid": row["pid"], "creation_time": row["creation_time"]}
    win_rows = [{**row, "gone": gone(row)} for row in win.values()]
    linux_resources = [read(path) for path in [BASE / "dp001/p0/WSL_RESOURCE.json"] + sorted(OUT.rglob("WSL_RESOURCE.json"))]
    linux = {(row["pid"], row["creation_time"]): {"pid": row["pid"], "creation_time": row["creation_time"]} for row in linux_resources}
    script = "import json,psutil,hashlib,pathlib\nrows=json.loads(" + repr(json.dumps(list(linux.values()))) + ")\nresult=[]\n"
    script += "for row in rows:\n try:\n  p=psutil.Process(row['pid']); alive=p.create_time()==row['creation_time'] and p.status()!=psutil.STATUS_ZOMBIE\n except psutil.NoSuchProcess: alive=False\n result.append(dict(row,gone=not alive))\n"
    script += "identities=json.loads(" + repr(json.dumps(native_identities)) + ")\nidentity_results=[]\n"
    script += "for row in identities:\n data=pathlib.Path(row['path']).read_bytes(); identity_results.append(dict(row,current_exact=len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']))\n"
    script += "copies=json.loads(" + repr(json.dumps(native_copies)) + ")\ncopy_results=[]\n"
    script += "for row in copies:\n copy_results.append(dict(row,exact=pathlib.Path(row['copy']).read_bytes()==pathlib.Path(row['source']).read_bytes()))\n"
    script += "print(json.dumps(dict(processes=result,identities=identity_results,copies=copy_results)))\n"
    peer = json.loads(subprocess.check_output(["wsl", "-d", "E26-C2-LIV-20260907-134802",
        "/opt/e26_thin_surface_v1/venv/bin/python", "-B", "-c", script], text=True, timeout=30))
    linux_rows = peer["processes"]
    identities_checked.extend(peer["identities"])
    for row in peer["copies"]:
        copies[row["index"]]["exact"] = row["exact"]
    saved = [read(OUT / "c0" / str(index) / mode / "RESULT.json") for index in range(3) for mode in ("cpu", "host")]
    raw = {mode: {key: statistics.median(row[key] for row in campaign["rows"] if row["mode"] == mode)
        for key in ("complete_phase_seconds", "complete_whole_seconds")} for mode in ("cpu", "host")}
    first = sum(read(path)["child_elapsed_seconds"] for path in [leases[0], leases[1], leases[3]] + failed_reviews)
    row_whole = sum(row["complete_whole_seconds"] for row in campaign["rows"])
    parent_extra = read(leases[2])["child_elapsed_seconds"] - row_whole
    if parent_extra < 0:
        raise AssertionError("The campaign envelope must include every complete side and parent comparison")
    writer = next(row for row in profile["profile_calls"] if row["file"].endswith("/ply_serialization.py") and row["function"] == "save_ply")
    write = next(row for row in profile["profile_calls"] if row["file"].endswith("/plyfile.py") and row["function"] == "write")
    assembly_envelope = writer["cumulative_seconds"] - write["cumulative_seconds"]
    r126_screen = {"status": "source_cost_diagnostic_defer_device_assembly_no_algorithm_implementation",
        "profile_only_not_formal_acceptance": True, "profiled_current_complete_save_verified": all(profile["checks"].values()),
        "maximum_profiled_writer_non_file_work_seconds": assembly_envelope,
        "writer_non_file_work_includes_assembly_transfer_and_other_work": True,
        "required_first_diagnostic_seconds": read(leases[0])["child_elapsed_seconds"],
        "diagnostic_speed_envelope_after_only_first_fee": assembly_envelope / (
            profile["complete_diagnostic_phase_seconds"] + read(leases[0])["child_elapsed_seconds"] / 6),
        "logical_removable_host_blocks_not_measured_candidate_RSS_bytes": profile["logical_full_matrix_bytes_not_memory_metric"],
        "diagnostic_memory_fraction_of_observed_current_peak": profile["logical_full_matrix_bytes_not_memory_metric"] /
            profile["observed_peak_before_original_tuple_oracle_bytes"],
        "device_implementation_three_pairs_external_training_and_recovery": "not_run", "new_optimization_count": 0}
    checks = {
        "three_fresh_original_alternating_pairs_complete": [(row["pair"], row["mode"]) for row in campaign["rows"]] ==
            [(0, "cpu"), (0, "host"), (1, "host"), (1, "cpu"), (2, "cpu"), (2, "host")],
        "all_six_complete_parameter_RNG_loader_and_original_byte_guards": len(saved) == 6 and all(all(row["checks"].values()) for row in saved),
        "all_three_complete_physical_artifact_pairs_exact_and_within_whole_cap": all(row["complete_two_file_bytes_equal"] and row["within_original_whole_cap"] for row in campaign["pairs"]),
        "original48_generic_and12_actual_host_workers": generic["status"] == "host_generic_passed" and all(generic["checks"].values()),
        "original12_fallback_and_four_actual_host_interrupt_recovery_rows": recovery["status"] == "host_cancel_fallback_permission_and_recovery_verified" and all(recovery["checks"].values()),
        "all_current_complete_sources_and_inputs_exact": all(row["current_exact"] for row in identities_checked),
        "all_current_source_copies_exact": len(copies) == 86 and all(row["exact"] for row in copies),
        "all_recorded_actual_Windows_and_Linux_instances_gone": all(row["gone"] for row in win_rows + linux_rows),
        "all_original_Windows_Linux_and_CUDA_resource_limits": all(read(path)["peak_tree_RSS_bytes"] <= 8 * 1024**3 for path in guards)
            and all(row["status"] == "completed" and row["peak_tree_RSS_bytes"] <= 32 * 1024**3 for row in linux_resources)
            and all(row["status"] == "completed" and row["peak_RSS_bytes"] <= 8 * 1024**3 and row["elapsed_seconds"] <= 120 for row in host_resources)
            and all(row["CUDA_peak_reserved_bytes"] <= 12 * 1024**3 for row in saved),
        "all_four_completed_experiment_leases_released": all(read(path)["returncode"] == 0 and read(path).get("lease_released_utc") for path in leases),
        "disk_group_and_free_reserve_preserved": sum(path.stat().st_size for path in OUT.rglob("*") if path.is_file()) < 30 * 1024**3
            and shutil.disk_usage(OUT).free >= 20 * 1024**3,
        "R126_old_tuple_cost_not_counted_as_new_assembly_benefit": r126_screen["new_optimization_count"] == 0,
        "parent50_active_and_no_prior_accepted_mechanism_recount": read(ROOT / "research/2026-10-05-gaussian-optimization-round3-50/CURRENT_STATE.json")["additional_accepted_count"] == 18,
        "raw_costs_recalculated_and_first_fees_not_omitted": raw == campaign["raw_medians_not_acceptance"] and first > 0 and parent_extra > 0}
    result = {"status": "required_host_save_review_passed_pending_final_complete_cost" if all(checks.values()) else "failed_review",
        "review_kind": "root_single_chat_self_review_no_subagents", "checks": checks,
        "current_identities": identities_checked, "source_copies": copies, "Windows": win_rows, "Linux": linux_rows,
        "host_resources": host_resources, "Linux_resources": linux_resources,
        "raw_medians": raw, "all_first_diagnostics_recovery_seconds": first,
        "campaign_envelope_extra_seconds": parent_extra, "campaign_side_whole_seconds_sum": row_whole,
        "R126_diagnostic_screen": r126_screen, "review_before_publication_seconds": time.perf_counter() - STARTED,
        "unverified": ["other_object_complete_Gaussian_model", "other_host_or_filesystem", "Windows_ACL_change_during_dispatched_copy",
            "full_new_training", "GT", "production_deployment", "new_browser_preview"], "new_optimization_count": 0}
    atomic_json(args.output / "REVIEW.json", result)
    atomic_json(BASE / "dp001/SOURCE_COST_SCREEN_001.json", r126_screen)
    print(json.dumps({"status": result["status"], "checks": checks, "identities": len(identities_checked),
        "source_copies": len(copies), "Windows": len(win_rows), "Linux": len(linux_rows),
        "first_seconds": first, "campaign_extra_seconds": parent_extra}), flush=True)
    if not all(checks.values()):
        raise AssertionError("A required complete host save self-review failed")


if __name__ == "__main__":
    main()
