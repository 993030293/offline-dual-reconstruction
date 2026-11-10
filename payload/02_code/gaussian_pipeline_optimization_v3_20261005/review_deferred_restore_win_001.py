"""Separately review full results, known costs and Windows process closure."""

import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

import psutil

from run_cpu_dependency_restore import ROOT, CODE, compare, read, data_identity
from run_deferred_restore_equal_pair_001 import BASE, save, gpu


def main():
    output = Path(sys.argv[1]).resolve()
    output.relative_to(BASE)
    output.mkdir(exist_ok=False)
    rs = [read(BASE / name / "RESULT.json") for name in ("d1", "d2", "d3")]
    before = [next(e["whole_complete_side_seconds"] for e in r["entries"] if e["mode"] == "native") for r in rs]
    after = [next(e["whole_complete_side_seconds"] for e in r["entries"] if e["mode"] == "deferred") for r in rs]
    ratios = [c / b for b, c in zip(before, after)]
    old = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005"
    fees = [{"reason": "known_related_R98_original_complete_guard", "seconds": read(old / "deferred_telemetry_001_process.json")["elapsed_seconds"],
        "receipt": str(old / "deferred_telemetry_001_process.json")},
        {"reason": "013_parent_missing_actual6e3c42", "seconds": .1881376},
        {"reason": "014_native_environment_preflight_actual4b77f5", "seconds": 1.2952799},
        {"reason": "014_fixed_official_Torch_source_and_LICENSE_fetch_actual1fc0b9", "seconds": 1.3262455},
        {"reason": "preregistered_metadata_source_root_and_external_close_reserve", "seconds": 25.0}]
    for name in ("s1", "a0"):
        path = old / "oup013" / (name + "_GUARD.json")
        fees.append({"reason": "known_original_source013_" + name, "seconds": read(path)["elapsed_seconds"], "receipt": str(path)})
    leases = [old / "oup014/p0_LEASE.json", BASE / "review_native0_LEASE.json"]
    leases += [BASE / (name + "_LEASE.json") for name in (
        "generic0", "g0", "c1", "c1b", "c2", "c3", "life0", "life1", "generic1", "generic2", "x0", "r1")]
    for path in leases:
        row = read(path)
        if not row.get("lease_released_utc") or row.get("status") != "child_exited":
            raise AssertionError("A related whole execution did not actually close")
        fees.append({"reason": "related_complete_nonformal_execution", "seconds": row["child_elapsed_seconds"], "receipt": str(path), "returncode": row["returncode"]})
    for name, b, c in zip(("d1", "d2", "d3"), before, after):
        path = BASE / (name + "_LEASE.json")
        row = read(path)
        if row["returncode"] != 0 or not row.get("lease_released_utc"):
            raise AssertionError("The six formal timed entries did not close")
        fees.append({"reason": "formal_outer_source_oracle_and_publish_residual_only", "seconds": max(0., row["child_elapsed_seconds"] - b - c), "receipt": str(path)})
        leases.append(path)
    fee = sum(row["seconds"] for row in fees)
    baseline, candidate = statistics.median(before), statistics.median(after)
    fee_side = fee / 6
    net_work = 20 / (22 + fee_side * 22 / baseline)
    net_speed = (baseline - candidate) / (baseline + fee_side)
    native = read(BASE / "review_native0/RESULT.json")
    native_writer_code = '''import json,psutil
from pathlib import Path
row=json.loads(Path("/mnt/c/lidar-camera-fusion-mask-robust-downsampling/04_outputs/gaussian_pipeline_optimization_v3_20261005/oup014/R168/review_native0/RESULT.json").read_text())["writer_identity"]
try:
 p=psutil.Process(row["pid"])
 closed=p.create_time()!=row["creation_time"] or not p.is_running()
except psutil.NoSuchProcess:
 closed=True
print(json.dumps({"saved_native_writer":row,"saved_instance_ended":closed}))
assert closed
'''
    native_writer = subprocess.run(["wsl", "-d", "E26-C2-LIV-20260907-134802", "--",
        "/opt/e26_thin_surface_v1/venv/bin/python", "-B", "-c", native_writer_code],
        check=True, timeout=15, text=True, encoding="utf-8", stdout=subprocess.PIPE,
        creationflags=subprocess.CREATE_NO_WINDOW)
    native_writer_status = json.loads(native_writer.stdout)
    generic = read(BASE / "generic2/RESULT.json")
    fault = read(BASE / "x0/RESULT.json")
    mismatch = read(BASE / "x0/cancel_deferred/first_mismatch/MISMATCH_PATH.json")
    fresh = read(BASE / "r1/RESULT.json")
    fresh_compare = compare(BASE / "d3/native", BASE / "r1")
    saved = {}
    for path in leases:
        row = read(path)
        for prefix in ("guard", "child"):
            if prefix + "_creation_time" in row:
                saved[(row[prefix + "_pid"], row[prefix + "_creation_time"])] = str(path)
    for scope in (BASE.parent, old / "oup013"):
        for path in scope.rglob("*_GUARD.json"):
            for row in read(path).get("owned_process_identities", []):
                saved[(row["pid"], row["creation_time"])] = str(path)
    ended = []
    for (pid, born), source in saved.items():
        try:
            process = psutil.Process(pid)
            closed = process.create_time() != born or not process.is_running()
        except psutil.NoSuchProcess:
            closed = True
        ended.append({"pid": pid, "creation_time": born, "saved_instance_ended": closed, "source": source})
    protocols = [read(BASE / name / "PROTOCOL.json") for name in ("d1", "d2", "d3")]
    sources_kept = all(p["sources"] == [data_identity(row["path"]) for row in p["sources"]] for p in protocols)
    master = read(ROOT / "research/2026-10-05-gaussian-optimization-round3-50/CURRENT_STATE.json")
    mechanism_names = master["accepted_mechanisms"]
    checks = {
        "all_three_NEW_complete_V2_pairs42_original_checks": all(all(r["checks"].values()) for r in rs),
        "full_actual_materializations22_to2_all_comparisons22": all({e["mode"]: (e["GPU_boolean_materializations"], e["complete_CUDA_comparison_leaves"]) for e in r["entries"]} == {"native": (22, 22), "deferred": (2, 22)} for r in rs),
        "both_raw_whole_median_methods_and_each_limit": candidate / baseline <= 1.01 and statistics.median(ratios) <= 1.01 and max(ratios) <= 1.1,
        "charged_whole_median_protection": (candidate + fee_side) / (baseline + fee_side) <= 1.01,
        "all_known_fees_net_work_at_least10percent": net_work >= .1,
        "repeated_native41_public360_profiler_and_early_interrupt": all(generic["checks"].values()),
        "fresh_actual_candidate_prepare_mismatch_and_SIGINT": all(fault["checks"].values()) and all(mismatch["checks"].values()),
        "original_actual_native_error_and_SIGINT_development_evidence": all(read(BASE / "life1/mismatch_native_CHECK.json")["checks"].values()) and all(read(BASE / "life1/cancel_native_CHECK.json")["checks"].values()),
        "fresh_complete1016_recovery_all_fields_and_reference": all(fresh["checks"].values()) and all(fresh_compare.values()),
        "fresh1038_and_formal_frozen_input_identity": read(BASE / "r1/CONTRACT.json")["inputs"] == read(BASE / "d3/native/CONTRACT.json")["inputs"],
        "current_complete_native_lineage_copies_and_births": all(native["checks"].values()),
        "actual_native_review_exit_and_writer_birth_closed": native_writer_status["saved_instance_ended"]
            and read(BASE / "review_native0_GUARD.json")["returncode"] == 0,
        "formal_controller_sources_and_snapshots_frozen": sources_kept and all(data_identity(BASE / name / "sources" / Path(row["path"]).name)["sha256"] == row["sha256"] for name, p in zip(("d1", "d2", "d3"), protocols) for row in p["sources"]),
        "all_saved_Windows_birth_instances_ended": all(row["saved_instance_ended"] for row in ended),
        "all_three_formal_source_and_order_contracts_same": [p["order"] for p in protocols] == ["AB", "BA", "AB"] and protocols[0]["sources"] == protocols[1]["sources"] == protocols[2]["sources"],
        "parents36_unique_and_no_accepted_deferred_readback_family": master["additional_accepted_count"] == len(mechanism_names) == len(set(mechanism_names)) == len(master["acceptance_receipts"]) == 36 and not any("deferred" in value.lower() for value in mechanism_names),
        "private_family2GiB_disk20GiB_GPU12GiB_free2GiB": sum(p.stat().st_size for p in BASE.rglob("*") if p.is_file()) < 2 * 1024**3 and shutil.disk_usage(ROOT).free >= 20 * 1024**3 and gpu()["free_MiB"] >= 2048,
    }
    result = {"checks": checks, "B_whole_seconds": before, "C_whole_seconds": after, "pair_ratios": ratios,
        "B_whole_median_seconds": baseline, "C_whole_median_seconds": candidate,
        "known_fee_seconds": fee, "fees": fees, "symmetric_fee_per_formal_side_seconds": fee_side,
        "raw_work_reduction": 20 / 22, "net_work_reduction_index": net_work,
        "net_complete_speed": net_speed, "full_speed_not_accepted": net_speed < .02,
        "fresh_reference_checks": fresh_compare, "windows_saved_instances": ended,
        "actual_native_writer_close": native_writer_status,
        "family_bytes_before_publication": sum(p.stat().st_size for p in BASE.rglob("*") if p.is_file()),
        "single_chat_separate_self_review_not_independent_agent_review": True,
        "unknown_text_research_cost_not_zero_not_total_research_ROI": True,
        "writer_identity": {"pid": psutil.Process().pid, "creation_time": psutil.Process().create_time()},
        "proposed_new_count": 1 if all(checks.values()) else 0, "parent_count": 36,
        "actual_review_exit_text_review_post_closure_and_master_publication_still_required": True}
    save(output / "RESULT.json", result)
    print(json.dumps({"checks": checks, "F": fee, "net_work": net_work, "net_speed": net_speed,
        "windows_births": len(ended), "proposed_new_count": result["proposed_new_count"]}), flush=True)
    if not all(checks.values()):
        raise AssertionError("The separate complete candidate review failed")


if __name__ == "__main__":
    main()
