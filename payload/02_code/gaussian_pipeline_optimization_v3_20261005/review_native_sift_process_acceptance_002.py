"""Recompute complete costs and review saved source and process evidence."""

import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def walk(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk(child)


def main():
    started = time.perf_counter()
    out = ROOT/"04_outputs/gaussian_pipeline_optimization_v3_20261005/sp001"
    output = out/"ACCEPTANCE.json"
    if output.exists():
        raise FileExistsError("A root acceptance must be published once")
    campaign = read(out/"c4/RESULT.json")
    generic, diagnostic, journal, write, external = [read(out/name/"RESULT.json") for name in ("g0", "d0", "j1", "w0", "x0")]
    checks = {
        "original_three_pair_science_and_cost_gates": len(campaign["pairs"]) == 3 and all(campaign["checks"].values()),
        "six_actual_native_features_exact": all(diagnostic["checks"].values()),
        "all16_cache_and_lifecycle_guards": len(generic["checks"]) == 16 and all(generic["checks"].values()),
        "all14_journal_failure_timeout_and_drain_guards": len(journal["checks"]) == 14 and all(journal["checks"].values()),
        "all8_write_atomic_scope_permanent_failure_guards": len(write["checks"]) == 8 and all(write["checks"].values()),
        "all_three_public_six_image_native_component_pairs": len(external["pairs"]) == 3 and all(external["checks"].values()),
        "public_scope_does_not_claim360_or_GT_or_full_training": not external["protocol"]["full360_or_full_pipeline_claim"]
            and not external["protocol"]["GT_pose_ROI_box_prior_used"] and not external["protocol"]["blind_test"]}
    first_names = ("DIAGNOSTIC_LEASE_001.json", "GENERIC_LEASE_001.json", "CAMPAIGN_LEASE_001.json", "CAMPAIGN_LEASE_002.json",
        "CAMPAIGN_LEASE_003.json", "ACTUAL_DIAG_LEASE_001.json", "JOURNAL_LEASE_001.json", "JOURNAL_LEASE_002.json",
        "CAMPAIGN_LEASE_004.json", "WRITE_GUARD_LEASE_001.json")
    first = [read(out/name) for name in first_names]
    checks["all_first_fees_match_original_ledger_once"] = abs(sum(r["child_elapsed_seconds"] for r in first)-campaign["initial_fee_seconds"]) < 1e-8
    source_ids = campaign["sources"]+[s for result in (generic, diagnostic, journal, write, external) for s in result["sources"]]
    checks["all_frozen_algorithm_and_checker_sources_unchanged"] = all(s == data_identity(Path(s["path"])) for s in source_ids)
    nodes = []
    for trial in range(3):
        folder = out/"c4"/f"t{trial}"
        invariant = read(folder/"INVARIANTS.json")
        checks[f"pair{trial}_all_persisted_scientific_fields_exact"] = invariant["all_protected"] and all(invariant["checks"].values())
        for mode, label in (("incumbent", "b"), ("process", "c")):
            node = read(folder/("nb" if label == "b" else "nc")/"NODE.json")
            result = read(folder/label/"RESULT.json")
            nodes.append(node)
            checks[f"pair{trial}_{mode}_complete_owned_node_closed"] = node["status"] == "completed" and node["returncode"] == 0
            checks[f"pair{trial}_{mode}_all_original_entry_guards"] = all(result["checks"].values()) and len(read(Path(result["localization"]))["rows"]) == 24
            peaks = node["resource_peaks"]
            checks[f"pair{trial}_{mode}_frozen_budgets"] = peaks["Windows_tree_RSS_bytes"] <= 8*1024**3 and peaks["GPU_total_used_bytes"] <= 12*1024**3
            checks[f"pair{trial}_{mode}_post_wait_not_live"] = not node["live_Windows_after_wait"] and not any(r["alive"] for r in node["actual_Linux_after_wait"])
    for path in (out/"c2/t0/nb/NODE.json", out/"nd1/NODE.json", out/"c3/t0/nb/NODE.json"):
        nodes.append(read(path))
    linux = {(r["pid"], r["creation_time"]): r for node in nodes for r in node.get("actual_Linux_after_wait", [])}
    code = "import json,psutil; rows=json.loads("+repr(json.dumps(list(linux.values())))+"); output=[]\n"
    code += "for r in rows:\n alive=False\n try:\n  p=psutil.Process(r['pid']); alive=p.create_time()==r['creation_time'] and p.status()!=psutil.STATUS_ZOMBIE\n except psutil.NoSuchProcess: pass\n output.append(dict(pid=r['pid'],creation_time=r['creation_time'],alive=alive))\nprint(json.dumps(output))"
    actual_linux = json.loads(subprocess.check_output(["wsl", "-d", "E26-C2-LIV-20260907-134802", "--",
        "/opt/e26_thin_surface_v1/venv/bin/python", "-B", "-c", code], text=True, timeout=30))
    checks["all9_saved_actual_Linux_instances_gone_now"] = len(actual_linux) == 9 and not any(r["alive"] for r in actual_linux)
    leases = [*first, read(out/"CAMPAIGN_LEASE_005.json"), read(out/"EXTERNAL_LEASE_001.json")]
    windows = {}
    receipts = [*leases, *nodes, generic, diagnostic, journal, write, external,
        read(out/"c2/t0/nc/NODE.json")]
    for name in ("d0_guard.json", "g0_guard.json", "j0_guard.json", "j1_guard.json", "w0_guard.json", "x0_guard.json"):
        receipts.append(read(out/name))
    for item in receipts:
        for row in walk(item):
            pairs = [(row.get("pid"), row.get("creation_time")), (row.get("guard_pid"), row.get("guard_creation_time")),
                (row.get("child_pid"), row.get("child_creation_time"))]
            for pid, creation in pairs:
                if pid is not None and creation is not None and (pid, creation) not in linux:
                    windows[pid, creation] = {"pid": pid, "creation_time": creation}
    alive_windows = []
    for (pid, creation), row in windows.items():
        try:
            process = psutil.Process(pid)
            if process.create_time() == creation and process.status() != psutil.STATUS_ZOMBIE:
                alive_windows.append(row)
        except psutil.NoSuchProcess:
            pass
    checks["all_saved_actual_Windows_instances_gone_now"] = not alive_windows
    checks["all12_actual_leases_released"] = len(leases) == 12 and all(r.get("lease_released_utc") for r in leases)
    checks["both_modes_share_same_journal_and_write_recovery"] = all(
        Path(n["guard"]["source"]["path"]).name == "run_bounded_journal_task.py" for n in nodes[:6])
    checks["first_preparation_and_runtime_failures_retained"] = (first[2]["returncode"] != 0 and first[3]["returncode"] != 0
        and first[4]["returncode"] != 0 and first[5]["returncode"] != 0 and first[8]["returncode"] != 0)
    review_seconds = time.perf_counter()-started
    total_fee = campaign["initial_fee_seconds"]+sum(p["complete_compare_seconds"] for p in campaign["pairs"])
    total_fee += read(out/"EXTERNAL_LEASE_001.json")["child_elapsed_seconds"]+review_seconds+.5
    allocation = total_fee/6
    medians = {key: {mode: statistics.median(p[key][mode] for p in campaign["pairs"])+allocation for mode in ("incumbent", "process")}
        for key in ("raw_CPU_phase_seconds", "raw_whole_seconds")}
    reductions = {key: 1-v["process"]/v["incumbent"] for key, v in medians.items()}
    checks["final_all_fees_complete_phase_reduction_at_least10percent"] = reductions["raw_CPU_phase_seconds"] >= .1
    checks["final_all_fees_separate_whole_medians_not_regressed"] = reductions["raw_whole_seconds"] >= 0
    if not all(checks.values()):
        raise AssertionError(json.dumps({k:v for k,v in checks.items() if not v}))
    atomic_json(output, {"status": "accepted_scoped_engineering_mechanism", "mechanism": "R124_BOUNDED_NATIVE_CPU_QUERY_SIFT_PROCESSES",
        "review_kind": "root_self_review_no_subagent", "independent_review": False, "checks": checks, "new_optimization_count": 1,
        "complete_first_fees_with_external_and_final_review": total_fee, "each_side_fee": allocation, "final_review_seconds": review_seconds,
        "first_failed_review_preflight_charged_seconds": .5,
        "first_failed_review_preflight_clock_provenance": "functions.exec measured rounded 0.5-second patch and AST cell; preserved original syntax failure",
        "complete_phase_medians_seconds": medians["raw_CPU_phase_seconds"], "complete_phase_reduction": reductions["raw_CPU_phase_seconds"],
        "complete_whole_medians_seconds": medians["raw_whole_seconds"], "complete_whole_reduction": reductions["raw_whole_seconds"],
        "raw_whole_medians_seconds": {m:statistics.median(p["raw_whole_seconds"][m] for p in campaign["pairs"]) for m in ("incumbent","process")},
        "saved_Windows_instances_checked": len(windows), "current_live_Windows": alive_windows, "current_Linux": actual_linux,
        "all_first_receipts_once": [dict(path=name, child_seconds=r["child_elapsed_seconds"], exit=r["returncode"]) for name,r in zip(first_names,first)],
        "source_ids": source_ids, "phase_scope": "cold ordered 24-image query loop plus native map verification and all required contracts and closure",
        "external_scope": external["claim_boundary"], "external_diagnostic_reduction": external["diagnostic_batch_reduction"],
        "production_modified": False, "new_training_or_preview": "not_run_identical_existing_inputs",
        "whole_RAW_pipeline_or_unseen_object_quality_generalization": "unverified"})
    print(json.dumps({"accepted": True, "checks": len(checks), "new_count": 1,
        "phase_reduction": reductions["raw_CPU_phase_seconds"], "whole_reduction": reductions["raw_whole_seconds"],
        "Windows_instances": len(windows), "Linux_instances": len(actual_linux)}))


if __name__ == "__main__":
    main()
