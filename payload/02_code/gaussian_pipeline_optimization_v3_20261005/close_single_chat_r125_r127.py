"""Audit actual saved evidence, current source identities and owned process closure."""

import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

import psutil

STARTED = time.perf_counter()
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity


def read(path): return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    out = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/dpv001"
    fa = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/fa001"
    campaign = read(out / "c1/RESULT.json")
    old = read(out / "c0/RESULT.json")
    event0, event1 = read(out / "e0/RESULT.json"), read(out / "e1/RESULT.json")
    generic = [read(out / (name + "/RESULT.json")) for name in ("g0", "g1")]
    fee_names = ["GENERIC_LEASE_001.json", "CAMPAIGN_LEASE_001.json", "EVENT_LEASE_001.json", "GENERIC_LEASE_002.json", "EVENT_LEASE_002.json", "CAMPAIGN_LEASE_002.json"]
    leases = [read(out / name) for name in fee_names] + [read(fa / "DIAGNOSTIC_LEASE_001.json")]
    nodes = [read(path) for path in sorted(out.glob("n*/NODE.json"))]
    nodes += [read(path) for folder in ("c0", "c1") for path in sorted((out / folder).glob("*_node/NODE.json"))]
    nodes.append(read(fa / "n0/NODE.json"))
    windows, linux, identities = {}, {}, {}
    for row in leases:
        for prefix in ("guard", "child"):
            windows[(row[prefix + "_pid"], row[prefix + "_creation_time"])] = {"pid": row[prefix + "_pid"], "creation_time": row[prefix + "_creation_time"]}
    for row in nodes:
        for item in [row["owner"], *row["guard"]["owned_process_identities"]]: windows[(item["pid"], item["creation_time"])] = item
        for item in row["actual_Linux_after_wait"]: linux[(item["pid"], item["creation_time"])] = {"pid": item["pid"], "creation_time": item["creation_time"]}
    for folder in [*out.glob("c*/t*_native"), *out.glob("c*/t*_stream"), out / "g0", out / "g1", fa / "d0"]:
        contract = read(folder / ("PROTOCOL.json" if folder == fa / "d0" else "CONTRACT.json"))
        for item in contract["sources_inputs"]:
            identities[(item["path"], item["sha256"])] = item
    live = []
    for item in windows.values():
        try:
            process = psutil.Process(item["pid"])
            if process.create_time() == item["creation_time"]: live.append(item)
        except psutil.NoSuchProcess: pass
    payload = {"processes": list(linux.values()), "files": list(identities.values())}
    code = "import json,hashlib,pathlib,psutil; p=json.loads(" + repr(json.dumps(payload)) + "); out={'live':[],'bad_files':[]}\n"
    code += "for i in p['processes']:\n try:\n  q=psutil.Process(i['pid'])\n  if q.create_time()==i['creation_time'] and q.status()!=psutil.STATUS_ZOMBIE: out['live'].append(i)\n except psutil.NoSuchProcess: pass\n"
    code += "for i in p['files']:\n f=pathlib.Path(i['path']); h=hashlib.sha256()\n with f.open('rb') as s:\n  for b in iter(lambda:s.read(1024**2),b''): h.update(b)\n if f.stat().st_size!=i['bytes'] or h.hexdigest()!=i['sha256']: out['bad_files'].append(i['path'])\nprint(json.dumps(out))"
    actual = json.loads(subprocess.check_output(["wsl", "-d", "E26-C2-LIV-20260907-134802", "--",
        "/opt/e26_thin_surface_v1/venv/bin/python", "-B", "-c", code], text=True, timeout=30))
    before_fees = campaign["complete_shared_first_fee_seconds"]
    raw = {key: {mode: statistics.median(row["sides"][mode]["raw_" + key] for row in campaign["pairs"])
        for mode in ("native", "stream")} for key in ("whole_seconds", "complete_phase_seconds")}
    checks = {"old_full_three_pairs_scientific_bytes_exact": len(old["pairs"]) == 3 and all(all(row["checks"].values()) for row in old["pairs"]),
        "revised_all_three_complete_pairs_exact": len(campaign["pairs"]) == 3 and all(all(row["checks"].values()) for row in campaign["pairs"]),
        "both_generic_versions_48_rows_passed": all(len(item["rows"]) == 48 and all(item["checks"].values()) for item in generic),
        "first_event_failure_preserved": event0["both_real_mismatches_reproduced"],
        "both_revised_original_event_guards_passed": event1["both_original_event_guards_matched"],
        "all_prior_seven_leases_closed": all(row.get("lease_released_utc") and row["returncode"] == 0 for row in leases),
        "all_measured_nodes_source_guard_and_close": all(row["status"] == "completed" and row["final_source_unchanged"]
            and not row["live_Windows_after_wait"] and not any(item["alive"] for item in row["actual_Linux_after_wait"]) for row in nodes),
        "saved_Windows_instances_actually_gone": not live, "saved_Linux_instances_actually_gone": not actual["live"],
        "complete_saved_sources_and_original_inputs_current": not actual["bad_files"],
        "only_original_complete_speed_gate_failed": campaign["status"] == "closed_negative_complete_cost"
            and campaign["checks"]["complete_phase_at_least10percent"] is False
            and all(value for key, value in campaign["checks"].items() if key != "complete_phase_at_least10percent"),
        "P5_R125_diagnostic_only_0_count": read(fa / "d0/RESULT.json")["new_optimization_count"] == 0}
    fee = before_fees + time.perf_counter() - STARTED
    medians = {key: {mode: value + fee / 6 for mode, value in row.items()} for key, row in raw.items()}
    memory = {mode: statistics.median(row["sides"][mode]["WSL_peak_RSS_bytes"] for row in campaign["pairs"]) for mode in ("native", "stream")}
    result = {"status": "closed_negative_complete_speed_gate_measured_memory_benefit_retained",
        "checks": checks, "review_kind": "root_self_review_no_subagent", "independent_review": False,
        "new_optimization_count": 0, "raw_complete_medians_seconds": raw,
        "median_seconds_including_review_before_its_outer_guard_close": medians,
        "reductions_including_review_before_outer_guard_close": {key: 1 - row["stream"] / row["native"] for key, row in medians.items()},
        "median_actual_process_peak_RSS_bytes": memory, "observed_RSS_reduction": 1 - memory["stream"] / memory["native"],
        "memory_was_not_the_preregistered_acceptance_gate": True,
        "historical_owned_Windows_instances_checked": len(windows), "historical_owned_Linux_instances_checked": len(linux),
        "current_source_input_identities_checked": len(identities), "actual_live_Windows": live, "actual_Linux_and_file_verification": actual,
        "remaining_formal_recovery_permission_or_interruption_and_external": "not_run_after_complete_speed_gate_failure",
        "production_modified": False, "new_full_training_or_other_object_GT_or_preview": "not_run",
        "review_fee_inside_script_seconds": time.perf_counter() - STARTED,
        "outer_review_guard_and_final_cost_accounting_remain_separate": True,
        "source": data_identity(Path(__file__))}
    atomic_json(out / "NEGATIVE_CLOSURE_001.json", result)
    print(json.dumps({key: result[key] for key in ("status", "checks", "raw_complete_medians_seconds", "observed_RSS_reduction", "historical_owned_Windows_instances_checked", "historical_owned_Linux_instances_checked", "review_fee_inside_script_seconds")}), flush=True)
    if not all(checks.values()): raise AssertionError("A root negative-evidence or actual process closure check failed")


if __name__ == "__main__": main()
