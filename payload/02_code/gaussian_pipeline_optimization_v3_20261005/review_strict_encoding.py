"""Audit the full strict encoder, actual outputs, recovery, sources and costs."""

import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

import numpy as np
import psutil

from run_strict_encoding_campaign import ROOT, CODE, OUT, read
from run_sift_process_campaign_005 import compare_pair
from runtime import atomic_json, data_identity


def ended(identity):
    try:
        return psutil.Process(identity["pid"]).create_time() != identity["creation_time"]
    except psutil.NoSuchProcess:
        return True


def census(trace):
    expected = sum(math.prod(row["shape"])*np.dtype(row["dtype"]).itemsize for call in trace["calls"]
                   for row in call["actual_successful_numpy_outputs"])
    return expected == trace["actual_recorded_successful_output_bytes"] and all(
        all(row["owns_data"] for row in call["actual_successful_numpy_outputs"]) for call in trace["calls"])


def main():
    started = time.perf_counter()
    folder = OUT / "review0"
    folder.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(Path(__file__), folder / "REVIEW_SOURCE.py")
    campaign, validation = read(OUT / "c0/RESULT.json"), read(OUT / "r0/RESULT.json")
    cpu, public = read(OUT.parent / "cpu0/RESULT.json"), read(OUT / "g0/RESULT.json")
    science = [compare_pair(OUT / f"c0/t{i}/b", OUT / f"c0/t{i}/c") for i in range(3)]
    recovery = [compare_pair(OUT / "c0/t0/b", OUT / "r0" / ("recover_"+mode)) for mode in ("raw", "native")]
    traces = [row[mode]["trace"] for row in cpu["trials"] for mode in ("raw", "native")]
    traces.extend(trace for row in cpu["generic"] for trace in row["traces"])
    traces.extend(public[mode] for mode in ("raw", "native"))
    manifests = [campaign, validation, cpu, public]
    for i in range(3):
        for tag in ("b", "c"):
            value = read(OUT / f"c0/t{i}/{tag}/RESULT.json")
            manifests.append(value)
            traces.append(value["encoding"])
    for kind in ("error", "cancel", "recover"):
        for mode in ("raw", "native"):
            leaf = OUT / "r0" / (kind+"_"+mode)
            manifests.append(read(leaf / "CONTRACT.json"))
            value = read(leaf / ("RESULT.json" if kind == "recover" else "FAULT_RESULT.json"))
            manifests.append(value)
            traces.append(value["encoding"])
    identities = {}
    for manifest in manifests:
        for identity in manifest.get("sources", []) + manifest.get("inputs", []):
            path = identity["path"]
            if path in identities and identities[path] != identity:
                raise ValueError("A recorded strict encoder source or input revision disagreed")
            identities[path] = identity
    current = [data_identity(Path(path)) == identity for path, identity in identities.items()]
    nodes, windows, linux = [], {}, {}
    for path in OUT.rglob("NODE.json"):
        node = read(path)
        nodes.append(node)
        for identity in node["guard"]["owned_process_identities"] + [node["owner"]]:
            windows[(identity["pid"], identity["creation_time"])] = identity
        for identity in node["Linux_resource_receipts"]:
            linux[(identity["pid"], identity["creation_time"])] = {"pid": identity["pid"], "creation_time": identity["creation_time"]}
    leases = []
    for path in OUT.parent.glob("*_LEASE.json"):
        if path.name == "REVIEW_LEASE.json":
            continue
        value = read(path)
        leases.append({"path": str(path), "returncode": value["returncode"], "seconds": value["child_elapsed_seconds"],
                       "released": "lease_released_utc" in value})
        for prefix in ("guard", "child"):
            if prefix+"_pid" in value and prefix+"_creation_time" in value:
                identity = {"pid": value[prefix+"_pid"], "creation_time": value[prefix+"_creation_time"]}
                windows[(identity["pid"], identity["creation_time"])] = identity
    code = "import json,psutil; rows=json.loads("+repr(json.dumps(list(linux.values())))+"); result=[]\n"
    code += "for row in rows:\n alive=False\n try:\n  p=psutil.Process(row['pid']); alive=p.create_time()==row['creation_time'] and p.status()!=psutil.STATUS_ZOMBIE\n except psutil.NoSuchProcess: pass\n result.append(dict(row,alive=alive))\nprint(json.dumps(result))"
    actual_linux = json.loads(subprocess.check_output(["wsl", "-d", "E26-C2-LIV-20260907-134802", "--",
        "/opt/e26_thin_surface_v1/venv/bin/python", "-B", "-c", code], text=True, timeout=30))
    fee_bytes = campaign["known_current_extra_array_output_bytes_plus_registered2GiB"] + validation["extra_actual_array_output_bytes"]
    totals = {mode: statistics.median(pair["actual_array_output_bytes"][mode] for pair in campaign["pairs"]) for mode in ("raw", "native")}
    net = (totals["raw"]-totals["native"])/(totals["raw"]+fee_bytes/6)
    whole = campaign["raw_whole_medians_seconds"]
    formal_sum = sum(sum(pair["whole_seconds"].values()) for pair in campaign["pairs"])
    related = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/oup018"
    related_seconds = sum(read(path)["child_elapsed_seconds"] for path in [*related.glob("*_LEASE.json"), *(related / "R170").glob("*_LEASE.json")])
    fee_seconds = sum(row["seconds"] for row in leases)-formal_sum+related_seconds+120
    charged = {mode: value+fee_seconds/6 for mode, value in whole.items()}
    checks = {
        "all20_original_layout_errors_warnings_and_full48_three_CPU_pairs": all(cpu["checks"].values()),
        "all360_complete_public_CPU_inputs_supported_and_fallback": all(public["checks"].values()),
        "three_fresh_full24_48_all_original_science_extra_numeric_exact": len(science) == 3 and all(row["all_protected"] for row in science),
        "four_original_caller_OSError_real_SIGINT_prefix_RNG_and_cleanup": all(validation["checks"].values()),
        "both_fresh_full24_48_recoveries_original_control_exact": all(row["all_protected"] for row in recovery),
        "all_actual_six_API_output_censuses_recomputed_and_owned": all(census(trace) for trace in traces),
        "all_current_source_input_and_compiled_DLL_identities": all(current),
        "all12_nodes_actual_Windows_and_Linux_closed": len(nodes) == 12 and all(not node["live_Windows_after_wait"]
            and not any(row["alive"] for row in node["actual_Linux_after_wait"]) for node in nodes),
        "all_saved_Windows_birth_instances_ended": all(ended(row) for row in windows.values()),
        "all_saved_Linux_birth_instances_ended": all(not row["alive"] for row in actual_linux),
        "all_prior_exclusive_leases_released": all(row["released"] for row in leases),
        "each_full_whole_within_original1_1": all(pair["whole_seconds"]["native"] <= 1.1*pair["whole_seconds"]["raw"] for pair in campaign["pairs"]),
        "raw_and_charged_whole_median_within_V7_1_03": whole["native"] <= 1.03*whole["raw"] and charged["native"] <= 1.03*charged["raw"],
        "net_registered_array_output_work_at_least10percent": net >= .1,
        "no_hidden_phase_speed_or_peak_memory_claim": campaign["no_speed_production_or_hardware_memory_claim"],
        "all_owned_family_disk_within12GiB": sum(path.stat().st_size for path in OUT.parent.rglob("*") if path.is_file()) <= 12*1024**3,
    }
    result = {"status": "scoped_work_all_checks_passed_pending_publication_and_actual_writer_exit" if all(checks.values()) else "failed_verification",
        "checks": checks, "original_raw_array_output_bytes": totals, "complete_extra_array_fee_bytes": fee_bytes,
        "net_array_output_reduction": net, "raw_whole_medians_seconds": whole,
        "known_related_all_measured_fee_plus_registered120_seconds": fee_seconds, "charged_whole_medians_seconds": charged,
        "root_separate_single_chat_self_review_not_independent": True, "current_identities": list(identities.values()),
        "Windows": list(windows.values()), "Linux": actual_linux, "leases": leases,
        "review_seconds": time.perf_counter()-started, "writer": {"pid": psutil.Process().pid, "creation_time": psutil.Process().create_time()},
        "new_optimization_count": 0}
    atomic_json(folder / "RESULT.json", result)
    print(json.dumps({key: result[key] for key in ("status", "checks", "net_array_output_reduction", "raw_whole_medians_seconds", "review_seconds")}), flush=True)
    if not all(checks.values()):
        raise AssertionError("A complete strict encoder work, source, science or actual closure audit failed")


if __name__ == "__main__":
    main()
