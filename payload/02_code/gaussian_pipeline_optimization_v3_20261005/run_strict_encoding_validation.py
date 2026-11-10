"""Run both native caller failures, real SIGINT and fresh complete recoveries."""

import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from run_strict_encoding_campaign import ROOT, CODE, OUT, read
from run_sift_process_campaign_005 import compare_pair
from runtime import atomic_json, data_identity


def main():
    started = time.perf_counter()
    output = OUT / "r0"
    output.mkdir(parents=True, exist_ok=False)
    campaign = read(OUT / "c0/RESULT.json")
    if len(campaign["pairs"]) != 3 or not campaign["raw_whole_median_within1_03"] or not all(
            all(row["checks"].values()) for row in campaign["pairs"]):
        raise ValueError("All three original full pairs and unchanged guards must pass first")
    source_paths = [Path(__file__), CODE / "fault_strict_encoding.py", CODE / "fault_codec_localization.py",
                    CODE / "localize_strict_encoding.py", CODE / "strict_encoding_node.py"]
    sources = [data_identity(path) for path in source_paths]
    for number, path in enumerate(source_paths):
        shutil.copyfile(path, output / (f"SOURCE_{number:02d}_" + path.name))
    nodes = []
    for kind, encoding in [(kind, encoding) for kind in ("error", "cancel", "recover") for encoding in ("raw", "native")]:
        if time.perf_counter()-started > 850 or sources != [data_identity(path) for path in source_paths]:
            raise RuntimeError("The bounded frozen full validation changed")
        name = kind + "_" + encoding
        leaf, node = output / name, output / ("n_" + name)
        if kind == "recover":
            child = [sys.executable, "-B", str(CODE / "localize_strict_encoding.py"),
                     "--encoding-mode", encoding, "--output", str(leaf)]
        else:
            child = [sys.executable, "-B", str(CODE / "fault_strict_encoding.py"),
                     "--wire-mode", "raw" if encoding == "raw" else "codec", "--kind", kind, "--output", str(leaf)]
        command = [sys.executable, "-B", str(CODE / "strict_encoding_node.py"), "--kind", "localization",
                   "--mode", "process", "--node", str(node), "--output", str(leaf), "--", *child]
        with (output / (name + ".log")).open("x", encoding="utf-8") as log:
            code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=900).returncode
        actor = read(node / "NODE.json")
        if actor["live_Windows_after_wait"] or any(row["alive"] for row in actor["actual_Linux_after_wait"]):
            raise RuntimeError("An actual original native validation process remained alive")
        if kind == "recover":
            result = read(leaf / "RESULT.json")
            if code or actor["status"] != "completed" or not all(result["checks"].values()):
                raise RuntimeError("A fresh original complete24/48 recovery failed")
        else:
            result = read(leaf / "FAULT_RESULT.json")
            if actor["guard"]["returncode"] != 86 or code != 1 or not all(result["checks"].values()):
                raise RuntimeError("An expected actual full-prefix caller error or native SIGINT failed")
        nodes.append({"name": name, "kind": kind, "encoding": encoding, "node": str(node), "output": str(leaf),
                      "actual_node_exit": code, "actual_child_exit": actor["guard"]["returncode"]})
        atomic_json(output / "STATE.json", {"status": "running", "nodes": nodes, "sources": sources, "new_optimization_count": 0})
    prefix_checks = {}
    for kind in ("error", "cancel"):
        a, b = [read(output / (kind + "_" + mode) / "FAULT_RESULT.json") for mode in ("raw", "native")]
        prefix_checks[kind] = (a["prefix"] == b["prefix"] and a["before_rng"] == b["before_rng"]
                              and a["after_rng"] == b["after_rng"] and a["error"] == b["error"])
    recovery_checks = [compare_pair(OUT / "c0/t0/b", output / ("recover_" + mode)) for mode in ("raw", "native")]
    checks = {"all_four_actual_errors_SIGINT_full_prefix_RNG_exact": all(prefix_checks.values()),
              "both_new_directory_full24_48_recoveries_original_control_exact": all(row["all_protected"] for row in recovery_checks),
              "all_six_actual_validation_nodes_closed": len(nodes) == 6,
              "all_registered_validation_sources_unchanged": sources == [data_identity(path) for path in source_paths]}
    extra = sum(read(output / row["name"] / ("RESULT.json" if row["kind"] == "recover" else "FAULT_RESULT.json"))[
        "actual_array_output_bytes" if row["kind"] == "recover" else "extra_actual_array_output_bytes"] for row in nodes)
    result = {"status": "full_faults_real_SIGINT_fresh_recovery_passed_pending_root0" if all(checks.values()) else "failed_verification",
              "checks": checks, "sources": sources, "nodes": nodes, "prefix_checks": prefix_checks,
              "recovery_invariants": recovery_checks, "extra_actual_array_output_bytes": extra,
              "complete_seconds": time.perf_counter()-started, "new_optimization_count": 0}
    atomic_json(output / "RESULT.json", result)
    print(json.dumps({key: result[key] for key in ("status", "checks", "extra_actual_array_output_bytes", "complete_seconds")}), flush=True)
    if not all(checks.values()):
        raise AssertionError("A full native prefix, RNG or recovered science guard failed")


if __name__ == "__main__":
    main()
