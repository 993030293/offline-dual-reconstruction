"""Queue fresh independently bounded paired metric nodes and stop at first failure."""

import argparse
import datetime
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parent
OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/direct_schur_001"
PYTHON = ROOT / "05_envs/sfm_py311_v1/Scripts/python.exe"
V1 = ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"
sys.path.insert(0, str(V1))
from runtime import atomic_json, data_identity


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def closed_node_seconds(lease):
    released = datetime.datetime.fromisoformat(lease["lease_released_utc"]).timestamp()
    elapsed = released - lease["guard_creation_time"] - lease["wait_seconds"]
    if elapsed < lease["child_elapsed_seconds"]:
        raise AssertionError("Closed-node clock does not include the complete child")
    return elapsed


def drained(resource, lease):
    owned = resource.get("owned_process_identities", []) + [
        {"pid": lease["guard_pid"], "creation_time": lease["guard_creation_time"]},
        {"pid": lease["child_pid"], "creation_time": lease["child_creation_time"]}]
    for identity in owned:
        try:
            if psutil.Process(identity["pid"]).create_time() == identity["creation_time"]: return False
        except psutil.NoSuchProcess:
            pass
    return bool(lease.get("lease_released_utc"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maximum-campaign-seconds", type=int, default=21600)
    parser.add_argument("--existing-first-auto", action="store_true")
    args = parser.parse_args()
    if not 0 < args.maximum_campaign_seconds <= 21600:
        raise ValueError("The controller requires a finite registered campaign deadline")
    result_path = OUT / "CAMPAIGN_RESULT.json"
    if result_path.exists(): raise FileExistsError(result_path)
    started = time.perf_counter()
    source = data_identity(Path(__file__))
    setup = []
    for name in ("backend_c001", "oracle_c001", "oracle_c002"):
        lease, resource = read(OUT / (name + "_LEASE.json")), read(OUT / (name + "_RESOURCE.json"))
        if not drained(resource, lease): raise AssertionError("Preparatory node did not drain")
        setup.append({"node": name, "closed_seconds": closed_node_seconds(lease),
                      "wait_seconds": lease["wait_seconds"], "status": resource["status"],
                      "assigned_once_to": "pair0_direct"})
    if read(OUT / "backend_c001/RESULT.json")["status"] != "actual_backend_generic_passed":
        raise ValueError("Actual backend is not proved")
    if read(OUT / "oracle_c002/RESULT.json")["status"] != "full_readback_and_projection_oracle_passed":
        raise ValueError("Repaired oracle is not proved")
    setup_seconds = sum(row["closed_seconds"] for row in setup)
    sides, pairs, failure = [], [], None
    atomic_json(OUT / "CAMPAIGN_CONTRACT.json", {"source": source, "maximum_controller_seconds": args.maximum_campaign_seconds,
                "per_node_seconds": 1800, "per_node_process_tree_RSS_bytes": 8*1024**3,
                "orders": [["auto", "direct"], ["direct", "auto"], ["auto", "direct"]], "setup": setup,
                "cost_definition": "Completephase and whole include the entire closed metric node from native guard process creation to lease release minus measured lock wait; prepare, imports, all checks/write/readback/publication/restore and resource close are billed. All setup nodes are charged once to pair0_direct.",
                "child_profile_phase_seconds_are_components_not_whole_speedup": True,
                "historical_elapsed_is_control": False, "counted": False})
    try:
        for pair, order in enumerate((("auto", "direct"), ("direct", "auto"), ("auto", "direct"))):
            current = {}
            for mode in order:
                if time.perf_counter()-started >= args.maximum_campaign_seconds:
                    raise TimeoutError("Registered finite parent controller deadline reached between independent nodes")
                name = "pair" + str(pair) + "_" + mode
                lease_path, resource_path = OUT / (name + "_LEASE.json"), OUT / (name + "_RESOURCE.json")
                output = OUT / name
                existing = pair == 0 and mode == "auto" and args.existing_first_auto
                if not existing:
                    if lease_path.exists() or resource_path.exists() or output.exists(): raise FileExistsError(name)
                    command = [str(PYTHON), "-X", "utf8", "-B", str(CODE / "run_with_resource_lease.py"),
                               "--owner", "R114_"+name, "--receipt", str(lease_path), "--wait-seconds", "1800", "--",
                               str(PYTHON), "-X", "utf8", "-B", str(V1 / "run_bounded_task.py"),
                               "--receipt", str(resource_path), "--log", str(OUT / (name + ".log")),
                               "--maximum-seconds", "1800", "--maximum-rss-gib", "8", "--",
                               str(PYTHON), "-X", "utf8", "-B", str(CODE / "profile_direct_schur.py"),
                               "--mode", mode, "--output", str(output)]
                    atomic_json(OUT / "CAMPAIGN_STATE.json", {"status": "running", "pair": pair, "mode": mode,
                                "command": command, "completed_nodes": len(sides), "counted": False})
                    child = subprocess.Popen(command, cwd=ROOT)
                    code = child.wait()
                else:
                    code = read(lease_path).get("returncode")
                lease, resource = read(lease_path), read(resource_path)
                item = read(output / "RESULT.json") if (output / "RESULT.json").exists() else None
                seconds = closed_node_seconds(lease)
                assigned = setup_seconds if pair == 0 and mode == "direct" else 0.
                record = {"node": name, "mode": mode, "pair": pair, "returncode": code,
                          "lease": lease, "resource": resource, "result": item,
                          "owned_processes_drained": drained(resource, lease),
                          "closed_node_seconds": seconds, "setup_seconds_charged_once": assigned,
                          "completephase_seconds": seconds+assigned, "whole_seconds": seconds+assigned}
                sides.append(record); current[mode] = record
                atomic_json(OUT / "CAMPAIGN_SIDES.json", sides)
                if (code != 0 or resource["status"] != "completed" or not record["owned_processes_drained"]
                        or resource["maximum_seconds"] != 1800 or resource["maximum_RSS_bytes"] != 8*1024**3
                        or resource["peak_tree_RSS_bytes"] > 8*1024**3 or item is None
                        or item["status"] != "metric_complete_passed"):
                    failure = {"node": name, "phase": "complete_metric_or_resource_or_close", "mode": mode,
                               "classification": "candidate_first_required_gate_failure" if mode == "direct" else "control_gate_failure",
                               "recorded_failure": item.get("failure") if item else resource.get("error"), "counted": False}
                    raise RuntimeError("First required independent metric node failed")
            left, right = current["auto"], current["direct"]
            if left["result"]["sources_inputs_unchanged"] is not True or right["result"]["sources_inputs_unchanged"] is not True:
                raise AssertionError("Paired source/input identity was not independently preserved")
            if read(OUT / left["node"] / "CONTRACT.json")["source"] != read(OUT / right["node"] / "CONTRACT.json")["source"]:
                raise AssertionError("Paired source changed")
            if read(OUT / left["node"] / "CONTRACT.json")["inputs"] != read(OUT / right["node"] / "CONTRACT.json")["inputs"]:
                raise AssertionError("Paired inputs changed")
            ratio = right["whole_seconds"]/left["whole_seconds"]
            pair_result = {"pair": pair, "completephase_direct_over_auto": right["completephase_seconds"]/left["completephase_seconds"],
                           "whole_direct_over_auto": ratio, "whole_at_most1_1": ratio <= 1.1}
            pairs.append(pair_result)
            atomic_json(OUT / "PAIR_COSTS.json", pairs)
            if ratio > 1.1:
                failure = {"node": "pair"+str(pair), "phase": "each_whole_cost_guard", "classification": "candidate_first_required_gate_failure",
                           "observed_ratio": ratio, "limit": 1.1, "counted": False}
                raise RuntimeError("First paired whole-cost guard failed")
        median_phase = statistics.median(row["completephase_direct_over_auto"] for row in pairs)
        median_whole = statistics.median(row["whole_direct_over_auto"] for row in pairs)
        if median_phase > .90 or median_whole > 1.:
            failure = {"phase": "three_pair_median_cost_guard", "classification": "candidate_first_required_gate_failure",
                       "completephase_ratio": median_phase, "limit": .90, "whole_ratio": median_whole, "whole_limit": 1., "counted": False}
            raise RuntimeError("Registered median cost gate failed")
    except BaseException as error:
        failure = failure or {"phase": "controller_or_checker", "error": repr(error), "classification": "controller_setup_or_execution_error", "counted": False}
    elapsed = time.perf_counter()-started
    dependents = ["pair0_own_map_roles_export_support_undistort", "all24_and48_localization_each_map", "three_seeds_12000_full_target_quality",
                  "original_value_gradient_pixels", "actual8000_to12000_checkpoint_recovery", "complete_preview_contract", "Oxford_compatibility", "root_independent_review"]
    unchanged = data_identity(Path(__file__)) == source
    if not unchanged: failure = failure or {"phase": "controller_source_recheck", "counted": False}
    atomic_json(result_path, {"status": "three_pairs_metric_passed_downstream_pending" if failure is None else "stopped_at_first_required_failure",
                "failure": failure, "pair_costs": pairs, "completed_node_count": len(sides), "setup": setup,
                "controller_elapsed_seconds_including_waits": elapsed, "setup_closed_seconds": setup_seconds,
                "total_closed_metric_seconds": sum(row["closed_node_seconds"] for row in sides),
                "total_measured_closed_seconds_including_setup": setup_seconds+sum(row["closed_node_seconds"] for row in sides),
                "total_recorded_lock_wait_seconds": sum(row["wait_seconds"] for row in setup)+sum(row["lease"]["wait_seconds"] for row in sides),
                "source_unchanged": unchanged, "counted": False,
                "dependent_status": {name: "not_run_after_failure" if failure else "pending" for name in dependents}})
    atomic_json(OUT / "CAMPAIGN_STATE.json", {"status": "failed_closed" if failure else "metric_closed_downstream_pending",
                "completed_nodes": len(sides), "failure": failure, "counted": False})
    if failure: raise SystemExit(1)


if __name__ == "__main__": main()
