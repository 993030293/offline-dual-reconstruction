"""Independently recompute the root-owned negative screen and process closure."""

import time
STARTED = time.perf_counter()

import json
import msvcrt
import os
from pathlib import Path
import subprocess
import sys

import psutil

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).parent
OWN = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/eg001"
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity
from run_graph_reference_screen import compare


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def windows(path):
    value = str(path)
    return Path("C:/" + value[7:]) if value.startswith("/mnt/c/") else Path(value)


def closed(identity):
    try:
        return psutil.Process(identity["pid"]).create_time() != identity["creation_time"]
    except psutil.NoSuchProcess:
        return True


def main():
    generic = read(OWN / "g0/RESULT.json")
    result = read(OWN / "s1/RESULT.json")
    contract = read(OWN / "s1/CONTRACT.json")
    comparison = compare(OWN / "s1/incumbent", OWN / "s1/graph")
    states, recalculated = [], []
    generic_resource = read(OWN / "g0/WSL_RESOURCE.json")
    ids = [{"pid": generic_resource["pid"], "creation_time": generic_resource["creation_time"]}]
    source_copies = 0
    for mode in ("incumbent", "graph"):
        folder = OWN / "s1" / mode
        measurement, graph = read(folder / "measurement/RESULT.json"), read(folder / "GRAPH_RESULT.json")
        stage = (sum(row["seconds_including_complete_input_protection"] for row in measurement["rows"])
                 + measurement["composition_source_copy_identity_seconds"]
                 + graph["close_seconds_all_actual_cache_calls"] + graph["actual_native_dependency_validation"]["seconds"])
        recalculated.append(stage)
        states.append({"mode": mode, "measurement": measurement, "graph": graph,
                       "linux_resource": read(folder / "measurement/WSL_RESOURCE.json")})
        ids.append(read(folder / "measurement/PROCESS_IDENTITY.json"))
    identities = list(dict((row["path"], row) for row in [*contract["sources"], *contract["inputs"],
        *(value for value in generic["sources"] if value["path"].startswith("/mnt/c/"))]).values())
    same = all(data_identity(windows(row["path"]))["sha256"] == row["sha256"] for row in identities)
    for index, row in enumerate(contract["sources"]):
        copy = OWN / "s1/source_copies" / (f"{index:02d}_" + Path(row["path"]).name)
        if data_identity(copy)["sha256"] != row["sha256"]:
            raise AssertionError("An actual executed source copy changed")
        source_copies += 1
    for value in states:
        folder = OWN / "s1" / value["mode"] / "measurement/source_copies"
        for index, row in enumerate(value["measurement"]["sources"]):
            if data_identity(folder / (f"{index:02d}_" + Path(row["path"]).name))["sha256"] != row["sha256"]:
                raise AssertionError("An actual Linux evaluator source copy changed")
            source_copies += 1
    leases = [read(OWN / name) for name in ("g0_lease.json", "s0_lease.json", "s1_lease.json")]
    windows_seen = {}
    for path in (OWN / "g0_GUARD.json", OWN / "s0_GUARD.json", OWN / "s1_GUARD.json",
                 OWN / "s1/incumbent/WINDOWS_GUARD.json", OWN / "s1/graph/WINDOWS_GUARD.json"):
        for identity in read(path)["owned_process_identities"]:
            windows_seen[(identity["pid"], identity["creation_time"])] = identity
    for lease in leases:
        for prefix in ("guard", "child"):
            identity = {"pid": lease[prefix + "_pid"], "creation_time": lease[prefix + "_creation_time"]}
            windows_seen[(identity["pid"], identity["creation_time"])] = identity
    windows_seen.update({(value["pid"], value["creation_time"]): value
                         for value in [read(OWN / "s1/PROCESS_IDENTITY.json")]})
    command = ("import json,psutil; rows=" + repr(ids) + "; result=[]\n"
        "for row in rows:\n"
        " try: done=psutil.Process(row['pid']).create_time()!=row['creation_time']\n"
        " except psutil.NoSuchProcess: done=True\n"
        " result.append(dict(row,instance_closed=done))\n"
        "print(json.dumps(result))")
    native = subprocess.run(["wsl", "-d", "E26-C2-LIV-20260907-134802", "--",
        "/opt/e26_thin_surface_v1/venv/bin/python", "-B", "-c", command],
        check=True, capture_output=True, text=True, timeout=30)
    linux_closed = json.loads(native.stdout)
    lock = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/coordination_001/exclusive_benchmark.lock"
    with open(lock, "r+b", buffering=0) as stream:
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
    parent = read(ROOT / "research/2026-10-05-gaussian-optimization-round3-50/CURRENT_STATE.json")
    graph_state = states[1]["measurement"]["cache_before_clear"]["owned_graph"]
    checks = {
        "all27_generic_original_scalars_and_inputs": len(generic["rows"]) == 27 and all(
            row["absolute_error"] <= 1e-7 and row["inputs_unchanged"] for row in generic["rows"]) and all(generic["checks"].values()),
        "complete_24_48_96_outputs_and_JPEG_recompared": all(value for key, value in comparison.items()
            if key != "maximum_absolute_SSIM_error") and comparison["maximum_absolute_SSIM_error"] == 0,
        "all192_actual_input_QA_and_four_full_model_QA": all(len(value["measurement"]["rows"]) == 96
            and all(row["input_bits_and_versions_unchanged"] for row in value["measurement"]["rows"])
            and len(value["measurement"]["model_states"]) == 2 and all(value["measurement"]["checks"].values()) for value in states),
        "all_complete_stage_values_recompute": all(abs(value - row["complete_SSIM_stage_seconds"]) < 1e-12
            for value, row in zip(recalculated, result["rows"])),
        "raw_speed_failure_not_accepted_or_reinterpreted": result["status"] == "closed_negative_first_raw_screen"
            and result["new_optimization_count"] == 0 and recalculated[1] >= recalculated[0]
            and not result["checks"]["raw_stage_reduction_at_least3percent"],
        "one_actual_graph_48_replays_and_24_view_forgets": graph_state["captures"] == 1
            and graph_state["replays"] == 48 and graph_state["native_calls"] == 192
            and graph_state["reference_forgets"] == 24 and graph_state["closed"] and graph_state["live_graph_entries"] == 0,
        "all_owned_caches_native_dependencies_and_bindings_closed": all(all(value["graph"]["checks"].values())
            and value["measurement"]["cache_after_clear"]["resident_accounted_bytes"] == 0 for value in states),
        "all_current_sources_and_complete_inputs_exact": same,
        "all_executed_Linux_Windows_source_copies_exact": source_copies > 30,
        "s0_actual_failure_and_original_source_preserved": read(OWN / "s0_GUARD.json")["returncode"] == 1
            and "FileNotFoundError" in (OWN / "s0.log").read_text(encoding="utf-8")
            and 'data_identity(windows(row["path"]))' in (OWN / "s0/FAILED_DRIVER_SOURCE.py").read_text(encoding="utf-8"),
        "screen_actual_controller17_two_actual_child0": read(OWN / "s1_GUARD.json")["returncode"] == 17
            and all(read(OWN / "s1" / mode / "WINDOWS_GUARD.json")["returncode"] == 0 for mode in ("incumbent", "graph")),
        "all_saved_Windows_instances_closed": all(closed(identity) for identity in windows_seen.values()),
        "all_saved_Linux_instances_closed": all(row["instance_closed"] for row in linux_closed),
        "three_original_leases_released_and_OS_lock_reacquired": all(row.get("lease_released_utc") for row in leases)
            and read(lock.with_name("LOCK_OWNER.json"))["status"] == "released",
        "all_original_time_RSS_GPU_output_disk_guards": all(value["linux_resource"]["status"] == "completed"
            and value["linux_resource"]["peak_tree_RSS_bytes"] <= 8 * 1024**3
            and value["measurement"]["peak_CUDA_reserved_bytes"] <= 12 * 1024**3 for value in states)
            and sum(path.stat().st_size for path in OWN.rglob("*") if path.is_file()) < 8 * 1024**3
            and shutil_disk_free() >= 20 * 1024**3,
        "parent_exact31_unique_and_target50_active": parent["additional_accepted_count"] == 31
            and len(parent["accepted_mechanisms"]) == len(set(parent["accepted_mechanisms"])) == 31
            and parent["additional_target"] == 50 and parent["status"].startswith("active"),
        "old_Q65_original_loss_graph_failure_remains": read(ROOT / "research/2026-10-05-gaussian-optimization-round2/CURRENT_STATE.json")["unaccepted"]["Q65"].startswith("native_loss_CUDA_graph")
            if "unaccepted" in read(ROOT / "research/2026-10-05-gaussian-optimization-round2/CURRENT_STATE.json") else
            "native_loss_CUDA_graph_speed1.64percent_below_original10percent" in (ROOT / "research/2026-10-05-gaussian-optimization-round2/CURRENT_STATE.json").read_text(encoding="utf-8")}
    receipt = {"status": "root_negative_review_passed" if all(checks.values()) else "failed_root_review",
        "checks": checks, "current_source_and_input_identities": len(identities), "executed_source_copies": source_copies,
        "Windows_instances": list(windows_seen.values()), "Linux_instance_closure": linux_closed,
        "recalculated_stage_seconds": recalculated, "maximum_absolute_SSIM_error": comparison["maximum_absolute_SSIM_error"],
        "known_new_three_lease_execution_seconds": sum(row["child_elapsed_seconds"] for row in leases),
        "old_Q65_full_envelope_and_text_research_cost": "unmeasured_not_zero_not_used_to_accept",
        "root_review_is_independent_agent": False, "new_optimization_count": 0,
        "reviewer_instance": {"pid": os.getpid(), "creation_time": psutil.Process().create_time()},
        "actual_reviewer_seconds_before_publication": time.perf_counter() - STARTED}
    atomic_json(OWN / "NEGATIVE_REVIEW_001.json", receipt)
    if not all(checks.values()):
        raise AssertionError("A root negative review failed")
    print(json.dumps({"status": receipt["status"], "checks": checks}), flush=True)


def shutil_disk_free():
    import shutil
    return shutil.disk_usage(OWN).free


if __name__ == "__main__":
    main()
