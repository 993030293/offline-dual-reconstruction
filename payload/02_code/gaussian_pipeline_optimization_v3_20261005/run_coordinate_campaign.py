"""Run fresh complete operand-reuse pairs with preserved native contracts."""

import argparse
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

import psutil
import run_native_zero_feature_campaign as common

ROOT, CODE = common.ROOT, common.CODE
OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/oup017/R169"
read, local = common.read, common.local
from runtime import atomic_json, data_identity


def command(leaf, node, entry, mode=None):
    mounted = "/mnt/c/" + leaf.relative_to(Path("C:/")).as_posix()
    result = [sys.executable, "-X", "utf8", str(CODE / "native_coordinate_node.py"),
        "--kind", "camera_oracle", "--mode", "diagnostic", "--node", str(node), "--output", str(leaf), "--",
        "wsl", "-d", "E26-C2-LIV-20260907-134802", "--", "/usr/bin/env", "OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1",
        "LD_PRELOAD=/mnt/c/lidar-camera-fusion-mask-robust-downsampling/04_outputs/gaussian_pipeline_optimization_v3_20261005/gzi001/build/native_transfer_counter.so",
        "/opt/e26_thin_surface_v1/venv/bin/python", "-B",
        "/mnt/c/lidar-camera-fusion-mask-robust-downsampling/02_code/gaussian_pipeline_optimization_v3_20261005/" + entry,
        "--output", mounted]
    if mode is not None:
        result.extend(["--mode", "baseline", "--zero-enabled", "1" if mode == "candidate" else "0"])
    return result


def run(leaf, node, entry, mode=None):
    started = time.perf_counter()
    cmd = command(leaf, node, entry, mode)
    with leaf.with_name(leaf.name + ".log").open("x", encoding="utf-8") as stream:
        process = subprocess.Popen(cmd, stdout=stream, stderr=subprocess.STDOUT)
        owner = {"pid": process.pid, "creation_time": psutil.Process(process.pid).create_time()}
        code = process.wait(timeout=380)
    row = {"actual_spawn_to_exit_seconds": time.perf_counter() - started, "returncode": code,
           "owner": owner, "leaf": str(leaf), "node": str(node), "command": cmd}
    atomic_json(leaf.with_name(leaf.name + "_EXIT.json"), row)
    if code:
        raise RuntimeError("An owned coordinate node failed: " + str(leaf))
    controller = read(node / "NODE.json")
    if (controller["status"] != "completed" or controller["live_Windows_after_wait"]
            or any(item["alive"] for item in controller["actual_Linux_after_wait"])
            or not controller["Linux_resource_receipts"]):
        raise AssertionError("A native coordinate node did not close with resource evidence")
    row["controller"] = controller
    return row


def compare(left, right):
    a, b = [read(path / "RESULT.json") for path in (left, right)]
    ta, tb = [read(path / "COORDINATE_TRANSFER_TRACE.json") for path in (left, right)]
    aa, ab = [read(path / "COORDINATE_ADAPTER_CHECKS.json") for path in (left, right)]
    files = {name: local(a["generated_assets"][name]["path"]).read_bytes()
                   == local(b["generated_assets"][name]["path"]).read_bytes() for name in a["generated_assets"]}
    checks = {"all15_original_science_fields_1016_cameras_exact": len(a["scientific"]) == 15
                and len(a["scientific"]["rows"]) == 1016 and a["scientific"] == b["scientific"],
              "all17_sources1036_inputs_runtime_configs_exact": len(a["sources"]) == 17 and len(a["inputs"]) == 1036
                and all(a[key] == b[key] for key in ("sources", "inputs", "runtime", "runtime_config")),
              "two_generated_files_full_bytes_exact": len(files) == 2 and all(files.values()),
              "all_native_and_private_side_checks": all(all(row["checks"].values()) for row in (a, b, aa, ab)),
              "same_private_source_identities": aa["sources"] == ab["sources"],
              "original_parameter_storage_alias_groups_exact": ta["calls"][0]["parameter_storage_alias_groups"] == tb["calls"][0]["parameter_storage_alias_groups"],
              "first_point_and_color_uploads_unchanged": ta["calls"][0]["uploads"][:2] == tb["calls"][0]["uploads"],
              "only_duplicate_point_upload_removed": len(ta["calls"][0]["uploads"]) == 3 and len(tb["calls"][0]["uploads"]) == 2
                and ta["actual_native_HtoD_bytes"] - tb["actual_native_HtoD_bytes"] == ta["calls"][0]["uploads"][2]["payload_bytes"]}
    if not all(checks.values()):
        raise AssertionError("A complete native coordinate pair changed a contract: " + repr(checks))
    return {"checks": checks, "actual_generated_files": files, "new_optimization_count": 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("first", "remaining"), required=True)
    args = parser.parse_args()
    output = OUT / "c0"
    paths = [Path(__file__), *[CODE / name for name in ("native_coordinate_reuse.py", "profile_coordinate_scene.py",
        "native_coordinate_node.py", "check_coordinate_reuse.py", "native_transfer_meter.py", "native_cuda_transfer_counter.c",
        "profile_mapped_scene.py", "scene_transform_cost_node.py", "streamed_ply_node_002.py",
        "run_bounded_journal_task.py", "run_with_resource_lease.py")],
        ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/gzi001/build/native_transfer_counter.so"]
    identities = [data_identity(path) for path in paths]
    if args.stage == "first":
        output.mkdir(parents=True, exist_ok=False)
        archive = output / "source_snapshot"
        archive.mkdir()
        for index, path in enumerate(paths):
            shutil.copyfile(path, archive / (f"{index:03d}_" + path.name))
        state = {"status": "running_registered_first_pair0", "sources": identities,
                 "orders": [["incumbent", "candidate"], ["candidate", "incumbent"], ["incumbent", "candidate"]],
                 "pairs": [], "common_R157_zero_count0": True, "new_optimization_count": 0,
                 "work_primary_gate": .10, "whole_median_max_ratio": 1.03, "each_whole_max_ratio": 1.1}
        atomic_json(output / "STATE.json", state)
        component = run(OUT / "g0", OUT / "g0_node", "check_coordinate_reuse.py")
        generic = read(OUT / "g0/RESULT.json")
        if not all(generic["checks"].values()):
            raise AssertionError("The actual generic public component failed")
        state["component"] = component
    else:
        state = read(output / "STATE.json")
        if state["status"] != "first_complete_coordinate_pair_pending_validation0" or state["sources"] != identities:
            raise ValueError("Only an unchanged first complete positive pair can continue")
    for index in range(len(state["pairs"]), 1 if args.stage == "first" else 3):
        pair = output / ("t" + str(index))
        pair.mkdir(exist_ok=False)
        rows = {}
        for mode in state["orders"][index]:
            row = run(pair / mode, pair / (mode + "_node"), "profile_coordinate_scene.py", mode)
            row["whole_seconds"] = row["actual_spawn_to_exit_seconds"]
            row["actual_native_HtoD_bytes"] = read(pair / mode / "COORDINATE_TRANSFER_TRACE.json")["actual_native_HtoD_bytes"]
            rows[mode] = row
        started = time.perf_counter()
        comparison = compare(pair / "incumbent", pair / "candidate")
        compared = time.perf_counter() - started
        for row in rows.values():
            row["whole_seconds"] += compared
        state["pairs"].append({"index": index, "order": state["orders"][index], "rows": rows,
                               "comparison": comparison, "compare_seconds_charged_each_side": compared})
        atomic_json(output / "STATE.json", state)
        if rows["candidate"]["whole_seconds"] / rows["incumbent"]["whole_seconds"] > 1.1:
            state["status"] = "closed_negative_each_whole_guard0"
            break
    medians = {mode: statistics.median(pair["rows"][mode]["whole_seconds"] for pair in state["pairs"])
               for mode in ("incumbent", "candidate")}
    ratios = [pair["rows"]["candidate"]["whole_seconds"] / pair["rows"]["incumbent"]["whole_seconds"] for pair in state["pairs"]]
    if state["status"] in ("running_registered_first_pair0", "first_complete_coordinate_pair_pending_validation0"):
        state["status"] = "first_complete_coordinate_pair_pending_validation0" if len(state["pairs"]) == 1 else "three_complete_coordinate_pairs_pending_validation0"
    if len(state["pairs"]) == 3 and medians["candidate"] / medians["incumbent"] > 1.03:
        state["status"] = "closed_negative_whole_median_guard0"
    state.update(whole_medians=medians, whole_ratios=ratios,
                 raw_whole_reduction=1 - medians["candidate"] / medians["incumbent"],
                 sources_unchanged=identities == [data_identity(path) for path in paths])
    if not state["sources_unchanged"]:
        raise AssertionError("A formal coordinate source changed")
    atomic_json(output / "STATE.json", state)
    atomic_json(output / "RESULT.json", state)
    print({"status": state["status"], "whole": medians, "ratios": ratios, "new": 0}, flush=True)


if __name__ == "__main__":
    main()
