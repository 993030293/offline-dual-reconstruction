"""Measure complete alternating Scene entries and successful native payloads."""

import argparse
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parent
OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/gzi001"
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity


def read(path):
    import json
    return json.loads(Path(path).read_text(encoding="utf-8"))


def local(path):
    return Path(str(path).replace("/mnt/c/", "C:/"))


def compare(left, right):
    a, b = [read(path / "RESULT.json") for path in (left, right)]
    ta, tb = [read(path / "ZERO_TRANSFER_TRACE.json") for path in (left, right)]
    aa, ab = [read(path / "ZERO_ADAPTER_CHECKS.json") for path in (left, right)]
    bytes_equal = {name: local(a["generated_assets"][name]["path"]).read_bytes()
                        == local(b["generated_assets"][name]["path"]).read_bytes() for name in a["generated_assets"]}
    checks = {"all15_original_science_fields_1016_cameras_exact": len(a["scientific"]) == 15
               and len(a["scientific"]["rows"]) == 1016 and a["scientific"] == b["scientific"],
              "all_original_source_input_runtime_configs_exact": len(a["sources"]) == 17 and len(a["inputs"]) == 1036
               and all(a[key] == b[key] for key in ("sources", "inputs", "runtime", "runtime_config")),
              "two_real_generated_files_full_bytes_exact": len(bytes_equal) == 2 and all(bytes_equal.values()),
              "all_original_and_private_side_checks": all(all(row["checks"].values()) for row in (a, b, aa, ab)),
              "same_all_private_source_identities": aa["sources"] == ab["sources"],
              "original_parameter_storage_alias_groups_exact": ta["calls"][0]["parameter_storage_alias_groups"] == tb["calls"][0]["parameter_storage_alias_groups"],
              "all_three_nonzero_uploads_preserved": ta["calls"][0]["uploads"][:2] + ta["calls"][0]["uploads"][3:] == tb["calls"][0]["uploads"],
              "native_zero_upload_only_removed": len(ta["calls"][0]["uploads"]) == 4 and len(tb["calls"][0]["uploads"]) == 3}
    if not all(checks.values()):
        raise AssertionError("The complete zero-feature pair changed an original contract: " + repr(checks))
    return {"checks": checks, "actual_generated_files": bytes_equal, "new_optimization_count": 0}


def command(leaf, node, mode):
    mounted = "/mnt/c/" + leaf.relative_to(Path("C:/")).as_posix()
    return [sys.executable, "-X", "utf8", str(CODE / "native_zero_feature_node.py"),
        "--kind", "camera_oracle", "--mode", "diagnostic", "--node", str(node), "--output", str(leaf), "--",
        "wsl", "-d", "E26-C2-LIV-20260907-134802", "--", "/usr/bin/env", "OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1",
        "LD_PRELOAD=/mnt/c/lidar-camera-fusion-mask-robust-downsampling/04_outputs/gaussian_pipeline_optimization_v3_20261005/gzi001/build/native_transfer_counter.so",
        "/opt/e26_thin_surface_v1/venv/bin/python", "-B",
        "/mnt/c/lidar-camera-fusion-mask-robust-downsampling/02_code/gaussian_pipeline_optimization_v3_20261005/profile_native_feature_zero.py",
        "--mode", "baseline", "--zero-enabled", "1" if mode == "direct" else "0", "--output", mounted]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--through", type=int, choices=(1, 3), required=True)
    args = parser.parse_args()
    output = OUT / "c0"
    paths = [Path(__file__), *[CODE / name for name in ("profile_native_feature_zero.py", "native_feature_zero.py",
        "native_feature_zero_v2.py", "native_transfer_meter.py", "native_cuda_transfer_counter.c",
        "native_zero_feature_node.py", "profile_mapped_scene.py", "scene_transform_cost_node.py",
        "streamed_ply_node_002.py", "run_bounded_journal_task.py", "run_with_resource_lease.py")],
        OUT / "build/native_transfer_counter.so"]
    sources = [data_identity(path) for path in paths]
    if (output / "STATE.json").exists():
        state = read(output / "STATE.json")
        if state["sources"] != sources or state["status"] != "first_complete_zero_feature_pair_pending_validation0":
            raise ValueError("Only an unchanged positive first complete pair can continue")
    else:
        output.mkdir(parents=True, exist_ok=False)
        copies = output / "source_snapshot"
        copies.mkdir()
        for index, path in enumerate(paths):
            shutil.copyfile(path, copies / (f"{index:03d}_" + path.name))
        state = {"status": "running_registered_complete_zero_pairs0", "sources": sources,
                 "orders": [["native", "direct"], ["direct", "native"], ["native", "direct"]],
                 "pairs": [], "primary_work_gate": .2, "whole_median_maximum_ratio": 1.01,
                 "each_whole_maximum_ratio": 1.1, "new_optimization_count": 0}
    for index in range(len(state["pairs"]), args.through):
        pair = output / ("t" + str(index))
        pair.mkdir(exist_ok=False)
        rows = {}
        for mode in state["orders"][index]:
            leaf, node = pair / mode, pair / (mode + "_node")
            begin = time.perf_counter()
            with (pair / (mode + ".log")).open("x", encoding="utf-8") as stream:
                child = subprocess.Popen(command(leaf, node, mode), stdout=stream, stderr=subprocess.STDOUT)
                owner = {"pid": child.pid, "creation_time": psutil.Process(child.pid).create_time()}
                code = child.wait()
            elapsed = time.perf_counter() - begin
            if code:
                state.update(status="failed_complete_native_zero_node0", failed_pair=index, failed_mode=mode,
                             failed_elapsed_seconds=elapsed, failed_owner=owner)
                atomic_json(output / "STATE.json", state)
                raise RuntimeError("The complete native zero-feature node failed; preserve original artifacts")
            controller = read(node / "NODE.json")
            trace = read(leaf / "ZERO_TRANSFER_TRACE.json")
            if controller["status"] != "completed" or controller["live_Windows_after_wait"] or any(row["alive"] for row in controller["actual_Linux_after_wait"]):
                raise AssertionError("An owned zero-feature process did not close")
            rows[mode] = {"whole_seconds": elapsed, "complete_stage_seconds": elapsed, "actual_spawn_to_exit_seconds": elapsed,
                          "owner": owner, "controller": controller,
                          "actual_native_HtoD_bytes": trace["actual_native_HtoD_bytes"]}
        begin = time.perf_counter()
        comparison = compare(pair / "native", pair / "direct")
        compared = time.perf_counter() - begin
        for row in rows.values():
            row["whole_seconds"] += compared
            row["complete_stage_seconds"] += compared
        state["pairs"].append({"index": index, "order": state["orders"][index], "rows": rows,
                               "comparison": comparison, "compare_seconds_charged_each_side": compared})
        atomic_json(output / "STATE.json", state)
        ratio = rows["direct"]["whole_seconds"] / rows["native"]["whole_seconds"]
        if ratio > 1.1:
            state.update(status="closed_negative_whole_pair_guard0", failed_pair=index)
            break
    medians = {mode: statistics.median(p["rows"][mode]["whole_seconds"] for p in state["pairs"]) for mode in ("native", "direct")}
    ratios = [p["rows"]["direct"]["whole_seconds"] / p["rows"]["native"]["whole_seconds"] for p in state["pairs"]]
    if state["status"] == "running_registered_complete_zero_pairs0" or state["status"] == "first_complete_zero_feature_pair_pending_validation0":
        state["status"] = "first_complete_zero_feature_pair_pending_validation0" if len(state["pairs"]) == 1 else "three_complete_zero_feature_pairs_pending_final_validation0"
    if len(state["pairs"]) == 3 and medians["direct"] / medians["native"] > 1.01:
        state["status"] = "closed_negative_whole_median_guard0"
    state.update(whole_medians=medians, whole_ratios=ratios,
                 raw_whole_reduction=1 - medians["direct"] / medians["native"],
                 physical_sources_unchanged=sources == [data_identity(path) for path in paths])
    if not state["physical_sources_unchanged"]:
        raise AssertionError("An actual complete zero-feature source changed")
    atomic_json(output / "STATE.json", state)
    atomic_json(output / "RESULT.json", state)
    print({"status": state["status"], "whole": medians, "ratios": ratios, "new": 0})


if __name__ == "__main__":
    main()
