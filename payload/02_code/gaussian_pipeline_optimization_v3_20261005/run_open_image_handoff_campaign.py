"""Measure three complete entries with exact quality, source work and exit bounds."""

import argparse
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

import psutil

from run_fused_camera_digest_campaign import compare as compare_parent
from review_scene_batch_negative import read
from scene_transform_cost_node import ROOT, CODE
from runtime import atomic_json, data_identity

OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/shf001"


def compare(left, right):
    result = compare_parent(left, right)
    identities = [read(path / "IDENTITY_TRACE.json") for path in (left, right)]
    adapters = [read(path / "HANDOFF_ADAPTER_CHECKS.json") for path in (left, right)]
    work = [read(path / "SOURCE_IMAGE_OPEN_WORK.json") for path in (left, right)]
    result["checks"].update({
        "all_original_complete_identity_phases_exact": all(all(row["checks"].values()) for row in identities) and [r["paths"] for r in identities[0]["phases"]] == [r["paths"] for r in identities[1]["phases"]],
        "both_actual_handoff_contracts_and_sources_exact": all(len(row["checks"]) == 5 and all(row["checks"].values()) for row in adapters) and adapters[0]["sources"] == adapters[1]["sources"],
        "all_native_source_work_census_paths_equal": {row["path"] for row in work[0]["rows"]} == {row["path"] for row in work[1]["rows"]} and all(row["native_functions_restored"] for row in work),
    })
    if not all(result["checks"].values()):
        raise AssertionError("A complete image handoff pair changed an original protected field")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--through", type=int, choices=(1, 3), required=True)
    args = parser.parse_args()
    output = OUT / "c0"
    state_path = output / "STATE.json"
    if not all(read(OUT / "PREFLIGHT_001.json")["checks"].values()) or not all(read(OUT / "g0/RESULT.json")["checks"].values()):
        raise ValueError("The current full input and native handoff component preflight did not pass")
    paths = [Path(__file__), *[CODE / name for name in ("profile_open_image_handoff.py", "open_image_handoff.py", "image_source_open_census.py", "open_image_handoff_node.py", "open_image_handoff_preflight.py", "test_open_image_handoff.py")]]
    paths += [Path(row["path"]) for row in read(OUT.parent / "scfq001/c0/RESULT.json")["sources"]]
    paths = list(dict.fromkeys(paths))
    identities = [data_identity(path) for path in paths]
    if state_path.exists():
        state = read(state_path)
        if state["status"] != "first_complete_handoff_work_positive_validation_pending0" or state["sources"] != identities:
            raise ValueError("Only an unchanged positive complete first pair can continue")
    else:
        output.mkdir(parents=True, exist_ok=False)
        snapshot = output / "source_snapshot"
        snapshot.mkdir()
        for index, path in enumerate(paths):
            shutil.copyfile(path, snapshot / (f"{index:03d}_" + path.name))
        state = {"status": "running_registered_complete_handoff_work_first_pair0", "sources": identities,
            "orders": [["control", "handoff"], ["handoff", "control"], ["control", "handoff"]], "pairs": [],
            "primary_work_gate": .10, "raw_whole_median_ratio_gate": 1.01, "each_whole_ratio_gate": 1.1,
            "final_known_fees_work_and_seconds": "not_yet_complete_no_acceptance",
            "registered_closing_execution_reserve_seconds": 10., "new_optimization_count": 0, "parent_accepted_count": 29}
        atomic_json(state_path, state)
    for index in range(len(state["pairs"]), args.through):
        pair = output / ("t" + str(index))
        pair.mkdir(exist_ok=False)
        rows = {}
        for mode in state["orders"][index]:
            leaf, node = pair / mode, pair / (mode + "_node")
            mounted = "/mnt/c/" + leaf.relative_to(Path("C:/")).as_posix()
            command = [sys.executable, "-X", "utf8", str(CODE / "open_image_handoff_node.py"),
                "--kind", "camera_oracle", "--mode", "diagnostic", "--node", str(node), "--output", str(leaf), "--",
                "wsl", "-d", "E26-C2-LIV-20260907-134802", "--", "/usr/bin/env", "OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1",
                "/opt/e26_thin_surface_v1/venv/bin/python", "-B",
                "/mnt/c/lidar-camera-fusion-mask-robust-downsampling/02_code/gaussian_pipeline_optimization_v3_20261005/profile_open_image_handoff.py",
                "--mode", "baseline", "--fused-enabled", "1", "--identity-enabled", "1", "--handoff-enabled", "1" if mode == "handoff" else "0", "--library",
                "/mnt/c/lidar-camera-fusion-mask-robust-downsampling/04_outputs/gaussian_pipeline_optimization_v3_20261005/scf001/g0/build/fused_camera_digest.so", "--output", mounted]
            started = time.perf_counter()
            with (pair / (mode + ".log")).open("x", encoding="utf-8") as stream:
                child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
                owner = {"pid": child.pid, "creation_time": psutil.Process(child.pid).create_time()}
                returncode = child.wait()
            elapsed = time.perf_counter() - started
            if returncode:
                state.update(status="failed_actual_complete_handoff_node0", failed_pair=index, failed_mode=mode, actual_failed_seconds=elapsed, failed_owner=owner)
                atomic_json(state_path, state)
                raise RuntimeError("The actual handoff node failed; preserve all original outputs")
            native, controller = read(leaf / "RESULT.json"), read(node / "NODE.json")
            work = read(leaf / "SOURCE_IMAGE_OPEN_WORK.json")
            if controller["status"] != "completed" or controller["live_Windows_after_wait"] or any(r["alive"] for r in controller["actual_Linux_after_wait"]):
                raise AssertionError("An actual handoff process remained after the original wait")
            rows[mode] = {"actual_spawn_to_exit_seconds": elapsed, "complete_stage_seconds": elapsed, "whole_seconds": elapsed,
                "owner": owner, "controller": controller, "Scene_load_seconds": native["Scene_load_seconds"],
                "source_image_native_read_opens": work["successful_native_read_opens"]}
            atomic_json(pair / (mode + "_CLOSED.json"), rows[mode])
        started = time.perf_counter()
        comparison = compare(pair / "control", pair / "handoff")
        comparison_seconds = time.perf_counter() - started
        for row in rows.values():
            row["whole_seconds"] += comparison_seconds
            row["complete_stage_seconds"] += comparison_seconds
        state["pairs"].append({"pair": index, "order": state["orders"][index], "rows": rows, "comparison": comparison,
                               "actual_compare_seconds_charged_each_side": comparison_seconds})
        atomic_json(state_path, state)
    medians = {mode: statistics.median(p["rows"][mode]["whole_seconds"] for p in state["pairs"]) for mode in ("control", "handoff")}
    ratios = [p["rows"]["handoff"]["whole_seconds"] / p["rows"]["control"]["whole_seconds"] for p in state["pairs"]]
    reductions = [1 - p["rows"]["handoff"]["source_image_native_read_opens"] / p["rows"]["control"]["source_image_native_read_opens"] for p in state["pairs"]]
    status = "first_complete_handoff_work_positive_validation_pending0" if len(state["pairs"]) == 1 else "three_complete_handoff_work_pairs_final_validation_and_cost_pending0"
    if min(reductions) < .10 or max(ratios) > 1.1 or len(state["pairs"]) == 3 and medians["handoff"] / medians["control"] > 1.01:
        status = "closed_negative_complete_handoff_screen0"
    state.update(status=status, raw_whole_medians=medians, raw_whole_reduction=1 - medians["handoff"] / medians["control"],
                 raw_native_source_work_reductions=reductions, whole_ratios=ratios, source_identities_unchanged=identities == [data_identity(path) for path in paths])
    if not state["source_identities_unchanged"]:
        raise AssertionError("A formal handoff source changed during actual complete entries")
    atomic_json(state_path, state)
    atomic_json(output / "RESULT.json", state)
    print({"status": status, "complete_medians": medians, "work_reductions": reductions, "whole_ratios": ratios, "pairs": len(state["pairs"]), "new": 0})


if __name__ == "__main__":
    main()
