"""Measure fresh complete phase ordering with original science and full fees."""

import argparse
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

import psutil

from equal_fee_cost_bound import equal_fee_cost_bound
from run_fused_camera_digest_campaign import compare as compare_parent
from review_scene_batch_negative import read
from scene_transform_cost_node import ROOT, CODE
from runtime import atomic_json, data_identity

OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/sip001"
COST = OUT.parent / "sclc001"


def compare(left, right):
    result = compare_parent(left, right)
    ia, ib = (read(path / "IDENTITY_TRACE.json") for path in (left, right))
    pa, pb = (read(path / "PHASE_ADAPTER_CHECKS.json") for path in (left, right))
    result["checks"].update({
        "original_four_complete_identity_phases_exact": all(ia["checks"].values()) and all(ib["checks"].values()) and [r["paths"] for r in ia["phases"]] == [r["paths"] for r in ib["phases"]],
        "both_actual_phase5_ownership_checks_and_source_versions_exact": len(pa["checks"]) == len(pb["checks"]) == 5 and all(pa["checks"].values()) and all(pb["checks"].values()) and pa["sources"] == pb["sources"],
    })
    if not all(result["checks"].values()):
        raise AssertionError("A complete phase pair changed an original protected field")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--through", type=int, choices=(1, 3), required=True)
    args = parser.parse_args()
    output = OUT / "c0"
    state_path = output / "STATE.json"
    if not all(read(OUT / "PREFLIGHT_001.json")["checks"].values()):
        raise ValueError("The current full-input phase campaign preflight did not pass")
    paths = [Path(__file__), *[CODE / name for name in ("profile_phase_isolated_scene.py", "phase_isolated_camera_prepare.py", "phase_isolated_scene_node.py", "phase_isolated_scene_preflight.py")]]
    paths += [Path(row["path"]) for row in read(OUT.parent / "scfq001/c0/RESULT.json")["sources"]]
    paths = list(dict.fromkeys(paths))
    identities = [data_identity(path) for path in paths]
    if state_path.exists():
        state = read(state_path)
        if state["status"] != "first_complete_phase_positive_validation_pending0" or state["sources"] != identities:
            raise ValueError("Only an unchanged positive complete first pair can continue")
    else:
        output.mkdir(parents=True, exist_ok=False)
        snapshot = output / "source_snapshot"
        snapshot.mkdir()
        for index, path in enumerate(paths):
            shutil.copyfile(path, snapshot / (f"{index:03d}_" + path.name))
        initial = read(OUT / "INITIAL_EXECUTION_FEES_001.json")
        fee = sum(initial["actual_measured_fee_seconds"].values())
        state = {"status": "running_registered_complete_phase_first_pair0", "sources": identities,
            "orders": [["control", "isolated"], ["isolated", "control"], ["control", "isolated"]], "pairs": [],
            "initial_known_diagnostic_and_probe_fees": initial, "initial_known_fee_seconds": fee,
            "registered_closing_execution_reserve_seconds": 10., "new_optimization_count": 0,
            "parent_accepted_count": 29, "existing_accepted_parent_costs_retained_separately_not_assumed_zero": True}
        atomic_json(state_path, state)
    for index in range(len(state["pairs"]), args.through):
        pair = output / ("t" + str(index))
        pair.mkdir(exist_ok=False)
        rows = {}
        for mode in state["orders"][index]:
            leaf, node = pair / mode, pair / (mode + "_node")
            mounted = "/mnt/c/" + leaf.relative_to(Path("C:/")).as_posix()
            command = [sys.executable, "-X", "utf8", str(CODE / "phase_isolated_scene_node.py"),
                "--kind", "camera_oracle", "--mode", "diagnostic", "--node", str(node), "--output", str(leaf), "--",
                "wsl", "-d", "E26-C2-LIV-20260907-134802", "--", "/usr/bin/env", "OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1",
                "/opt/e26_thin_surface_v1/venv/bin/python", "-B",
                "/mnt/c/lidar-camera-fusion-mask-robust-downsampling/02_code/gaussian_pipeline_optimization_v3_20261005/profile_phase_isolated_scene.py",
                "--mode", "baseline", "--fused-enabled", "1", "--identity-enabled", "1", "--phase-enabled", "1" if mode == "isolated" else "0", "--library",
                "/mnt/c/lidar-camera-fusion-mask-robust-downsampling/04_outputs/gaussian_pipeline_optimization_v3_20261005/scf001/g0/build/fused_camera_digest.so", "--output", mounted]
            started = time.perf_counter()
            with (pair / (mode + ".log")).open("x", encoding="utf-8") as stream:
                child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
                owner = {"pid": child.pid, "creation_time": psutil.Process(child.pid).create_time()}
                returncode = child.wait()
            elapsed = time.perf_counter() - started
            if returncode:
                state.update(status="failed_actual_complete_phase_node0", failed_pair=index, failed_mode=mode, actual_failed_seconds=elapsed, failed_owner=owner)
                atomic_json(state_path, state)
                raise RuntimeError("The actual phase node failed; preserve original outputs")
            native, controller = read(leaf / "RESULT.json"), read(node / "NODE.json")
            if controller["status"] != "completed" or controller["live_Windows_after_wait"] or any(r["alive"] for r in controller["actual_Linux_after_wait"]):
                raise AssertionError("An actual phase process remained after the original wait")
            rows[mode] = {"actual_spawn_to_exit_seconds": elapsed, "complete_stage_seconds": elapsed, "whole_seconds": elapsed,
                "owner": owner, "controller": controller, "Scene_load_seconds": native["Scene_load_seconds"],
                "all_pixel_and_state_check_seconds": native["all_pixel_and_state_check_seconds"]}
            atomic_json(pair / (mode + "_CLOSED.json"), rows[mode])
        started = time.perf_counter()
        comparison = compare(pair / "control", pair / "isolated")
        compared_seconds = time.perf_counter() - started
        for row in rows.values():
            row["complete_stage_seconds"] += compared_seconds
            row["whole_seconds"] += compared_seconds
        state["pairs"].append({"pair": index, "order": state["orders"][index], "rows": rows, "comparison": comparison,
            "actual_compare_seconds_charged_each_side": compared_seconds})
        atomic_json(state_path, state)
    medians = {mode: statistics.median(p["rows"][mode]["complete_stage_seconds"] for p in state["pairs"]) for mode in ("control", "isolated")}
    ratios = [p["rows"]["isolated"]["whole_seconds"] / p["rows"]["control"]["whole_seconds"] for p in state["pairs"]]
    fee = state["initial_known_fee_seconds"] + state["registered_closing_execution_reserve_seconds"]
    bound = equal_fee_cost_bound(medians["control"], medians["isolated"], fee)
    status = "first_complete_phase_positive_validation_pending0" if len(state["pairs"]) == 1 else "three_complete_phase_pairs_final_validation_and_cost_pending0"
    if bound["conservative_sample_reduction_upper_bound"] < .03 or max(ratios) > 1.1:
        status = "closed_negative_complete_phase_screen0"
    state.update(status=status, raw_complete_stage_medians=medians, raw_stage_reduction=1 - medians["isolated"] / medians["control"],
        whole_ratios=ratios, known_fee_including_registered10_reserve_seconds=fee, sampled_known_fee_cost_bound=bound,
        source_identities_unchanged=identities == [data_identity(p) for p in paths], unknown_text_research_and_future_validation_not_assumed_zero=True)
    if not state["source_identities_unchanged"]:
        raise AssertionError("A formal phase source changed during the actual complete comparison")
    atomic_json(state_path, state)
    atomic_json(output / "RESULT.json", state)
    print({"status": status, "complete_medians": medians, "raw_gain": state["raw_stage_reduction"], "known_fee_bound": bound, "pairs": len(state["pairs"]), "new": 0})


if __name__ == "__main__":
    main()
