"""Measure complete ordered file identity composition without recounting parents."""

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

OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/scfq001"
PARENT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/scf001"


def compare(left, right):
    result = compare_parent(left, right)
    a, b = (read(path / "IDENTITY_TRACE.json") for path in (left, right))
    result["checks"].update({
        "both_complete_identity_phase5_checks": len(a["checks"]) == len(b["checks"]) == 5 and all(a["checks"].values()) and all(b["checks"].values()),
        "all_four_identity_lists_same_order": [p["paths"] for p in a["phases"]] == [p["paths"] for p in b["phases"]],
        "both_sides_same_R153_only_identity_scheduling_changes": all(not p["enabled"] for p in a["phases"]) and all(p["enabled"] for p in b["phases"])})
    if not all(result["checks"].values()):
        raise AssertionError("The complete file identity composition changed an original contract")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--through", type=int, choices=(1, 3), required=True)
    parser.add_argument("--preflight-command-seconds", type=float, required=True)
    args = parser.parse_args()
    output = OUT / "c0"
    state_path = output / "STATE.json"
    library = PARENT / "g0/build/fused_camera_digest.so"
    current = read(ROOT / "research/2026-10-05-gaussian-optimization-round3-50/CURRENT_STATE.json")
    if current["additional_accepted_count"] != 29 or not current["R153_algorithm_accepted"]:
        raise ValueError("The declared existing parent must remain accepted at 29")
    if args.preflight_command_seconds < 0 or not all(read(OUT / "PREFLIGHT_001.json")["checks"].values()):
        raise ValueError("The current complete environment preflight did not pass")
    paths = [Path(__file__), CODE / "profile_q68_fused_fingerprint.py", CODE / "q68_fused_fingerprint_node.py",
        CODE / "q68_fused_preflight.py", CODE / "parallel_b_005/ordered_identity.py"]
    paths += [Path(row["path"]) for row in read(PARENT / "c0/RESULT.json")["sources"]]
    paths = list(dict.fromkeys(paths))
    identities = [data_identity(path) for path in paths]
    if state_path.exists():
        state = read(state_path)
        if state["status"] != "first_complete_composition_positive_validation_pending0" or state["sources"] != identities:
            raise ValueError("Only an unchanged positive complete first pair can continue")
    else:
        output.mkdir(exist_ok=False)
        snapshots = output / "source_snapshot"
        snapshots.mkdir()
        for index, path in enumerate(paths):
            shutil.copyfile(path, snapshots / (f"{index:03d}_" + path.name))
        state = {"status": "running_registered_complete_composition_first_pair0", "sources": identities,
            "orders": [["control", "combo"], ["combo", "control"], ["control", "combo"]], "pairs": [],
            "initial_necessary_integration_fees": {"actual_preflight_command_seconds": args.preflight_command_seconds,
                "actual_first_failed_entry_command_seconds": read(OUT / "FAILED_WINDOWS_ENTRY_COST_001.json")["actual_failed_command_seconds"],
                "actual_failed_source_copy_command_seconds": read(OUT / "FAILED_WINDOWS_ENTRY_COST_001.json")["actual_copy_command_seconds"],
                "actual_source_syntax_and_generated_boundaries_check_seconds": read(OUT / "VALIDATED_SYNTAX_001.json")["actual_command_seconds"],
                "registered_closing_source_and_cost_execution_reserve_seconds": 10.},
            "existing_parent_cost_receipt": str(PARENT / "FINAL_COST_001.json"),
            "existing_Q68_acceptance_receipt": str(OUT.parent / "ordered_identity_acceptance_001/ACCEPTANCE.json"),
            "no_new_library_build_existing_parents_not_recounted": True,
            "fee_scope_marginal_composition_all_new_first_failures_validation_recovery_and_review": True,
            "new_optimization_count": 0}
        atomic_json(state_path, state)
    for index in range(len(state["pairs"]), args.through):
        pair = output / ("t" + str(index))
        pair.mkdir(exist_ok=False)
        rows = {}
        for mode in state["orders"][index]:
            leaf, node = pair / mode, pair / (mode + "_node")
            mounted = "/mnt/c/" + leaf.relative_to(Path("C:/")).as_posix()
            command = [sys.executable, "-X", "utf8", str(CODE / "q68_fused_fingerprint_node.py"), "--kind", "camera_oracle", "--mode", "diagnostic",
                "--node", str(node), "--output", str(leaf), "--", "wsl", "-d", "E26-C2-LIV-20260907-134802", "--", "/usr/bin/env",
                "OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1", "/opt/e26_thin_surface_v1/venv/bin/python", "-B",
                "/mnt/c/lidar-camera-fusion-mask-robust-downsampling/02_code/gaussian_pipeline_optimization_v3_20261005/profile_q68_fused_fingerprint.py",
                "--mode", "baseline", "--fused-enabled", "1", "--identity-enabled", "1" if mode == "combo" else "0",
                "--library", "/mnt/c/lidar-camera-fusion-mask-robust-downsampling/04_outputs/gaussian_pipeline_optimization_v3_20261005/scf001/g0/build/fused_camera_digest.so",
                "--output", mounted]
            started = time.perf_counter()
            with (pair / (mode + ".log")).open("x", encoding="utf-8") as stream:
                child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
                owner = {"pid": child.pid, "creation_time": psutil.Process(child.pid).create_time()}
                returncode = child.wait()
            elapsed = time.perf_counter() - started
            if returncode:
                state.update(status="failed_actual_complete_composition_node0", failed_pair=index, failed_mode=mode,
                    actual_failed_seconds=elapsed, actual_failed_owner=owner)
                atomic_json(state_path, state)
                raise RuntimeError("The composed full node failed; preserve its original output")
            result, controller = read(leaf / "RESULT.json"), read(node / "NODE.json")
            if controller["status"] != "completed" or controller["live_Windows_after_wait"] or any(row["alive"] for row in controller["actual_Linux_after_wait"]):
                raise AssertionError("An actual complete composition worker did not close")
            rows[mode] = {"actual_spawn_to_exit_seconds": elapsed, "complete_stage_seconds": elapsed,
                "whole_seconds": elapsed, "owner": owner, "controller": controller,
                "Scene_load_seconds": result["Scene_load_seconds"], "all_pixel_and_state_check_seconds": result["all_pixel_and_state_check_seconds"]}
            atomic_json(pair / (mode + "_CLOSED.json"), rows[mode])
        started = time.perf_counter()
        comparison = compare(pair / "control", pair / "combo")
        compare_seconds = time.perf_counter() - started
        for row in rows.values():
            row["complete_stage_seconds"] += compare_seconds
            row["whole_seconds"] += compare_seconds
        state["pairs"].append({"pair": index, "order": state["orders"][index], "rows": rows, "comparison": comparison,
            "actual_compare_seconds_charged_each_side": compare_seconds})
        atomic_json(state_path, state)
    medians = {mode: statistics.median(pair["rows"][mode]["complete_stage_seconds"] for pair in state["pairs"]) for mode in ("control", "combo")}
    fee = sum(state["initial_necessary_integration_fees"].values())
    bound = equal_fee_cost_bound(medians["control"], medians["combo"], fee)
    ratios = [pair["rows"]["combo"]["whole_seconds"] / pair["rows"]["control"]["whole_seconds"] for pair in state["pairs"]]
    status = "first_complete_composition_positive_validation_pending0" if len(state["pairs"]) == 1 else "three_complete_composition_pairs_validation_and_final_cost_pending0"
    if bound["conservative_sample_reduction_upper_bound"] < .03 or max(ratios) > 1.1:
        status = "closed_negative_complete_composition_cost_or_whole_screen0"
    state.update(status=status, raw_complete_stage_medians=medians, raw_stage_reduction=1 - medians["combo"] / medians["control"],
        whole_ratios=ratios, initial_necessary_integration_fee_seconds=fee, sampled_known_fee_cost_bound=bound,
        source_identities_unchanged=identities == [data_identity(path) for path in paths],
        unknown_text_parent_tail_and_future_validation_not_assumed_zero=True)
    if not state["source_identities_unchanged"]:
        raise AssertionError("An actual composition campaign source changed")
    atomic_json(state_path, state)
    atomic_json(output / "RESULT.json", state)
    print({"status": status, "complete_medians": medians, "known_fee_bound": bound, "pairs": len(state["pairs"]), "new": 0})


if __name__ == "__main__":
    main()
