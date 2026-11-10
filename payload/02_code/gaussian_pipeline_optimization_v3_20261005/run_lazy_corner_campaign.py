"""Run three registered fresh complete RAW entry pairs under full wall gates."""

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
OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/lco001"
sys.path[:0] = [str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"),
    str(ROOT / "02_code/gaussian_pipeline_optimization_v2_20261005")]
from runtime import atomic_json, data_identity
from raw_initial_invariants import compare_initial_outputs


def read(path):
    import json
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--through", type=int, choices=(1, 3), required=True)
    args = parser.parse_args()
    output = OUT / "c0"
    protocol_old = read(OUT.parent / "vector_gray_stats_001/c0/PROTOCOL.json")
    reference = Path(protocol_old["reference_initial"])
    paths = [Path(__file__), CODE / "run_lazy_corner_initial.py", CODE / "lazy_corner_execution.py",
        CODE / "lazy_corner_order.py", CODE / "native_lazy_corner_order.cpp", CODE / "profile_lazy_corner_scoring.py",
        CODE / "check_lazy_corner_order.py", CODE / "build_lazy_corner_order.py",
        OUT / "g0/build/native_lazy_corner_order.dll"]
    paths += [Path(row["path"]) for row in protocol_old["sources"]]
    paths = list(dict.fromkeys(paths))
    identities = [data_identity(path) for path in paths]
    if (not all(read(OUT / "g0/RESULT.json")["checks"].values())
            or read(OUT / "p0/RESULT.json")["status"] != "complete128_screen_positive_complete7338_pending"
            or not all(read(OUT / "p0/PILOT_ORDER_ADAPTER.json")["checks"].values())):
        raise ValueError("The current full RAW consumer prerequisites failed")
    state_path = output / "STATE.json"
    if state_path.exists():
        state = read(state_path)
        if state["status"] != "first_full_RAW_pair_positive_remaining_fresh_and_quality_pending0" or state["sources"] != identities:
            raise ValueError("Only the unchanged positive complete first pair can resume")
    else:
        output.mkdir(parents=True, exist_ok=False)
        snapshot = output / "source_snapshot"
        snapshot.mkdir()
        for index, path in enumerate(paths):
            if path.suffix != ".dll":
                (snapshot / (f"{index:03d}_" + path.name)).write_bytes(path.read_bytes())
        fees = {name + "_actual_lease": read(OUT / (name + "_lease.json"))["child_elapsed_seconds"] for name in ("g0", "p0")}
        fees.update(conservative_source_metadata_execution_reserve=15., conservative_closing_execution_reserve=20.)
        state = {"status": "running_registered_full_RAW_first_pair0", "sources": identities,
            "orders": [["native", "lazy"], ["lazy", "native"], ["native", "lazy"]], "pairs": [], "runs": [],
            "primary_complete_affected_initial_entry_wall_including_all_compare_and_actual_exit": True,
            "original_denominators": [7338, 6797, 1002, 11583], "known_extra_costs": fees,
            "net_gate": .02, "raw_and_adjusted_whole_median_ratio_gate": 1.01, "each_whole_ratio_gate": 1.1,
            "all_original_current_vector_statistics_on_both_sides": True,
            "source_reference_reused_observed_development_not_fresh_blind": str(reference),
            "new_optimization_count": 0, "parent_accepted_count": 33}
        atomic_json(state_path, state)
    began_group = time.perf_counter()
    for trial in range(len(state["pairs"]), args.through):
        folder = output / ("t" + str(trial))
        folder.mkdir(exist_ok=False)
        rows = {}
        for mode in state["orders"][trial]:
            if identities != [data_identity(path) for path in paths] or time.perf_counter() - began_group > 1200:
                raise ValueError("The frozen full RAW sources or group budget changed")
            leaf = folder / mode
            cache = folder / (mode + "_quality")
            command = [sys.executable, "-X", "utf8", "-B", str(CODE / "run_bounded_journal_task.py"),
                "--receipt", str(folder / (mode + "_GUARD.json")), "--log", str(folder / (mode + ".log")),
                "--maximum-seconds", "600", "--maximum-rss-gib", "8", "--", sys.executable, "-X", "utf8", "-B",
                str(CODE / "run_lazy_corner_initial.py"), "--lazy-enabled", "1" if mode == "lazy" else "0",
                "--order-library", str(OUT / "g0/build/native_lazy_corner_order.dll"), "--mode", "vector",
                "--library", protocol_old["library"], "--gray-library", protocol_old["gray_library"],
                "--corner-library", protocol_old["corner_library"], "--output", str(leaf), "--quality-cache", str(cache)]
            begin = time.perf_counter()
            with (folder / (mode + "_controller.log")).open("x", encoding="utf-8") as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                owner = {"pid": child.pid, "creation_time": psutil.Process(child.pid).create_time()}
                returncode = child.wait()
            elapsed = time.perf_counter() - begin
            row = {"mode": mode, "trial": trial, "owner": owner, "returncode": returncode,
                "actual_spawn_to_exit_seconds": elapsed, "command": command}
            state["runs"].append(row)
            atomic_json(state_path, state)
            if returncode:
                state.update(status="failed_actual_full_RAW_worker0", failed_mode=mode, failed_trial=trial)
                atomic_json(output / "FAILURE.json", state)
                raise RuntimeError("An actual complete RAW worker failed")
            guard, priority = read(folder / (mode + "_GUARD.json")), read(leaf / "PRIORITY_INPUT.json")
            receipt = read(leaf / "initial/SELECTION_RECEIPT.json")
            if guard["returncode"] or not all(priority["checks"].values()):
                raise AssertionError("The full RAW priority owner failed its actual closed entry checks")
            rows[mode] = dict(row, guard=guard, priority=priority, selection=receipt,
                complete_stage_seconds=elapsed, whole_seconds=elapsed)
            if sum(path.stat().st_size for path in OUT.rglob("*") if path.is_file()) > 12 * 1024**3 or shutil.disk_usage(OUT).free < 20 * 1024**3:
                raise RuntimeError("The registered full RAW output or disk budget was exceeded")
            print({"completed_full7338_leaf": mode, "trial": trial, "whole_seconds": elapsed}, flush=True)
        begin = time.perf_counter()
        invariant = {"pair": compare_initial_outputs(folder / "native/initial", folder / "lazy/initial"),
            "native_reference": compare_initial_outputs(reference, folder / "native/initial"),
            "lazy_reference": compare_initial_outputs(reference, folder / "lazy/initial")}
        left, right = folder / "native_quality", folder / "lazy_quality"
        names = sorted(path.name for path in left.iterdir())
        checks = {"all_original_full_decisions1002_JPEG_K_poses_pairs_and_reference_exact": all(row["all_protected"] for row in invariant.values()),
            "all7338_original_quality_cache_files_byte_exact": len(names) == 7338 and names == sorted(path.name for path in right.iterdir())
                and all((left / name).read_bytes() == (right / name).read_bytes() for name in names),
            "both_original_full_denominators_and_cold_cache": all((row["selection"]["raw_frame_count"], row["selection"]["quality_and_coarse_pose_eligible_count"],
                row["selection"]["selected_count"], row["selection"]["matching_candidate_pair_count"], row["selection"]["timings"]["raw_image_scoring_detail"]["cache_hits"])
                == (7338, 6797, 1002, 11583, 0) for row in rows.values()),
            "each_complete_whole_ratio_within1point1": rows["lazy"]["whole_seconds"] <= 1.1 * rows["native"]["whole_seconds"]}
        compared = time.perf_counter() - begin
        for row in rows.values():
            row["complete_stage_seconds"] += compared
            row["whole_seconds"] += compared
        state["pairs"].append({"trial": trial, "order": state["orders"][trial], "rows": rows,
            "invariants": invariant, "checks": checks, "actual_full_comparison_seconds_charged_each_side": compared})
        atomic_json(state_path, state)
        if not all(checks.values()):
            state.update(status="closed_negative_full_RAW_protection_failed0")
            break
    medians = {mode: statistics.median(pair["rows"][mode]["whole_seconds"] for pair in state["pairs"]) for mode in ("native", "lazy")}
    upper = (medians["native"] - medians["lazy"]) / (medians["native"] + sum(state["known_extra_costs"].values()) / 6)
    if state["status"].startswith("running") or state["status"].startswith("first_full"):
        state["status"] = "first_full_RAW_pair_positive_remaining_fresh_and_quality_pending0" if len(state["pairs"]) == 1 else "three_fresh_full_RAW_pairs_quality_recovery_cost_review_pending0"
        if upper < .02 or medians["lazy"] / medians["native"] > 1.01:
            state["status"] = "closed_negative_full_RAW_registered_cost_screen0"
    state.update(raw_complete_medians=medians, raw_complete_reduction=1 - medians["lazy"] / medians["native"],
        net_complete_reduction_upper_bound_not_final_acceptance=upper,
        sources_unchanged=identities == [data_identity(path) for path in paths])
    atomic_json(state_path, state)
    atomic_json(output / "RESULT.json", state)
    print({"status": state["status"], "full_pairs": len(state["pairs"]), "medians": medians, "net_upper": upper, "new_count": 0}, flush=True)


if __name__ == "__main__":
    main()
