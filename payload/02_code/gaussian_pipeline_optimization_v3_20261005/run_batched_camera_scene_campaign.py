"""Measure registered complete Scene pairs and screen actual lower-bound costs."""

import argparse
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

import psutil

from scene_transform_cost_node import ROOT, CODE, OUT
from equal_fee_cost_bound import equal_fee_cost_bound
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def complete_cost_screen(medians, known_fee_seconds):
    bound = equal_fee_cost_bound(medians["native"], medians["batch"], known_fee_seconds)
    return {"complete_stage_medians_lower_bound_seconds": {
                "native": bound["native_lower_cost_seconds"],
                "batch": bound["candidate_lower_cost_seconds"]},
            "complete_stage_signed_reduction_at_known_lower_fee":
                bound["signed_reduction_at_known_lower_fee"],
            "complete_stage_reduction_upper_bound":
                bound["conservative_sample_reduction_upper_bound"]}


def compare(left, right):
    native, batch = read(left / "RESULT.json"), read(right / "RESULT.json")
    a, b = read(left / "BATCH_TRACE.json"), read(right / "BATCH_TRACE.json")
    differences = [key for key in set(native["scientific"]) | set(batch["scientific"])
                   if native["scientific"].get(key) != batch["scientific"].get(key)]
    byte_equal = {key: Path(native["generated_assets"][key]["path"].replace("/mnt/c/", "C:/")).read_bytes()
                  == Path(batch["generated_assets"][key]["path"].replace("/mnt/c/", "C:/")).read_bytes()
                  for key in native["generated_assets"]}
    checks = {"all15_full_scientific_fields_exact": len(native["scientific"]) == 15 and not differences,
              "all_original11_and_new4_side_checks": all(native["checks"].values()) and all(batch["checks"].values())
                and all(a["checks"].values()) and all(b["checks"].values()),
              "same_complete_input_source_runtime_config": native["inputs"] == batch["inputs"]
                and native["sources"] == batch["sources"] and native["runtime"] == batch["runtime"]
                and native["runtime_config"] == batch["runtime_config"] and a["sources"] == b["sources"],
              "both_generated_files_entire_bytes_exact": len(byte_equal) == 2 and all(byte_equal.values())}
    if not all(checks.values()):
        raise AssertionError("A registered full Scene comparison changed scientific fields: " + repr(differences))
    return {"checks": checks, "scientific_fields": 15, "complete_camera_rows": len(native["scientific"]["rows"]),
            "generated_file_byte_comparisons": byte_equal, "different_scientific_keys": differences}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--through", type=int, choices=(1, 3), required=True)
    parser.add_argument("--output", type=Path, default=OUT / "c0")
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(OUT.resolve())
    state_path = output / "STATE.json"
    sources = [Path(__file__), *[CODE / name for name in (
        "batched_camera_transforms.py", "batched_camera_execution.py", "profile_batched_camera_scene.py",
        "batched_camera_scene_node.py", "scene_transform_cost_node.py", "profile_scene_transform_cost.py",
        "profile_mapped_scene.py", "mapped_image_linux_guard.py", "streamed_ply_node_002.py",
        "run_bounded_journal_task.py", "run_with_resource_lease.py", "equal_fee_cost_bound.py")]]
    sources += [Path(row["path"].replace("/mnt/c/", "C:/")) for row in read(OUT / "p0/RESULT.json")["sources"]]
    identities = [data_identity(p) for p in sources]
    started = time.perf_counter()
    if state_path.exists():
        state = read(state_path)
        if state["sources"] != identities or state["status"] != "first_full_pair_passed_remaining_checks_pending":
            raise ValueError("Only the exact registered source and positive first pair may continue")
    else:
        output.mkdir(exist_ok=False)
        snapshot = output / "source_snapshot"
        snapshot.mkdir()
        for index, p in enumerate(sources):
            shutil.copyfile(p, snapshot / (f"{index:03d}_" + p.name))
        state = {"status": "running_first_full_pair", "orders": [["native", "batch"], ["batch", "native"], ["native", "batch"]],
                 "sources": identities, "pairs": [], "new_optimization_count": 0,
                 "initial_known_family_fees": {"diagnostic": read(OUT / "p0_lease.json")["child_elapsed_seconds"],
                     "semantic_probe": read(OUT / "g0_lease.json")["child_elapsed_seconds"],
                     "first_failed_actual_full_consumer_preserved": read(OUT / "c0_first_lease.json")["child_elapsed_seconds"],
                     "failed_g1_regression_API_reader": read(OUT / "g1_lease.json")["child_elapsed_seconds"],
                     "actual_g2_prefetch_binding_regression": read(OUT / "g2_lease.json")["child_elapsed_seconds"],
                     "actual_preflight_compile_and_task_read_command": 5.43143,
                     "actual_saved_Scene_and_source_audit_command": .3286287,
                     "prior_full_profile_compile_actual_command": .1651998}}
        atomic_json(state_path, state)
    for index in range(len(state["pairs"]), args.through):
        pair = output / ("t" + str(index))
        pair.mkdir()
        rows = {}
        for mode in state["orders"][index]:
            leaf, node = pair / mode, pair / (mode + "_node")
            mounted = "/mnt/c/" + str(leaf.relative_to(Path("C:/"))).replace("\\", "/")
            child = ["wsl", "-d", "E26-C2-LIV-20260907-134802", "--", "/usr/bin/env",
                "OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1", "/opt/e26_thin_surface_v1/venv/bin/python", "-B",
                "/mnt/c/lidar-camera-fusion-mask-robust-downsampling/02_code/gaussian_pipeline_optimization_v3_20261005/profile_batched_camera_scene.py",
                "--mode", "baseline", "--batch-enabled", "1" if mode == "batch" else "0", "--output", mounted]
            command = [sys.executable, "-X", "utf8", str(CODE / "batched_camera_scene_node.py"),
                "--kind", "camera_oracle", "--mode", "diagnostic", "--node", str(node), "--output", str(leaf), "--", *child]
            begin = time.perf_counter()
            with (pair / (mode + ".log")).open("x", encoding="utf-8") as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                owner = {"pid": process.pid, "creation_time": psutil.Process(process.pid).create_time()}
                returncode = process.wait()
            elapsed = time.perf_counter() - begin
            if returncode:
                state.update(status="failed_original_complete_node", failed_pair=index, failed_mode=mode,
                             actual_failed_side_seconds=elapsed, actual_failed_side_owner=owner)
                atomic_json(state_path, state)
                raise RuntimeError("The registered actual complete Scene side failed; no replacement run")
            result, controller = read(leaf / "RESULT.json"), read(node / "NODE.json")
            if controller["status"] != "completed" or controller["live_Windows_after_wait"] or any(p["alive"] for p in controller["actual_Linux_after_wait"]):
                raise AssertionError("An actual complete Scene controller has not closed")
            rows[mode] = {"actual_spawn_to_exit_seconds": elapsed, "owner": owner,
                          "native_Scene_load_seconds": result["Scene_load_seconds"],
                          "complete_stage_seconds": elapsed, "whole_seconds": elapsed, "controller": controller}
            atomic_json(pair / (mode + "_CLOSED.json"), rows[mode])
        begin = time.perf_counter()
        comparison = compare(pair / "native", pair / "batch")
        compare_seconds = time.perf_counter() - begin
        for row in rows.values():
            row["complete_stage_seconds"] += compare_seconds
            row["whole_seconds"] += compare_seconds
        state["pairs"].append({"pair": index, "order": state["orders"][index], "rows": rows,
                               "comparison": comparison, "actual_compare_seconds_charged_each_side": compare_seconds})
        atomic_json(state_path, state)
    medians = {mode: statistics.median(p["rows"][mode]["complete_stage_seconds"] for p in state["pairs"])
               for mode in ("native", "batch")}
    known = sum(state["initial_known_family_fees"].values())
    screen = complete_cost_screen(medians, known)
    upper = screen["complete_stage_reduction_upper_bound"]
    status = "first_full_pair_passed_remaining_checks_pending" if len(state["pairs"]) == 1 else "three_full_pairs_cost_and_remaining_checks_pending"
    if upper < .03:
        status = "closed_negative_current_complete_stage_cost_screen_below3percent0"
    state.update(status=status, raw_complete_stage_medians=medians,
        raw_stage_reduction=1 - medians["batch"] / medians["native"],
        initial_known_family_fee_lower_bound_seconds=known,
        **screen,
        this_campaign_entry_before_result_seconds=time.perf_counter() - started,
        source_identities_unchanged=identities == [data_identity(p) for p in sources],
        unmetered_text_and_campaign_parent_tail_not_assumed_zero=True)
    if not state["source_identities_unchanged"]:
        raise AssertionError("A complete Scene campaign source changed")
    atomic_json(state_path, state)
    atomic_json(output / "RESULT.json", state)
    print(json.dumps({"status": status, "complete_medians": medians, "raw_stage_reduction": state["raw_stage_reduction"],
                      "known_fee_stage_upper": upper, "new_optimization_count": 0}))


if __name__ == "__main__":
    main()
