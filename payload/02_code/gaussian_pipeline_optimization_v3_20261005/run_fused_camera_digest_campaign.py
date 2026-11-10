"""Run registered complete Scene pairs with exact pixels and bounded full costs."""

import argparse
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

import psutil

from equal_fee_cost_bound import equal_fee_cost_bound
from review_scene_batch_negative import read
from scene_transform_cost_node import ROOT, CODE
from runtime import atomic_json, data_identity

OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/scf001"


def compare(left, right):
    a, b = (read(path / "RESULT.json") for path in (left, right))
    ta, tb = (read(path / "FINGERPRINT_TRACE.json") for path in (left, right))
    differences = [key for key in set(a["scientific"]) | set(b["scientific"]) if a["scientific"].get(key) != b["scientific"].get(key)]
    byte_equal = {name: Path(a["generated_assets"][name]["path"].replace("/mnt/c/", "C:/")).read_bytes()
                        == Path(b["generated_assets"][name]["path"].replace("/mnt/c/", "C:/")).read_bytes() for name in a["generated_assets"]}
    checks = {"all15_scientific_fields_full1016_exact": len(a["scientific"]) == 15 and len(a["scientific"]["rows"]) == 1016 and not differences,
              "all_original11_and_private5_side_checks": all(a["checks"].values()) and all(b["checks"].values()) and all(ta["checks"].values()) and all(tb["checks"].values()),
              "same_complete_inputs17_sources_runtime_config_and_own_sources": a["inputs"] == b["inputs"] and a["sources"] == b["sources"]
                and a["runtime"] == b["runtime"] and a["runtime_config"] == b["runtime_config"] and ta["sources"] == tb["sources"],
              "two_actual_generated_files_full_bytes_exact": len(byte_equal) == 2 and all(byte_equal.values())}
    if not all(checks.values()):
        raise AssertionError("An actual complete pair changed its original science: " + repr((checks, differences)))
    return {"checks": checks, "complete_camera_rows": 1016, "different_scientific_keys": differences, "generated_file_bytes_equal": byte_equal}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--through", type=int, choices=(1, 3), required=True)
    args = parser.parse_args()
    output = OUT / "c0"
    state_path = output / "STATE.json"
    library = OUT / "g0/build/fused_camera_digest.so"
    paths = [Path(__file__), library, *[CODE / name for name in (
        "fused_camera_digest.c", "fused_camera_digest.py", "fused_camera_fingerprint.py", "profile_fused_camera_fingerprint.py",
        "fused_fingerprint_scene_node.py", "profile_scene_fingerprint_cost.py", "fingerprint_scene_node.py",
        "profile_scene_transform_cost.py", "scene_transform_cost_node.py", "mapped_image_linux_guard.py",
        "profile_mapped_scene.py", "streamed_ply_node_002.py", "run_bounded_journal_task.py", "run_with_resource_lease.py", "equal_fee_cost_bound.py")]]
    paths += [Path(row["path"].replace("/mnt/c/", "C:/")) for row in read(OUT / "p0/RESULT.json")["sources"]]
    identities = [data_identity(path) for path in paths]
    if state_path.exists():
        state = read(state_path)
        if state["status"] != "first_complete_pair_positive_remaining_checks_pending0" or state["sources"] != identities:
            raise ValueError("Only an unchanged registered positive first full pair can continue")
    else:
        output.mkdir(exist_ok=False)
        snapshot = output / "source_snapshot"
        snapshot.mkdir()
        for index, path in enumerate(paths):
            shutil.copyfile(path, snapshot / (f"{index:03d}_" + path.name))
        state = {"status": "running_registered_complete_first_pair0", "sources": identities,
                 "orders": [["original", "fused"], ["fused", "original"], ["original", "fused"]], "pairs": [],
                 "initial_known_family_fees": {"actual_complete_p0_cost_diagnostic": read(OUT / "p0_lease.json")["child_elapsed_seconds"],
                    "actual_g0_build_generic_and_component_included_once": read(OUT / "g0_lease.json")["child_elapsed_seconds"],
                    "actual_failed_Windows_entry_before_any_complete_node": read(OUT / "c0_first_lease.json")["child_elapsed_seconds"],
                    "actual_preflight_spawn_to_exit_seconds": 5.0979168,
                    "registered_source_and_cost_closing_conservative_reserve_seconds": 10.},
                 "new_optimization_count": 0}
        atomic_json(state_path, state)
    for index in range(len(state["pairs"]), args.through):
        pair = output / ("t"+str(index))
        pair.mkdir(exist_ok=False)
        rows = {}
        for mode in state["orders"][index]:
            leaf, node = pair / mode, pair / (mode+"_node")
            mounted = "/mnt/c/" + str(leaf.relative_to(Path("C:/"))).replace("\\", "/")
            command = [sys.executable, "-X", "utf8", str(CODE / "fused_fingerprint_scene_node.py"), "--kind", "camera_oracle", "--mode", "diagnostic",
                "--node", str(node), "--output", str(leaf), "--", "wsl", "-d", "E26-C2-LIV-20260907-134802", "--", "/usr/bin/env",
                "OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1", "/opt/e26_thin_surface_v1/venv/bin/python", "-B",
                "/mnt/c/lidar-camera-fusion-mask-robust-downsampling/02_code/gaussian_pipeline_optimization_v3_20261005/profile_fused_camera_fingerprint.py",
                "--mode", "baseline", "--fused-enabled", "1" if mode == "fused" else "0", "--library",
                "/mnt/c/lidar-camera-fusion-mask-robust-downsampling/04_outputs/gaussian_pipeline_optimization_v3_20261005/scf001/g0/build/fused_camera_digest.so",
                "--output", mounted]
            begin = time.perf_counter()
            with (pair / (mode+".log")).open("x", encoding="utf-8") as stream:
                child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
                owner = {"pid": child.pid, "creation_time": psutil.Process(child.pid).create_time()}
                returncode = child.wait()
            elapsed = time.perf_counter()-begin
            if returncode:
                state.update(status="failed_actual_complete_node0", failed_pair=index, failed_mode=mode,
                             actual_failed_seconds=elapsed, actual_failed_owner=owner)
                atomic_json(state_path, state)
                raise RuntimeError("The registered full consumer failed; no replacement run")
            result, controller = read(leaf / "RESULT.json"), read(node / "NODE.json")
            if controller["status"] != "completed" or controller["live_Windows_after_wait"] or any(row["alive"] for row in controller["actual_Linux_after_wait"]):
                raise AssertionError("A registered actual full worker did not close")
            rows[mode] = {"actual_spawn_to_exit_seconds": elapsed, "complete_stage_seconds": elapsed, "whole_seconds": elapsed,
                          "owner": owner, "controller": controller, "Scene_load_seconds": result["Scene_load_seconds"],
                          "all_pixel_and_state_check_seconds": result["all_pixel_and_state_check_seconds"]}
            atomic_json(pair / (mode+"_CLOSED.json"), rows[mode])
        begin = time.perf_counter()
        comparison = compare(pair / "original", pair / "fused")
        compare_seconds = time.perf_counter()-begin
        for row in rows.values():
            row["complete_stage_seconds"] += compare_seconds
            row["whole_seconds"] += compare_seconds
        state["pairs"].append({"pair": index, "order": state["orders"][index], "rows": rows, "comparison": comparison,
                               "actual_compare_seconds_charged_each_side": compare_seconds})
        atomic_json(state_path, state)
    medians = {mode: statistics.median(pair["rows"][mode]["complete_stage_seconds"] for pair in state["pairs"]) for mode in ("original", "fused")}
    fee = sum(state["initial_known_family_fees"].values())
    bound = equal_fee_cost_bound(medians["original"], medians["fused"], fee)
    ratios = [pair["rows"]["fused"]["whole_seconds"] / pair["rows"]["original"]["whole_seconds"] for pair in state["pairs"]]
    status = "first_complete_pair_positive_remaining_checks_pending0" if len(state["pairs"]) == 1 else "three_complete_pairs_done_validation_and_final_cost_pending0"
    if bound["conservative_sample_reduction_upper_bound"] < .03 or max(ratios) > 1.1:
        status = "closed_negative_complete_cost_or_whole_screen0"
    state.update(status=status, raw_complete_stage_medians=medians, raw_stage_reduction=1-medians["fused"]/medians["original"],
        whole_ratios=ratios, initial_known_family_fee_seconds=fee, sampled_known_fee_cost_bound=bound,
        source_identities_unchanged=identities == [data_identity(path) for path in paths],
        unknown_text_and_parent_tail_not_assumed_zero=True)
    if not state["source_identities_unchanged"]:
        raise AssertionError("An actual complete campaign source changed during execution")
    atomic_json(state_path, state)
    atomic_json(output / "RESULT.json", state)
    print({"status": status, "complete_stage_medians": medians, "known_fee_bound": bound, "pairs": len(state["pairs"]), "new": 0})


if __name__ == "__main__":
    main()
