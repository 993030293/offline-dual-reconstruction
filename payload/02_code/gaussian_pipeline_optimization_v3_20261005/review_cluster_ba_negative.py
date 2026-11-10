"""Audit the saved cluster BA quality rejection and close actual owned resources."""

import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import psutil

from cluster_ba_node import linux_identity_check
from cluster_ba_node_config import CODE, OUT, V1, atomic_json, data_identity, read


def windows_path(value):
    text = str(value)
    return Path("C:/" + text[7:]) if text.startswith("/mnt/c/") else Path(text)


def collect_identities(value, target):
    if isinstance(value, dict):
        if {"path", "bytes", "sha256"}.issubset(value):
            path = windows_path(value["path"])
            if path.suffix.lower() in (".md", ".rst", ".txt", ".tex"):
                return
            target.setdefault(str(path.resolve()), []).append(value)
        for item in value.values():
            collect_identities(item, target)
    elif isinstance(value, list):
        for item in value:
            collect_identities(item, target)


def main():
    started = time.perf_counter()
    destination = OUT/"r0"
    destination.mkdir(parents=True, exist_ok=False)
    quality = read(OUT/"q0/RESULT.json")
    deltas = quality["per_view_deltas"]
    failed = [row for row in deltas if row["PSNR_delta_dB"] < -.1 or row["SSIM_delta"] < -.002]
    names = {row["image_name"] for row in deltas}
    checks = {
        "actual_required_quality_failure": quality["status"] == "required_own_full24_quality_failed",
        "complete_all24_both_regions": len(deltas) == 48 and len(names) == 24 and all(
            {row["region"] for row in deltas if row["image_name"] == name} == {"full_image", "target"} for name in names),
        "original_thresholds_and_failure_count": len(failed) == 16 and all(
            row["passed"] == (row["PSNR_delta_dB"] >= -.1 and row["SSIM_delta"] >= -.002) for row in deltas),
        "mean_not_substituted_for_each_view": quality["checks"]["original_mean_guard"]
            and not quality["checks"]["all_original_per_view_quality"],
        "quality_nonmetric_guards_preserved": all(quality["checks"][key] for key in (
            "all24_full_target_metrics", "WSL_CUDA_bounds", "original_time_bound", "source_unchanged")),
        "later_seeds_not_run": not (OUT/"t1").exists() and not (OUT/"t2").exists(),
        "formal_three_pairs_not_run": not (OUT/"campaign_001").exists(),
    }
    native = read(windows_path(quality["native_evaluation"]))
    native_rows = {side: {row["image_name"]: row for row in native["results"][side]["rows"]}
                   for side in ("auto", "cluster")}
    checks["native_all24_names_retained"] = all(set(rows) == names for rows in native_rows.values())
    checks["native_all48_original_deltas_recomputed"] = all(
        row["PSNR_delta_dB"] == native_rows["cluster"][row["image_name"]][row["region"]]["PSNR_dB"]
            - native_rows["auto"][row["image_name"]][row["region"]]["PSNR_dB"]
        and row["SSIM_delta"] == native_rows["cluster"][row["image_name"]][row["region"]]["SSIM"]
            - native_rows["auto"][row["image_name"]][row["region"]]["SSIM"] for row in deltas)
    configurations = []
    localizations = []
    data_ids = {}
    for side, admission, location in (("auto", "a0", "h0"), ("cluster", "a1", "h1")):
        training = read(OUT/"t0"/side/"RESULT.json")
        state = read(OUT/"t0"/side/"training/STATE.json")
        config = read(OUT/"t0"/side/"training/CONFIG.json")
        manifest = read(OUT/"t0"/side/"training/CAMERA_MANIFEST.json")
        configurations.append(config)
        checks[side+"_actual_full12000_seed0_1016"] = all(training["checks"].values()) and (
            state["status"] == "completed" and state["iteration"] == 12000 and state["seed"] == config["seed"] == 0
            and state["image_count"] == config["expected_image_count"] == len(manifest["cameras"]) == 1016
            and config["parameters"]["iterations"] == 12000)
        admitted = read(OUT/admission/"RESULT.json")
        localized = read(OUT/location/"RESULT.json")
        checks[side+"_own_complete_admission"] = all(admitted["checks"].values()) and (
            admitted["training_count"] == 1016 and admitted["training_points"] == 75767
            and admitted["training_observations"] == 959120)
        checks[side+"_own_all24_48_localization"] = all(localized["checks"].values()) and len(localized["worker"]["calls"]) == 48
        localizations.append(read(windows_path(localized["localization"])))
        collect_identities(config, data_ids)
    exclusions = {"source_path", "model_path"}
    checks["all_training_math_and_camera_flags_same"] = (
        {k: v for k, v in configurations[0]["parameters"].items() if k not in exclusions}
        == {k: v for k, v in configurations[1]["parameters"].items() if k not in exclusions}
        and configurations[0]["camera_optimization"] == configurations[1]["camera_optimization"])
    by_name = [{row["image_name"]: row for row in value["rows"]} for value in localizations]
    reference_checks = []
    for name, first in by_name[0].items():
        second = by_name[1][name]
        with Image.open(windows_path(first["image_reference"])) as a, Image.open(windows_path(second["image_reference"])) as b:
            reference_checks.append(np.array_equal(np.asarray(a), np.asarray(b)) and (
                first["raw_image"] == second["raw_image"] and first["target_window_PINHOLE"] == second["target_window_PINHOLE"]
                and first["status"] == second["status"] == "localized" and first["loc_eval_3D_overlap"] == second["loc_eval_3D_overlap"] == 0))
    checks["all24_original_reference_pixels_K_ROI_and_split"] = len(reference_checks) == 24 and all(reference_checks) and (
        localizations[0]["PINHOLE_camera"] == localizations[1]["PINHOLE_camera"])
    windows = {}
    linux = {}
    nodes = []
    for node_name in ("ng0", "nd0", "nd1", "na0", "na1", "nh0", "nh1", "nt0a", "nt0c", "nq0"):
        node = read(OUT/node_name/"NODE.json")
        expected_return = 1 if node_name == "nq0" else 0
        checks[node_name+"_expected_exit_and_owned_close"] = node["returncode"] == expected_return and (
            not node["live_Windows_after_wait"] and not any(row["alive"] for row in node["actual_Linux_after_wait"])
            and node["final_source_unchanged"] and not node.get("controller_error"))
        nodes.append({"node": node_name, "status": node["status"], "complete_node_seconds": node["complete_node_seconds"],
                      "resource_peaks": node["resource_peaks"]})
        for row in [node["owner"], *node["guard"].get("owned_process_identities", [])]:
            windows[(row["pid"], row["creation_time"])] = row
        for row in node["actual_Linux_after_wait"]:
            linux[(row["pid"], row["creation_time"])] = row
    fees = []
    for path in sorted(OUT.glob("*_LEASE_001.json")):
        value = read(path)
        if "child_elapsed_seconds" not in value:
            continue
        checks[path.stem+"_released"] = bool(value.get("lease_released_utc"))
        for role in ("guard", "child"):
            row = {"pid": value[role+"_pid"], "creation_time": value[role+"_creation_time"]}
            windows[(row["pid"], row["creation_time"])] = row
        fees.append({"receipt": str(path), "complete_child_seconds": value["child_elapsed_seconds"],
                     "returncode": value["returncode"], "lock_wait_seconds_separate": value["wait_seconds"]})
    checks["ten_actual_prior_node_fees_retained"] = len(fees) == 10
    alive = []
    for row in windows.values():
        try:
            process = psutil.Process(row["pid"])
            if process.create_time() == row["creation_time"] and process.is_running():
                alive.append(row)
        except psutil.NoSuchProcess:
            pass
    linux_checks = [linux_identity_check(row) for row in linux.values()]
    checks["all_actual_saved_Windows_instances_gone"] = not alive
    checks["all_actual_saved_Linux_instances_gone"] = not any(row["alive"] for row in linux_checks)
    for contract in sorted(OUT.rglob("CONTRACT.json")):
        collect_identities(read(contract), data_ids)
    differences = []
    for path, identities in data_ids.items():
        current = data_identity(Path(path))
        if any(current["bytes"] != value["bytes"] or current["sha256"] != value["sha256"] for value in identities):
            differences.append(path)
    checks["all_current_recorded_source_and_input_identities"] = not differences
    result = {
        "status": "negative_scope_review_passed" if all(checks.values()) else "negative_scope_review_failed",
        "review_kind": "root_self_review_no_subagent", "independent_review": False,
        "checks": checks, "failed_quality_rows": failed, "failed_count": len(failed),
        "minimum_PSNR_delta_dB": min(row["PSNR_delta_dB"] for row in deltas),
        "minimum_SSIM_delta": min(row["SSIM_delta"] for row in deltas),
        "current_source_input_identities_checked": len(data_ids), "identity_differences": differences,
        "saved_Windows_instances_checked": len(windows), "actual_alive_Windows": alive,
        "saved_Linux_instances_checked": len(linux), "actual_Linux": linux_checks,
        "prior_node_fee_receipts_once_each": fees, "prior_complete_child_seconds": sum(row["complete_child_seconds"] for row in fees),
        "nodes": nodes, "new_optimization_count": 0, "complete_semantic_acceptance": False,
        "quality_failure_root_cause": "unconfirmed_no_GT_or_extra_experiment",
        "later_quality_formal_cost_recovery_external_preview": "not_run_after_required_quality_failure",
        "inside_review_seconds": time.perf_counter()-started,
        "review_own_node_exit_and_lease_close_measured_separately": True,
    }
    atomic_json(destination/"RESULT.json", result)
    print(json.dumps({"status": result["status"], "checks": len(checks), "failed_checks": [k for k, v in checks.items() if not v],
                      "source_input_identities": len(data_ids), "Windows_instances": len(windows), "Linux_instances": len(linux),
                      "failed_quality_rows": len(failed), "new_optimization_count": 0}))
    if not all(checks.values()):
        raise AssertionError("Saved negative result or actual ownership review failed")


if __name__ == "__main__":
    main()

