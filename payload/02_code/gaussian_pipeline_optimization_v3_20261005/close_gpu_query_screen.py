"""Close the sequential all-query diagnostic using actual source and owner checks."""

import argparse
import json
from pathlib import Path
import time

import psutil

from gpu_query_node_config import ROOT, CODE, OUT, read, atomic_json, data_identity
from gpu_query_node import remaining, linux_identity_check


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.resolve().relative_to(OUT.resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    names = (("c0", "nc0", "SCREEN_CPU_LEASE_001.json"),
        ("g0", "ng0", "SCREEN_GPU_LEASE_001.json"),
        ("l0", "nl0", "LOCALIZATION_GPU_LEASE_001.json"),
        ("l1", "nl1", "LOCALIZATION_CPU_LEASE_001.json"))
    runs, win, linux, identities = {}, {}, {}, {}
    for folder, node_name, lease_name in names:
        result, node, lease = read(OUT / folder / "RESULT.json"), read(OUT / node_name / "NODE.json"), read(OUT / lease_name)
        if not all(result["checks"].values()) or node["status"] != "completed" or not lease.get("lease_released_utc"):
            raise AssertionError("Every actual all-query node and primary lease must close")
        for row in node["guard"].get("owned_process_identities", []) + [node["owner"]] + result.get("owners", []):
            win[(row["pid"], row["creation_time"])] = row
        for row in node["Linux_resource_receipts"]:
            linux[(row["pid"], row["creation_time"])] = {k: row[k] for k in ("pid", "creation_time")}
        for row in result["sources"] + result["inputs"] + [node["source"]]:
            previous = identities.setdefault(row["path"], row)
            if previous != row:
                raise ValueError("A shared frozen source or input changed between the actual sides")
        runs[folder] = {"phase_seconds": result.get("complete_query_stage_seconds", result.get("complete_phase_seconds")),
            "node_seconds": node["complete_node_seconds"], "complete_lease_child_seconds": lease["child_elapsed_seconds"],
            "resource_peaks": node["resource_peaks"], "result": result}
    for folder in ("l0", "l1"):
        trace = runs[folder]["result"]["trace"]["broker"]
        for row in trace.get("saved_workers", []) + trace.get("actual_CLI_owners", []) + ([trace["profile_owner"]] if "profile_owner" in trace else []):
            win[(row["pid"], row["creation_time"])] = row
    before = read(runs["l1"]["result"]["localization"])
    after = read(runs["l0"]["result"]["localization"])
    live_win = remaining(list(win.values()))
    live_linux = [linux_identity_check(row) for row in linux.values()]
    current = [data_identity(Path(row["path"])) for row in identities.values()]
    source_guard = list(identities.values()) == current
    pixel = {Path(r["path"]).name: r["gray_sha256"] for r in runs["c0"]["result"]["CPU_worker_trace"]}
    whole_ratio = runs["l0"]["node_seconds"] / runs["l1"]["node_seconds"]
    checks = {"all_four_original_side_guards_passed": True,
        "actual_saved_Windows_instances_gone": not live_win,
        "actual_saved_Linux_instances_gone": not any(r["alive"] for r in live_linux),
        "all_current_sources_and_inputs_unchanged": source_guard,
        "all24_original_gray_pixels_preserved": all(pixel[r["original"]] == r["gray_sha256"] for r in runs["g0"]["result"]["gray_rows"]),
        "same_all24_names_and_order": [r["image_name"] for r in before["rows"]] == [r["image_name"] for r in after["rows"]],
        "same_frozen_camera_and_ROIs": before["PINHOLE_camera"] == after["PINHOLE_camera"]
            and [r["target_window_PINHOLE"] for r in before["rows"]] == [r["target_window_PINHOLE"] for r in after["rows"]],
        "all24_loc_eval_3D_disjoint": all(r["loc_eval_3D_overlap"] == 0 for h in (before, after) for r in h["rows"]),
        "same_complete_localization_inputs": runs["l0"]["result"]["inputs"] == runs["l1"]["result"]["inputs"],
        "required_cost_screen_is_negative": whole_ratio > 1.1 and runs["l0"]["phase_seconds"] > runs["l1"]["phase_seconds"]}
    if not all(checks.values()):
        raise AssertionError("A current source, closure or sequential diagnostic finding changed")
    import numpy as np
    changes = [float(np.linalg.norm(np.asarray(a["camera_center_m"]) - np.asarray(b["camera_center_m"])))
        for a, b in zip(before["rows"], after["rows"])]
    receipt = {"status": "closed_negative_sequential_GPU_full_entry_diagnostic_cost",
        "review_kind": "root_self_review_no_subagent", "independent_review": False,
        "new_optimization_count": 0, "checks": checks,
        "complete_sides": {k: {a: b for a, b in r.items() if a != "result"} for k, r in runs.items()},
        "single_diagnostic_not_three_formal_cost_pairs": True,
        "feature_component_reduction": 1 - runs["g0"]["phase_seconds"] / runs["c0"]["phase_seconds"],
        "full_query_phase_reduction": 1 - runs["l0"]["phase_seconds"] / runs["l1"]["phase_seconds"],
        "full_entry_whole_ratio": whole_ratio,
        "preserved_first_four_complete_lease_child_seconds": sum(r["complete_lease_child_seconds"] for r in runs.values()),
        "all_Windows_instances_checked": len(win), "all_Linux_instances_checked": len(linux),
        "current_source_input_identities_checked": len(identities),
        "actual_live_Windows": live_win, "actual_Linux": live_linux,
        "center_changes_m_are_consistency_proxy_not_GT": {"max": max(changes), "mean": float(np.mean(changes))},
        "quality_three_seeds_formal_cost_generic_cancel_external_preview": "not_run_after_sequential_cost_screen_failed",
        "GPU_backend_unavailable_claim": False, "GPU_mathematical_equivalence_claim": False,
        "production_or_training_changed": False,
        "next_route": "explore_major_costs_do_not_count_a_GPU_API_or_feature_component_win",
        "review_inside_seconds": time.perf_counter() - started,
        "outer_review_lease_and_own_process_close_remain_separate": True}
    atomic_json(args.output / "RESULT.json", receipt)
    print(json.dumps({"status": receipt["status"], "checks": checks, "full_entry_whole_ratio": whole_ratio}), flush=True)


if __name__ == "__main__":
    main()
