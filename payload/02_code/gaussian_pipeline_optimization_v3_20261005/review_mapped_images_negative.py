"""Close the negative result with current source, input, quality and owner evidence."""

import argparse
import json
from pathlib import Path
import subprocess
import time

import psutil

from mapped_image_node_config import ROOT, CODE, OUT, atomic_json, data_identity, read


def windows_path(value):
    value = str(value).replace("\\", "/")
    if value.startswith("/mnt/c/"):
        return Path("C:/"+value[len("/mnt/c/"):])
    return Path(value)


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.resolve().relative_to(OUT.resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    profiles = [OUT/"p1"/mode/"RESULT.json" for mode in ("native", "mapped")]
    profiles += [OUT/"c0"/f"pair_{index:03d}"/mode/"RESULT.json" for index in range(3) for mode in ("native", "mapped")]
    records = [read(path) for path in profiles]
    identities = {}

    def require(row):
        path = windows_path(row["path"])
        actual = data_identity(path)
        if actual["bytes"] != row["bytes"] or actual["sha256"] != row["sha256"]:
            raise AssertionError("An observed input or source changed: "+str(path))
        identities[str(path.resolve()).lower()] = actual

    for record in records:
        for row in record["sources"]+record["inputs"]:
            path = windows_path(row["path"])
            if str(path.resolve()).lower() not in identities:
                require(row)
            else:
                saved = identities[str(path.resolve()).lower()]
                if saved["bytes"] != row["bytes"] or saved["sha256"] != row["sha256"]:
                    raise AssertionError("Complete paired inputs have contradictory identities")
        for row in record["generated_assets"].values():
            require(row)
    for mode in ("native", "mapped"):
        result = read(OUT/"t0"/mode/"training/MAPPED_TRAINING_RESULT.json")
        for row in result["sources"]+read(OUT/"t0"/mode/"training/INPUT_MANIFEST.json")["files"]:
            require(row)
    for name in ("g2", "g3", "u0"):
        result = read(OUT/name/"RESULT.json")
        for row in result.get("sources", [result.get("source")]):
            if row is not None:
                require(row)
    source_copies = 0
    for path, record in zip(profiles, records):
        for index, row in enumerate(record["sources"]):
            copied = path.parent/"source_snapshot"/(f"{index:03d}_"+Path(row["path"]).name)
            actual = data_identity(copied)
            if actual["bytes"] != row["bytes"] or actual["sha256"] != row["sha256"]:
                raise AssertionError("A frozen complete Scene source copy changed")
            source_copies += 1
    historical = read(OUT/"source_history_001/RECEIPT.json")
    old = read(OUT/"g1/RESULT.json")
    old_source_preserved = data_identity(historical["copy"])["sha256"] == old["source"]["sha256"] == historical["sha256"]
    quality = read(OUT/"t0/quality/RESULT.json")
    state = {mode: read(OUT/"t0"/mode/"training/STATE.json") for mode in ("native", "mapped")}
    traces = {mode: read(OUT/"t0"/mode/"training/SAMPLING_TRACE.json") for mode in state}
    own_nodes = [OUT/"np0n", OUT/"np1n", OUT/"np1m", OUT/"nt0n", OUT/"nt0m", OUT/"nq0"]
    own_nodes += [OUT/"c0"/f"pair_{index:03d}"/(mode+"_node") for index in range(3) for mode in state]
    windows, linux, costs, node_results = {}, {}, [], []
    for folder in own_nodes:
        node = read(folder/"NODE.json")
        node_results.append(node)
        for row in node["guard"].get("owned_process_identities", [])+[node["owner"]]:
            windows[(row["pid"], row["creation_time"])] = row
        for row in node["Linux_resource_receipts"]:
            linux[(row["pid"], row["creation_time"])] = {"pid": row["pid"], "creation_time": row["creation_time"]}
    guards = list(OUT.glob("*GUARD*.json"))
    for path in guards:
        for row in read(path).get("owned_process_identities", []):
            windows[(row["pid"], row["creation_time"])] = row
    for path in OUT.rglob("WSL_RESOURCE.json"):
        row = read(path)
        linux[(row["pid"], row["creation_time"])] = {"pid": row["pid"], "creation_time": row["creation_time"]}
    prior_leases = [path for path in OUT.rglob("*lease*.json") if path.name != "NEGATIVE_REVIEW_LEASE_001.json"]
    prior_leases += [path for path in OUT.glob("*LEASE*.json") if path.name != "NEGATIVE_REVIEW_LEASE_001.json"]
    prior_leases = list({str(path.resolve()): path for path in prior_leases}.values())
    for path in prior_leases:
        row = read(path)
        if not row.get("lease_released_utc") or row.get("returncode") not in (0, 1):
            raise AssertionError("A prior measured lease did not close its actual child")
        for pid_key, time_key in (("guard_pid", "guard_creation_time"), ("child_pid", "child_creation_time")):
            windows[(row[pid_key], row[time_key])] = {"pid": row[pid_key], "creation_time": row[time_key]}
        if path.parent == OUT:
            costs.append({"path": str(path), "seconds": row["child_elapsed_seconds"], "scope": "complete_root_lease_child", "returncode": row["returncode"]})
    alive_windows = []
    for row in windows.values():
        try:
            process = psutil.Process(row["pid"])
            if process.create_time() == row["creation_time"]:
                alive_windows.append(row)
        except psutil.NoSuchProcess:
            pass
    code = "import json,psutil; rows=json.loads("+repr(json.dumps(list(linux.values())))+"); out=[]\n"
    code += "for row in rows:\n alive=False\n try:\n  p=psutil.Process(row['pid']); alive=p.create_time()==row['creation_time'] and p.status()!=psutil.STATUS_ZOMBIE\n except psutil.NoSuchProcess: pass\n out.append(dict(row,alive=alive))\nprint(json.dumps(out))"
    linux_actual = json.loads(subprocess.check_output(["wsl", "-d", "E26-C2-LIV-20260907-134802", "--",
        "/opt/e26_thin_surface_v1/venv/bin/python", "-B", "-c", code], text=True, timeout=30))
    full_cache = OUT/"full_cache_001"
    payload_count, payload_bytes = 0, 0
    for path in full_cache.glob("*.json"):
        info = read(path)
        payload = path.with_suffix(".f32")
        actual = data_identity(payload)
        if actual["bytes"] != info["payload_bytes"] or actual["sha256"] != info["payload_sha256"] or info["dtype"] != "<f4":
            raise AssertionError("An immutable complete image cache payload changed")
        require(info["source"])
        require(info["prepare_source"])
        payload_count += 1
        payload_bytes += actual["bytes"]
    failed = [row for row in quality["deltas"] if not row["passed"]]
    training_nodes = {mode: read(OUT/("nt0n" if mode == "native" else "nt0m")/"NODE.json") for mode in state}
    peaks = {mode: node["Linux_resource_receipts"][0]["peak_tree_RSS_bytes"] for mode, node in training_nodes.items()}
    checks = {
        "three_fresh_complete_Scene_pairs_exact_and_conservative_memory_passed": read(OUT/"c0/RESULT.json")["three_pair_peak_RSS_reduction"] >= .2 and all(all(row["checks"].values()) for row in records),
        "all_current_input_sources_and_saved_source_copies_verified": bool(identities) and source_copies == 8*17,
        "initial_numpy_mutation_failure_retained_with_original_source": old["status"] == "failed_verification" and not old["checks"]["numpy_private_write_survived_release"] and old_source_preserved,
        "same_numpy_failure_repaired_without_gate_relaxation": all(read(OUT/"g3/RESULT.json")["checks"].values()),
        "both_own_complete12000_full_state_sampling_denominators": all(result["status"] == "completed" and result["iteration"] == 12000 and result["image_count"] == 1016 for result in state.values()) and traces["native"] == traces["mapped"] and state["native"]["sample_counts"] == state["mapped"]["sample_counts"],
        "all1016_actual_CPU_CUDA_uploads_exact_after_release": all(read(OUT/"u0/RESULT.json")["checks"].values()),
        "required_full24_quality_negative_not_bypassed": quality["status"] == "failed_original_quality_protection" and len(quality["deltas"]) == 48 and len(failed) == 30 and not quality["checks"]["original_mean_quality_protection"],
        "seed1_seed2_recovery_formal_new_training_absent": not any((OUT/name).exists() for name in ("t1", "t2", "recovery", "training_campaign")),
        "actual_saved_Windows_and_Linux_owners_all_gone": not alive_windows and all(not row["alive"] for row in linux_actual),
        "all_prior_leases_released": len(prior_leases) == 17,
        "unchanged_complete1016_cache_within_original_budget": payload_count == 1016 and payload_bytes == 6137940480 and payload_bytes <= 8*1024**3,
        "no_new_accepted_item_or_goal_completion": read(ROOT/"research/2026-10-05-gaussian-optimization-round3-50/CURRENT_STATE.json")["additional_accepted_count"] == 17,
    }
    campaign = read(OUT/"c0/RESULT.json")
    costs.append({"path": str(OUT/"c0/RESULT.json"), "seconds": campaign["complete_campaign_seconds"],
        "scope": "six_scene_nodes_and_measured_campaign_window_no_duplicate_nested_leases"})
    result = {"status": "R131_closed_negative_required_quality_failure" if all(checks.values()) else "failed_negative_review",
        "review_kind": "root_self_review_no_subagent", "checks": checks, "new_optimization_count": 0,
        "unique_current_inputs_sources_verified": len(identities), "saved_complete_Scene_source_copies": source_copies,
        "actual_saved_Windows_instances_checked": len(windows), "actual_saved_Linux_instances_checked": len(linux),
        "live_Windows": alive_windows, "Linux_current_checks": linux_actual, "prior_released_leases": len(prior_leases),
        "seed0_complete_node_seconds": {mode: node["complete_node_seconds"] for mode, node in training_nodes.items()},
        "seed0_complete_node_ratio": training_nodes["mapped"]["complete_node_seconds"]/training_nodes["native"]["complete_node_seconds"],
        "seed0_actual_RSS_bytes": peaks, "seed0_actual_RSS_reduction": 1-peaks["mapped"]/peaks["native"],
        "quality_failed_rows": failed, "quality_minimum_PSNR_delta_dB": min(row["PSNR_delta_dB"] for row in quality["deltas"]),
        "quality_minimum_SSIM_delta": min(row["SSIM_delta"] for row in quality["deltas"]),
        "quality_root_cause": "unconfirmed_no_native_native_variance_or_complete_new_recovery_experiment",
        "recorded_costs_without_double_counting": costs, "recorded_children_and_campaign_seconds": sum(row["seconds"] for row in costs),
        "cost_not_claimed_as_three_pair_acceptance": True,
        "unmeasured_envelope": "c0 startup/first comparison before its timer and unwrapped observational/setup commands are not an exact full entrance cost; no accepted claim",
        "review_body_seconds_before_final_publication": time.perf_counter()-started,
        "further_full_training_GT_external_preview_or_visual_quality_review": "not_run_after_required_quality_failure"}
    atomic_json(args.output/"REVIEW.json", result)
    print(json.dumps({"status": result["status"], "checks": checks, "unique_current_inputs_sources_verified": len(identities), "new_optimization_count": 0}))
    if not all(checks.values()):
        raise AssertionError("The negative root review did not satisfy its exact evidence scope")


if __name__ == "__main__":
    main()
