"""Close the first registered full Adam pair without promoting a negative screen."""

import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).parent
V1 = ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"
OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/fused_adam_current_001"
sys.path.insert(0, str(V1))
from runtime import atomic_json, data_identity
from current_fused_adam_node import linux_identity_check


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def windows_path(value):
    value = value.replace("\\", "/")
    if value.startswith("/mnt/c/"):
        return Path("C:/" + value[7:])
    if value.startswith("/opt/"):
        raise ValueError("An installed Linux identity requires the recorded Linux source probe")
    return Path(value)


def digest(path):
    return data_identity(path)["sha256"]


def main():
    started = time.perf_counter()
    process = psutil.Process()
    destination = OUT / "FIRST_FULL_SCREEN_CLOSURE_001.json"
    if destination.exists():
        raise FileExistsError(destination)
    models = {label: read(OUT / "t1" / label / "STATE.json") for label in ("native", "fused")}
    manifests = {label: read(OUT / "t1" / label / "INPUT_MANIFEST.json") for label in models}
    cameras = {label: read(OUT / "t1" / label / "CAMERA_MANIFEST.json") for label in models}
    configurations = {label: read(OUT / "t1" / label / "CONFIG.json") for label in models}
    checks = {"both_actual_full12000_and1016": all(row["status"] == "completed" and row["iteration"] == 12000
        and row["image_count"] == 1016 and sum(row["sample_counts"].values()) == 12000 for row in models.values()),
        "full_inputs1020_and_actual_sequence_exact": manifests["native"] == manifests["fused"]
        and len(manifests["native"]["files"]) == 1020
        and models["native"]["sample_sequence_sha256"] == models["fused"]["sample_sequence_sha256"],
        "all1016_complete_camera_fields_exact": cameras["native"] == cameras["fused"]
        and len(cameras["native"]["cameras"]) == 1016,
        "full_current_input_payloads_unchanged": all(digest(windows_path(row["path"])) == row["sha256"]
            for row in manifests["native"]["files"])}
    params = [{key: value for key, value in configurations[label]["parameters"].items() if key != "model_path"}
              for label in models]
    native_mode, fused_mode = [dict(configurations[label]["intervention"]) for label in models]
    checks["only_registered_Adam_training_mode_changed"] = (native_mode.pop("dense_adam_backend") == "native"
        and fused_mode.pop("dense_adam_backend") == "fused" and native_mode == fused_mode and params[0] == params[1]
        and configurations["native"]["external_adapters"] == configurations["fused"]["external_adapters"]
        and configurations["native"]["ply_serialization"] == configurations["fused"]["ply_serialization"])
    events = []
    source_checks = []
    for label in models:
        base = OUT / "t1" / label
        camera_result = read(base / "CAMERA_EXECUTION_RESULT.json")
        checks[label + "_actual_current_camera_and_prefetch_hooks"] = (camera_result["status"] == "actual_training_completed_acceptance_pending"
            and len(camera_result["receipts"]) == 1
            and camera_result["receipts"][0]["prefetch"]["prepared"] == 1016
            and camera_result["receipts"][0]["prefetch"]["consumed"] == 1016
            and camera_result["receipts"][0]["camera_execution"]["native_camera_calls"] == 1016)
        saved = read(base / "PLY_SERIALIZATION.json")
        checks[label + "_all3_actual_complete_PLY_oracles"] = (len(saved["events"]) == 3
            and saved["schedule_validation"]["observed_process_iterations"] == [2000, 7000, 12000])
        for event in saved["events"]:
            oracle = event["native_byte_oracle"]
            actual = windows_path(oracle["candidate"]["path"])
            original = windows_path(oracle["reference"]["path"])
            if not oracle["identical"] or actual.read_bytes() != original.read_bytes():
                raise AssertionError("An actual complete saved PLY differs from its independent native oracle")
            events.append({"mode": label, "path": str(actual), "vertices": event["vertices"], "full_byte_equal": True})
        for row in configurations[label]["implementation"] + configurations[label]["external_adapter_sources"]:
            path = windows_path(row["path"])
            source_checks.append({"path": str(path), "unchanged": digest(path) == row["sha256"]})
    for name in ("train_current_fused_adam.py", "check_current_fused_adam.py", "current_fused_adam_node.py", "evaluate_current_fused_adam.py"):
        source_checks.append({"path": str(CODE / name),
                              "unchanged": (CODE / name).read_bytes() == (OUT / "frozen_sources_002" / name).read_bytes()})
    checks["all_executed_current_source_snapshots_unchanged"] = all(row["unchanged"] for row in source_checks)
    quality = read(OUT / "t1/evaluation/EVALUATION.json")
    delta_rows = []
    for a, b in zip(quality["results"]["native"]["rows"], quality["results"]["fused"]["rows"]):
        if a["image_name"] != b["image_name"]:
            raise AssertionError("A full24 evaluation row changed its name or order")
        for region in ("full_image", "target"):
            dp, ds = b[region]["PSNR_dB"] - a[region]["PSNR_dB"], b[region]["SSIM"] - a[region]["SSIM"]
            delta_rows.append({"image_name": a["image_name"], "region": region, "PSNR_delta_dB": dp,
                "SSIM_delta": ds, "registered_protection_passed": dp >= -.1 and ds >= -.002})
    checks["actual_all24_full_and_target_denominator_preserved"] = (len(delta_rows) == 48
        and quality["all_frames_included_in_denominator"] == 24
        and all(quality["results"][label]["summary"][region]["count"] == 24 for label in models for region in ("full_image", "target")))
    nodes, leases = [], []
    windows = {}
    for name in ("n0g", "n0t_native", "n1g", "n1t_native", "n1t_fused", "n1q"):
        node = read(OUT / name / "NODE.json")
        expected = "failed" if name == "n0t_native" else "completed"
        if node["status"] != expected or node["live_Windows_after_wait"] or any(row["alive"] for row in node["actual_Linux_after_wait"]):
            raise AssertionError("An original bounded node did not close with its recorded expected status")
        nodes.append({"name": name, "status": node["status"], "seconds": node["complete_node_seconds"]})
        rows = node["guard"].get("owned_process_identities", []) + [node["owner"]]
        for row in rows:
            windows[(row["pid"], row["creation_time"])] = row
    for name in ("GENERIC_LEASE_001.json", "NATIVE_LEASE_001.json", "GENERIC_LEASE_002.json", "NATIVE_LEASE_002.json", "FUSED_LEASE_001.json", "QUALITY_LEASE_001.json"):
        value = read(OUT / name)
        if not value.get("lease_released_utc"):
            raise AssertionError("A measured lease is not recorded released")
        leases.append({"name": name, "seconds": value["child_elapsed_seconds"], "returncode": value["returncode"]})
        for pid_key, born_key in (("guard_pid", "guard_creation_time"), ("child_pid", "child_creation_time")):
            row = {"pid": value[pid_key], "creation_time": value[born_key]}
            windows[(row["pid"], row["creation_time"])] = row
    for row in windows.values():
        try:
            if psutil.Process(row["pid"]).create_time() == row["creation_time"]:
                raise AssertionError("An owned saved Windows instance is still active")
        except psutil.NoSuchProcess:
            pass
    linux = []
    for rel in ("g0/PROCESS_IDENTITY.json", "g1/PROCESS_IDENTITY.json", "t0/native_PROCESS_IDENTITY.json",
                "t1/native/PROCESS_IDENTITY.json", "t1/fused/PROCESS_IDENTITY.json", "t1/evaluation/PROCESS_IDENTITY.json"):
        value = linux_identity_check(read(OUT / rel))
        if value["alive"]:
            raise AssertionError("An owned saved Linux instance is still active")
        linux.append({"receipt": rel, "actual_owned_instance_exited": True})
    checks["all6_nodes_and_leases_expected_closed_actual_owned_instances_gone"] = True
    generic = read(OUT / "g1/RESULT.json")
    checks["current_live_binding_regression_and_all_original_generic_passed"] = all(generic["checks"].values())
    historic = read(OUT / "INTAKE_LOWER_BOUND.json")
    extra_fees = sum(row["seconds"] for row in leases if row["name"] not in ("NATIVE_LEASE_002.json", "FUSED_LEASE_001.json"))
    lower = historic["known_seconds_lower_bound"] + extra_fees
    phase = {label: value["training_call_seconds"] for label, value in models.items()}
    whole = {"native": next(row["seconds"] for row in nodes if row["name"] == "n1t_native"),
             "fused": next(row["seconds"] for row in nodes if row["name"] == "n1t_fused")}
    phase_upper = (phase["native"] - phase["fused"]) / (phase["native"] + lower / 6)
    failed = [row for row in delta_rows if not row["registered_protection_passed"]]
    checks["negative_cost_bound_below5_and_original_view_protections_failed"] = phase_upper < .05 and bool(failed)
    result = {"status": "first_current_full_pair_closed_negative_required_cost_and_registered_per_view_quality",
        "checks": checks, "phase_seconds": phase, "raw_phase_reduction": 1 - phase["fused"] / phase["native"],
        "whole_complete_node_seconds": whole, "raw_whole_reduction": 1 - whole["fused"] / whole["native"],
        "registered_view_region_rows": delta_rows, "failed_view_regions": len(failed),
        "failed_unique_images": len({row["image_name"] for row in failed}),
        "minimum_PSNR_delta_dB": min(row["PSNR_delta_dB"] for row in delta_rows),
        "minimum_SSIM_delta": min(row["SSIM_delta"] for row in delta_rows),
        "original_evaluator_mean_guard_passed": quality["development_metric_guard_passed"],
        "mean_deltas": quality["candidate_minus_control"], "sources": source_checks,
        "complete_PLY_byte_comparisons": events, "current_1020_inputs_and_1016_cameras_exact": True,
        "historical_cost_lower_bound_seconds": historic["known_seconds_lower_bound"],
        "new_nonformal_lease_costs_seconds": extra_fees, "required_shared_fee_lower_bound_seconds": lower,
        "six_side_share_lower_bound_seconds": lower / 6, "complete_phase_reduction_upper_bound": phase_upper,
        "cost_scope": "known_nonoverlapping_recorded_lower_bound_only_not_final_three_pair_fees",
        "actual_Windows_owned_instances_gone": len(windows), "actual_Linux_owned_instances_gone": linux,
        "nodes": nodes, "leases": leases, "review_kind": "root_separate_self_review_no_subagent",
        "not_run": ["seed1", "seed2", "formal_three_pair_speed_acceptance", "actual_complete_fused_checkpoint_recovery",
                    "protected_six_view_visual_review", "external_or_unseen_object_full_pipeline", "GT", "production", "preview"],
        "new_optimization_count": 0, "parent_count": 22,
        "review_process": {"pid": process.pid, "creation_time": process.create_time()},
        "review_seconds": time.perf_counter() - started,
        "observed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    if not all(checks.values()):
        atomic_json(OUT / "FIRST_FULL_SCREEN_REVIEW_FAILURE_001.json", result)
        raise AssertionError("The registered negative screen closure contains a failed protection")
    atomic_json(destination, result)
    print(json.dumps({key: result[key] for key in ("status", "phase_seconds", "raw_phase_reduction",
        "failed_view_regions", "minimum_PSNR_delta_dB", "complete_phase_reduction_upper_bound", "parent_count")}), flush=True)


if __name__ == "__main__":
    main()
