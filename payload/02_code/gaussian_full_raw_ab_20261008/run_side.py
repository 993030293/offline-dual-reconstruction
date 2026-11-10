"""Execute one isolated recording side with durable stage receipts and no old assets."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parent
V1 = ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"
sys.path[:0] = [str(V1), str(ROOT / "02_code/box2_initial_pose_quality_v2_20261002")]
from runtime import data_identity
from atomic_json_windows import atomic_json, read_json
from pose_gap_candidates import select_pose_gap_candidates


def read(path):
    return read_json(path)


def windows_to_wsl(path):
    value = Path(path).resolve().as_posix()
    if len(value) < 3 or value[1:3] != ":/":
        raise ValueError("A runtime input must be an absolute Windows drive path")
    return "/mnt/" + value[0].lower() + value[2:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--mode", choices=("baseline", "optimized"), required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--wsl-distribution", default="E26-C2-LIV-20260907-134802")
    parser.add_argument("--stop-after", choices=("sfm", "export", "training"), default="training")
    parser.add_argument("--disable-sim3-recovery", action="store_true")
    parser.add_argument("--render-near-recovery", action="store_true")
    args = parser.parse_args()
    cases = read(args.campaign / "CASES.json")
    entries = cases["cases"] if isinstance(cases, dict) else cases
    case = next(item for item in entries if item["case"] == args.case)
    raw = Path(case["raw"])
    destination = args.campaign / args.case / args.mode
    destination.mkdir(parents=True, exist_ok=True)
    attempt = destination / ("controller_" + str(time.time_ns()))
    attempt.mkdir()
    state_path = destination / "PIPELINE_STATE.json"
    previous = read(state_path) if state_path.exists() else None
    if previous and previous["status"] == "running":
        import psutil
        try:
            process = psutil.Process(previous["pid"])
            if process.create_time() == previous["process_birth"]:
                raise RuntimeError("The existing owned side controller is still alive")
        except psutil.NoSuchProcess:
            pass
    import psutil
    state = {"status": "running", "case": args.case, "mode": args.mode, "raw": str(raw),
             "pid": os.getpid(), "process_birth": psutil.Process().create_time(),
             "started_UTC": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "controller_attempt": str(attempt), "source": data_identity(Path(__file__)),
             "stages": [], "historical_derived_inputs_used": False}
    state["interrupted_attempts"] = [read(path) for path in sorted(destination.glob("INTERRUPTED_ATTEMPT_*.json"))]
    state["failed_stage_attempts"] = [read(path) for path in sorted(destination.glob("FAILED_STAGE_ATTEMPT_*.json"))]
    atomic_json(attempt / "CONTRACT.json", {**state, "case_receipt": case,
        "same_campaign_completed_stage_reuse": "Only validated finished stages from this new raw campaign",
        "stop_after": args.stop_after, "source": data_identity(Path(__file__)),
        "metric_failure_recovery_policy": "disabled" if args.disable_sim3_recovery else "pure_similarity_of_valid_pre_BA_source_map_only",
        "original_metric_BA_failures_remain_failed": True})
    started = time.monotonic()
    environment = {**os.environ, "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1"}

    def publish():
        state["controller_elapsed_seconds"] = time.monotonic() - started
        atomic_json(state_path, state)

    def stage(label, command, budget, result_path, rss=32, allowed_failure=None):
        receipt = destination / (label + "_process.json")
        log = destination / (label + ".log")
        result_path = Path(result_path)
        if receipt.exists():
            existing = read(receipt)
            if existing["status"] == "running":
                tracked = {item["pid"]: item["creation_time"] for item in existing["owned_process_identities"]}
                child_pid = existing["child_pid"]
                try:
                    child = psutil.Process(child_pid)
                    alive = child_pid in tracked and child.create_time() == tracked[child_pid]
                except psutil.NoSuchProcess:
                    alive = False
                if not alive or not any(str(destination).replace("\\", "/") in str(item).replace("\\", "/") for item in existing["command"]):
                    raise RuntimeError("A pending stage has no verified live owned process: " + label)
                state["current_stage"] = "waiting_for_owned_" + label
                publish()
                while existing["status"] == "running":
                    time.sleep(2)
                    existing = read(receipt)
            permitted_failure = (allowed_failure is not None and result_path.exists()
                and read(result_path).get("status") == allowed_failure and existing["status"] == "failed_child")
            if not permitted_failure and (existing["status"] != "completed" or existing.get("returncode") != 0 or not result_path.exists()):
                raise RuntimeError("An unfinished or failed stage cannot be silently reused: " + label)
            state["stages"].append({"stage": label, "status": "failed_original_stage_preserved" if permitted_failure else "completed_this_campaign_before_controller",
                "process_receipt": str(receipt), "result": str(result_path),
                "elapsed_seconds": existing["elapsed_seconds"], "peak_tree_RSS_bytes": existing["peak_tree_RSS_bytes"]})
            publish()
            return read(result_path)
        if result_path.exists() or log.exists():
            raise FileExistsError("Unreceipted stage assets require inspection: " + label)
        state["current_stage"] = label
        publish()
        guarded = [sys.executable, str(CODE / "run_owned_task.py"), "--receipt", str(receipt),
                   "--log", str(log), "--maximum-seconds", str(budget), "--maximum-rss-gib", str(rss),
                   "--", *map(str, command)]
        print(json.dumps({"case": args.case, "mode": args.mode, "stage": label, "status": "starting"}), flush=True)
        completed = subprocess.run(guarded, env=environment)
        measured = read(receipt)
        state["stages"].append({"stage": label, "status": measured["status"],
            "process_receipt": str(receipt), "result": str(result_path),
            "elapsed_seconds": measured["elapsed_seconds"], "peak_tree_RSS_bytes": measured["peak_tree_RSS_bytes"]})
        publish()
        permitted_failure = (allowed_failure is not None and result_path.exists()
            and read(result_path).get("status") == allowed_failure and measured["status"] == "failed_child")
        if not permitted_failure and (completed.returncode or measured.get("returncode") != 0 or not result_path.exists()):
            raise RuntimeError("Native stage failed; source and partial outputs retained: " + label)
        return read(result_path)

    def native(kind, *parameters):
        return [sys.executable, str(CODE / "run_native_stage.py"), "--kind", kind,
                "--mode", args.mode, "--raw", str(raw), *map(str, parameters)]

    try:
        coarse = None
        if args.case == "c0_july":
            stage("coarse", [sys.executable, CODE / "run_coarse_c0.py", "--raw", raw,
                            "--output", destination / "coarse"], 3600,
                  destination / "coarse/COARSE_RECEIPT.json")
            coarse = destination / "coarse/coarse_body_odometry.npz"
        command = [sys.executable, CODE / "run_initial.py", "--raw", raw, "--case", args.case,
                   "--mode", args.mode, "--output", destination / "initial"]
        if coarse:
            command.extend(["--coarse-trajectory", coarse])
        bindings_path = args.campaign / "CASE_CALIBRATION_BINDINGS.json"
        if bindings_path.exists():
            binding = read(bindings_path).get(args.case)
            if binding:
                command.extend(["--body-camera-calibration", binding["path"]])
        stage("initial", command, 1200, destination / "initial/SELECTION_RECEIPT.json")
        sfm = stage("sfm", [sys.executable, CODE / "run_sfm.py", "--initial", destination / "initial",
                    "--output", destination / "sfm", "--mode", args.mode], 11400,
                    destination / "sfm/RESULT.json")
        if args.stop_after == "sfm":
            state["status"] = "requested_SfM_boundary_completed_later_stages_pending"
            publish()
            return
        if sfm["all_initial_registered"]:
            current = destination / "recovery_not_triggered"
            current.mkdir(exist_ok=True)
            best = max(sfm["models"], key=lambda row: row["registered_count"])
            import shutil
            shutil.copyfile(Path(sfm["database"]), current / "database.db")
            recovered = {"status": "initial_all_registered_recovery_not_triggered", "image_root": sfm["image_root"],
                "models": [{**best, "remaining_failures": best["failed_images"]}],
                "original_registration_guard_passed": True}
            atomic_json(current / "RESULT.json", recovered)
        else:
            current = destination / "recovery"
            recovered = stage("recovery", native("recovery", "--source-run",
                  destination / "sfm", "--input-images", destination / "initial/images", "--output", current),
                  3650, current / "RESULT.json")
        best = max(recovered["models"], key=lambda row: row["registered_count"])
        original_remaining = best["remaining_failures"]
        state["initial_registration_after_recovery"] = {"initial_count": sfm["initial_count"],
            "registered": best["registered_count"], "remaining_failures": original_remaining,
            "all_initial_registered": not original_remaining}
        # The accepted native augmentation has a hard input-independent 64-image bound.
        if original_remaining:
            rows = [json.loads(line) for line in (destination / "initial/frame_decisions.jsonl").read_text().splitlines()]
            names = {row["image_name"] for row in read(destination / "initial/INITIAL_FRAME_MANIFEST.json")["selected_frames"]}
            lookup = {row["image_name"]: row for row in rows}
            candidates = [row for row in rows if row["decision"] in ("eligible_not_selected", "selected_initial_sfm_pending")
                          and row["image_name"] not in names]
            additions = {}
            for name in sorted(original_remaining):
                for low, high in [(-.8, -.4), (-.4, -.15), (-.15, 0), (0, .15), (.15, .4), (.4, .8)]:
                    supported = [row for row in candidates if low * 1e9 <= row["camera_header_ns"] - lookup[name]["camera_header_ns"] < high * 1e9]
                    if supported:
                        chosen = min(supported, key=lambda row: (-row["quality"]["quality_score"], row["camera_frame"]))
                        additions[chosen["image_name"]] = chosen
            atomic_json(destination / "AUGMENTATION_DECISION.json", {"native_candidate_count": len(additions),
                "maximum_accepted_native_additions": 64, "remaining_original_failures": original_remaining,
                "source": data_identity(V1 / "run_temporal_augmentation.py"), "no_quality_gate_relaxed": True})
            if 0 < len(additions) <= 64:
                latest = destination / "temporal"
                recovered = stage("temporal", native("temporal", "--initial", destination / "initial",
                      "--recovery", current, "--output", latest, "--purpose", "registration", "--threads", 8),
                      6650, latest / "RESULT.json")
                current = latest
                best = max(recovered["models"], key=lambda row: row["registered_count"])
            else:
                state["augmentation"] = "not_triggered_empty_candidates" if not additions else "failed_native_candidate_budget_guard"
                state["augmentation_native_candidate_count"] = len(additions)
        manifest = read(destination / "initial/INITIAL_FRAME_MANIFEST.json")
        rows = [json.loads(line) for line in (destination / "initial/frame_decisions.jsonl").read_text().splitlines()]
        gaps, decisions = select_pose_gap_candidates(rows, manifest["policy"], manifest["receipt"]["sharpness_min"])
        gap_path = destination / "POSE_GAP_DECISIONS.json"
        value = {"rows": decisions, "unchanged_quality_policy": manifest["policy"],
                 "sharpness_min": manifest["receipt"]["sharpness_min"], "position_priors_added": 0, "extrapolation_used": False}
        if gap_path.exists() and read(gap_path) != value:
            raise ValueError("Raw pose-gap decisions changed during campaign continuation")
        atomic_json(gap_path, value)
        if gaps:
            latest = destination / "pose_gap"
            recovered = stage("pose_gap", native("temporal", "--initial", destination / "initial",
                "--recovery", current, "--output", latest, "--purpose", "pose_gap", "--threads", 8),
                6650, latest / "RESULT.json")
            current = latest
            best = max(recovered["models"], key=lambda row: row["registered_count"])
            gap_path = latest / "POSE_GAP_DECISIONS.json"
        model = Path(best["path"])
        metric = stage("metric", native("metric", "--model", model, "--initial-frames", destination / "initial",
             "--output", destination / "metric", "--include-raw-candidates", "--pose-gap-decisions", gap_path,
             "--threads", 8), 600, destination / "metric/RESULT.json",
             allowed_failure=None if args.disable_sim3_recovery else "failed_projection_validation")
        metric_root = destination / "metric"
        pure_sim3 = metric["status"] == "failed_projection_validation"
        if pure_sim3:
            state["original_metric_BA_status"] = "failed_projection_validation"
            previous_fallback_receipt = destination / "metric_sim3_recovery_process.json"
            if previous_fallback_receipt.exists():
                failed = read(previous_fallback_receipt)
                if failed["status"] == "failed_child":
                    state["stages"].append({"stage": "original_strict_sim3_precision_attempt", "status": "failed_original_stage_preserved",
                        "process_receipt": str(previous_fallback_receipt), "result": None,
                        "elapsed_seconds": failed["elapsed_seconds"], "peak_tree_RSS_bytes": failed["peak_tree_RSS_bytes"]})
            if args.render_near_recovery:
                previous_track_receipt = destination / "metric_sim3_track_recovery_process.json"
                if previous_track_receipt.exists() and read(previous_track_receipt)["status"] == "failed_child":
                    failed = read(previous_track_receipt)
                    state["stages"].append({"stage": "original_track_sim3_precision_attempt", "status": "failed_original_stage_preserved",
                        "process_receipt": str(previous_track_receipt), "result": None,
                        "elapsed_seconds": failed["elapsed_seconds"], "peak_tree_RSS_bytes": failed["peak_tree_RSS_bytes"]})
                label = "metric_sim3_renderer_recovery"
            else:
                label = "metric_sim3_track_recovery"
            metric_root = destination / label
            recovery_command = [sys.executable, CODE / "metric_sim3_recovery.py", "--failed-metric",
                destination / "metric", "--output", metric_root, "--prefilter-original-track-policy"]
            if args.render_near_recovery:
                recovery_command.append("--prefilter-renderer-near-plane")
            metric = stage(label, recovery_command, 120, metric_root / "RESULT.json")
        selected = Path(metric["results"][metric["selected"]]["model"])
        stage("roles", native("roles", "--model", selected, "--initial-frames", destination / "initial",
             "--pose-gap-decisions", gap_path, "--phase", "after_BA", "--output", destination / "roles"),
             180, destination / "roles/TRAINING_ROLES.json")
        image_root = Path(recovered.get("image_root", str(destination / "initial/images")))
        export_command = native("export", "--ba", metric_root, "--roles",
              destination / "roles/TRAINING_ROLES.json", "--images", image_root, "--output", destination / "export")
        if pure_sim3:
            export_command.append("--allow-pure-sim3-recovery")
        admission = stage("export", export_command, 950, destination / "export/TRAINING_ADMISSION.json")
        state["metric_selected"] = metric["selected"]
        state["admitted_training_count"] = admission["training_count"]
        if args.stop_after == "export":
            state["status"] = "requested_export_boundary_completed_training_and_evaluation_pending"
            publish()
            return
        arguments = ["--mode", args.mode, "--runtime", windows_to_wsl(ROOT / "04_outputs/box2_3dgs_display_v1_20260930"),
             "--dataset", windows_to_wsl(destination / "export/dataset"), "--output", windows_to_wsl(destination / "training"),
             "--projection", "full", "--seed", "0", "--iterations", "10000" if args.mode == "baseline" else "12000",
             "--maximum-seconds", "1800", "--input-version", args.case + "_fresh_raw_20261008",
             "--expected-image-count", str(admission["training_count"])]
        if args.mode == "optimized":
            arguments.extend(["--gsplat-build", windows_to_wsl(ROOT / "04_outputs/gaussian_pipeline_optimization_v1_20261005/p6_gsplat_build_001"),
                              "--vectorized-ply", "--verify-native-ply"])
        command = ["wsl.exe", "-d", args.wsl_distribution, "--", "env", "PYTHONUTF8=1", "PYTHONDONTWRITEBYTECODE=1",
                   "/opt/e26_thin_surface_v1/venv/bin/python", windows_to_wsl(CODE / "train_variant.py"), *arguments]
        command = [sys.executable, str(CODE / "gpu_lease.py"), "--receipt",
                   str(attempt / "training_gpu_lease.json"), "--label", args.case + "_" + args.mode + "_training",
                   "--", *command]
        trained = stage("training", command, 4350, destination / "training/STATE.json")
        if trained["status"] != "completed":
            raise ValueError("Training returned without native full iteration completion")
        state["status"] = "fresh_training_completed_evaluation_and_module_audit_pending"
        state["total_measured_stage_seconds"] = sum(row["elapsed_seconds"] for row in state["stages"])
        state["interrupted_attempt_seconds_lower_bound"] = sum(row["elapsed_seconds_lower_bound"] for row in state["interrupted_attempts"])
        state["archived_failed_stage_seconds"] = sum(row["elapsed_seconds"] for row in state["failed_stage_attempts"])
        state["total_operational_seconds_lower_bound"] = state["total_measured_stage_seconds"] + state["interrupted_attempt_seconds_lower_bound"] + state["archived_failed_stage_seconds"]
        state["cost_boundary"] = "Sum of owned stage durations, excludes controller construction and operator idle"
        publish()
    except BaseException as error:
        state.update(status="failed_pipeline_stage", error=repr(error))
        publish()
        atomic_json(attempt / "FAILURE.json", state)
        raise


if __name__ == "__main__":
    main()
