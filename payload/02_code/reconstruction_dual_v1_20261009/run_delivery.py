"""Run an isolated raw recording through shared geometry and both delivery branches."""

import argparse
from pathlib import Path
import sys
import shutil
import re
import time
import traceback

import psutil

from common import HERE, OLD, ROOT, SFM_PYTHON, atomic_json, read_json, execute, linux_command, linux_path


def finish_review(root, state, ledger):
    if ledger['status'] != 'dual_outputs_complete_with_recorded_exceptions_and_intervention':
        raise ValueError('The exception and actual-output review is incomplete')
    state.update(status='dual_outputs_completed_with_recorded_exceptions',
        exception_review=str(root / 'EXCEPTION_LEDGER.json'),
        independent_geometry_accuracy_verified=False,
        unassisted_cold_start_verified=ledger['unassisted_single_uninterrupted_raw_to_delivery_claim_allowed'])
    atomic_json(root / 'STATE.json', state)
    atomic_json(root / 'RESULT.json', state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--case", default="box1")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--repair-note", default=None)
    parser.add_argument("--retry-stage", choices=("shared", "gaussian", "pgsr", "surface", "render_review", "delivery"))
    parser.add_argument("--stop-after", choices=("front", "complete"), default="complete")
    parser.add_argument("--gaussian-iterations", type=int, default=12000)
    parser.add_argument("--pgsr-iterations", type=int, default=15000)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,40}", args.case):
        raise ValueError("Case identifiers must be safe lowercase names")
    root = args.run_root.resolve()
    raw = args.raw.resolve(strict=True)
    config = {"schema": 1, "case": args.case, "raw": str(raw), "raw_bytes": raw.stat().st_size,
              "raw_mtime_ns": raw.stat().st_mtime_ns, "gaussian_iterations": args.gaussian_iterations,
              "pgsr_iterations": args.pgsr_iterations, "initial_solver": "COLMAP_native_global_CPU",
              "historical_derived_inputs_used": False, "heavy_GPU_jobs_serial": True,
              "surface_voxel_m": .004, "surface_truncation_m": .016, "surface_depth_max_m": 20.0,
              "surface_sample_count": 500000, "GT_used": False}
    if args.resume:
        if read_json(root / "CONFIG.json") != config:
            raise ValueError("Resume input or frozen parameters changed")
        state = read_json(root / "STATE.json")
        if state.get("status") == "failed_delivery" and not args.repair_note:
            raise ValueError("A failed run requires an explicit repair note; failure is never erased")
        if args.repair_note:
            atomic_json(root / ("MANUAL_INTERVENTION_" + str(time.time_ns()) + ".json"),
                {"status": "operator_or_assistant_intervention", "note": args.repair_note,
                 "previous_status": state["status"], "counts_as_algorithm_autonomous_recovery": False})
        if state.get("status") == "running":
            try:
                owner = psutil.Process(state["pid"])
                if owner.create_time() == state["process_birth"]:
                    raise RuntimeError("The registered delivery controller is still alive")
            except psutil.NoSuchProcess:
                pass
    else:
        root.mkdir(parents=True, exist_ok=False)
        atomic_json(root / "CONFIG.json", config)
        atomic_json(root / "CASES.json", {"cases": [{"case": args.case, "raw": str(raw),
                    "output": str(root / args.case)}]})
        state = {"stages": [], "first_started_unix": time.time()}
    state.update(status="running", pid=psutil.Process().pid, process_birth=psutil.Process().create_time())
    side = root / args.case / "optimized"
    if args.retry_stage:
        if not args.resume or not args.repair_note:
            raise ValueError("Retry requires resume and an explicit intervention note")
        failed = root / "stages" / args.retry_stage
        if not failed.is_dir() or (failed / "COMPLETED.json").exists():
            raise ValueError("Only an unfinished owned stage can be archived for a retry")
        process = read_json(failed / "PROCESS.json")
        for item in process.get("owned_process_identities", []):
            try:
                if psutil.Process(item["pid"]).create_time() == item["creation_time"]:
                    raise ValueError("A failed stage still has a live owned child")
            except psutil.NoSuchProcess:
                pass
        archive = root / "failures" / (args.retry_stage + "_" + str(time.time_ns()))
        archive.mkdir(parents=True)
        outputs = {"shared": [root / "shared"],
            "gaussian": [side / "training", side / "training_execution_sources", side / "training_execution_contract.json", root / "shared/gaussian_runtime"],
            "pgsr": [root / "pgsr"],
            "surface": [root / "pgsr/surface", root / "pgsr/SURFACE_FAILURE.json", root / "pgsr/SURFACE_RESULT.json"],
            "render_review": [root / "gaussian_diagnostics"],
            "delivery": [root / "delivery", root / "viewer", root / "native_packing", root / "browser_review", root / "viewer_staging.log", root / "DELIVERY_RESULT.json"]}
        if args.retry_stage == "delivery" and (root / "PREVIEW_SERVER.json").exists():
            server = read_json(root / "PREVIEW_SERVER.json")
            from start_viewer import stop_owned
            stop_owned(root, server)
            outputs["delivery"].append(root / "PREVIEW_SERVER.json")
        for path in [failed, *outputs[args.retry_stage]]:
            path.resolve().relative_to(root)
            if path.exists():
                target = archive / "stage" if path == failed else archive / "outputs" / path.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(path), str(target))
        atomic_json(archive / "RETRY.json", {"status": "failed_attempt_archived_for_explicit_manual_retry",
            "repair_note": args.repair_note, "algorithm_autonomous_recovery": False,
            "frozen_inputs_and_successful_upstream_stages_reused": True})

    def publish():
        state["total_wall_seconds_including_controller_pauses"] = time.time() - state["first_started_unix"]
        atomic_json(root / "STATE.json", state)

    def stage(label, command, budget, result, gpu=False):
        completed = root / "stages" / label / "COMPLETED.json"
        if completed.exists():
            receipt = read_json(completed)
            if receipt["status"] != "completed" or not Path(receipt["result"]).exists():
                raise ValueError("An incomplete stage cannot be reused")
            return read_json(receipt["result"])
        state["current_stage"] = label
        publish()
        value = execute(root, label, command, budget, result, gpu)
        state["stages"].append({"stage": label, "receipt": str(completed)})
        publish()
        return value

    try:
        publish()
        atomic_json(root / ("CONTROLLER_ATTEMPT_" + str(time.time_ns()) + ".json"),
            {"status": "controller_attempt_started", "pid": state["pid"], "process_birth": state["process_birth"],
             "resume": args.resume, "stop_after": args.stop_after, "repair_note": args.repair_note})
        atomic_json(root / "OFFLINE_CONTRACT.json", {"status": "runtime_offline_policy_active",
            "input": "local_recording_only", "LLM_required_by_runtime": False,
            "cloud_or_LLM_API_calls_authorized": False, "runtime_dependency_downloads_authorized": False,
            "Python_child_network_policy": "loopback_only_socket_guard",
            "native_COLMAP": "local_files_and_installed_executable",
            "viewer_assets": "installed_local_vendor_files", "viewer_binding": "127.0.0.1",
            "development_assistant_repairs_are_not_algorithm_recovery": True,
            "front_stage_started_before_socket_guard_added": not (root / "stages/front/SOURCE_MANIFEST.json").exists(),
            "scope": "workflow_processes_not_global_OS_firewall"})
        stage("preflight", [SFM_PYTHON, HERE / "preflight.py", "--run-root", root,
            "--output", root / "PREFLIGHT.json"], 120, root / "PREFLIGHT.json")
        stage("front", [SFM_PYTHON, HERE / "front_end.py", "--case", args.case,
            "--mode", "optimized", "--campaign", root, "--stop-after", "export",
            "--render-near-recovery"], 14400, side / "export/TRAINING_ADMISSION.json")
        if args.stop_after == "front":
            state["status"] = "front_completed_downstream_pending"
            publish()
            return
        stage("shared", [SFM_PYTHON, HERE / "prepare_shared.py", "--run-root", root], 900,
              root / "shared/RESULT.json")
        count = read_json(root / "shared/RESULT.json")["registered_frames"]
        command = linux_command(HERE / "gaussian_branch.py", "--mode", "optimized", "--runtime",
            linux_path(ROOT / "04_outputs/box2_3dgs_display_v1_20260930"), "--dataset",
            linux_path(root / "shared/gaussian"), "--output", linux_path(side / "training"),
            "--projection", "full", "--seed", "0", "--iterations", str(args.gaussian_iterations),
            "--maximum-seconds", "2400", "--input-version", root.name,
            "--expected-image-count", str(count), "--gsplat-build",
            linux_path(ROOT / "04_outputs/gaussian_pipeline_optimization_v1_20261005/p6_gsplat_build_001"),
            "--vectorized-ply", "--verify-native-ply")
        stage("gaussian", command, 3000, side / "training/STATE.json", gpu=True)
        stage("pgsr", linux_command(HERE / "pgsr_branch.py", "train", linux_path(root)),
              7500, root / "pgsr/TRAINING_RESULT.json", gpu=True)
        stage("surface", linux_command(HERE / "pgsr_branch.py", "surface", linux_path(root)),
              7500, root / "pgsr/SURFACE_RESULT.json", gpu=True)
        stage("render_review", linux_command(HERE / "render_gaussian.py", linux_path(root)),
              900, root / "gaussian_diagnostics/RESULT.json", gpu=True)
        stage("delivery", [SFM_PYTHON, HERE / "package_delivery.py", "--run-root", root],
              900, root / "DELIVERY_RESULT.json")
        if (raw.stat().st_size, raw.stat().st_mtime_ns) != (config["raw_bytes"], config["raw_mtime_ns"]):
            raise ValueError("Original raw identity changed")
        state["status"] = "dual_outputs_completed_exceptions_review_pending"
        if "traceback" in state:
            state["historical_controller_traceback"] = state.pop("traceback")
        publish()
        atomic_json(root / "RESULT.json", state)
        from audit_run import review
        finish_review(root, state, review(root))
    except BaseException:
        state.update(status="failed_delivery", traceback=traceback.format_exc())
        publish()
        atomic_json(root / ("FAILURE_" + str(time.time_ns()) + ".json"), state)
        from audit_run import review
        review(root)
        raise


if __name__ == "__main__":
    from run_target_delivery import main as target_main
    target_main()
