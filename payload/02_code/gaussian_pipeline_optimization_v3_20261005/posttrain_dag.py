"""Schedule independent complete evaluation and checkpoint archival before publication."""

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"
V2 = ROOT/"02_code/gaussian_pipeline_optimization_v2_20261005"
sys.path[:0] = [str(V1), str(V2)]
from runtime import atomic_json, data_identity
from zstd_checkpoint import verified_expand


def read(path):
    return json.loads(Path(path).read_text())


def task_payload(output, task):
    if task == "evaluate":
        path = output/"pair/evaluation/EVALUATION.json"
        result = read(path)
        if not result["development_metric_guard_passed"] or result["all_frames_included_in_denominator"] != 24:
            raise ValueError("The complete frozen development evaluation guard failed")
        return [data_identity(output/"pair/evaluation"/name) for name in
            ("EVALUATION.json", "COMBINATION.json", "EVALUATION_CONTRACT.json", "SIX_VIEW_COMPARISON.jpg")]
    elif task == "archive":
        archive = output/"archive/checkpoint.archive"
        manifest = read(output/"archive/ARCHIVE.json")
        if manifest["native_receipt"]["status"] != "complete":
            raise ValueError("The complete native state receipt is missing")
        expanded = output/"archive/verified_native.pth"
        receipt = verified_expand(archive, expanded)
        if receipt["native_payload"] != manifest["native_payload"]:
            raise ValueError("The complete native archive payload changed")
        expanded.unlink()
        atomic_json(output/"archive/VERIFIED.json", receipt)
        return [data_identity(path) for path in (archive, archive.with_suffix(".archive.json"), output/"archive/ARCHIVE.json")]
    raise ValueError("Unsupported finalization DAG task")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["serial", "parallel"], required=True)
    for name in ("pair-source", "checkpoint", "runtime", "reference-run", "localization", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--interrupt-after-evaluation", action="store_true")
    args = parser.parse_args()
    if args.interrupt_after_evaluation and (args.mode != "serial" or args.resume):
        raise ValueError("The injected stop is an explicit one-time serial lifecycle test")
    started = time.perf_counter()
    source_paths = [Path(__file__), Path(__file__).with_name("evaluate_dag_models.py"), V2/"vectorized_gaussian_read.py",
        V2/"reference_ssim.py", V1/"evaluate_pair.py", Path(__file__).with_name("zstd_checkpoint.py"),
        Path(__file__).with_name("effective_route.py")]
    inputs = [args.checkpoint, args.checkpoint.with_suffix(".json"), args.localization]
    versions = set()
    for label in ("continuous", "resumed"):
        inputs += [args.pair_source/label/name for name in ("STATE.json", "INPUT_MANIFEST.json", "model/point_cloud/iteration_12000/point_cloud.ply")]
        if read(args.pair_source/label/"STATE.json")["status"] != "completed" or read(args.pair_source/label/"STATE.json")["iteration"] != 12000:
            raise ValueError("The finalization DAG requires actual completed12000-step models")
        versions.add(read(args.pair_source/label/"INPUT_MANIFEST.json")["input_version"])
    if len(versions) != 1:
        raise ValueError("The two completed finalization models belong to different input versions")
    input_version = versions.pop()
    if input_version == "C212_frozen":
        if args.localization.resolve() != (args.reference_run/"05_evaluation/C_augmented_212/H_LOCALIZATION.json").resolve():
            raise ValueError("C212 finalization requires its own frozen localization")
    elif read(args.localization).get("input_version") != input_version:
        raise ValueError("The finalization localization belongs to a different map version")
    sources, identity = [data_identity(p) for p in source_paths], [data_identity(p) for p in inputs]
    protocol = {"sources": sources, "input": identity, "mode": args.mode, "DAG": {"evaluate": [], "archive": [], "publish": ["evaluate", "archive"]},
        "workers": 1 if args.mode == "serial" else 2, "GPU_heavy_tasks": 1, "isolated_outputs": True,
        "fixed_models_iterations": 12000, "archive_iteration": 8000, "archive_is_last_retained_checkpoint_not_final12000_state": True,
        "same_actual_training_recovery_run": str(args.pair_source), "all24_original_development_views": True,
        "input_version": input_version,
        "complete_native_payload_and_original_codec": True, "counted": False}
    if args.resume:
        stored = read(args.output/"PROTOCOL.json")
        if stored != protocol:
            raise ValueError("A restart changed the finalization DAG sources, inputs or configuration")
        state = read(args.output/"STATE.json")
        if state["status"] != "interrupted_after_completed_evaluation":
            raise ValueError("Only the owned explicitly interrupted finalization DAG can resume")
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        atomic_json(args.output/"PROTOCOL.json", protocol)
        state = {"status": "running", "tasks": {}, "publication_created": False, "counted": False}
        for source, label in (("continuous", "baseline"), ("resumed", "combined")):
            for name in ("STATE.json", "INPUT_MANIFEST.json", "model/point_cloud/iteration_12000/point_cloud.ply"):
                origin, target = args.pair_source/source/name, args.output/"pair"/label/name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(origin, target)
                if data_identity(origin)["sha256"] != data_identity(target)["sha256"]:
                    raise ValueError("A final model or its original manifest changed during owned preparation")
        atomic_json(args.output/"STATE.json", state)
    commands = {
        "evaluate": [sys.executable, str(Path(__file__).with_name("evaluate_dag_models.py")), "--runtime", str(args.runtime),
            "--pair", str(args.output/"pair"), "--localization", str(args.localization), "--reference-run", str(args.reference_run),
            "--input-version", input_version, "--iterations", "12000"],
        "archive": [sys.executable, str(Path(__file__).with_name("effective_route.py")), "--stage", "archive",
            "--checkpoint", str(args.checkpoint), "--output", str(args.output/"archive")]
    }
    running, streams = {}, {}

    def launch(task):
        command = [sys.executable, str(V1/"run_bounded_task.py"), "--receipt", str(args.output/(task+"_process.json")),
            "--log", str(args.output/(task+".log")), "--maximum-seconds", "150", "--maximum-rss-gib", "16", "--", *commands[task]]
        streams[task] = (args.output/(task+"_controller.log")).open("x")
        process = subprocess.Popen(command, stdout=streams[task], stderr=subprocess.STDOUT, start_new_session=True)
        state["tasks"][task] = {"status": "running", "controller_pid": process.pid, "start_seconds": time.perf_counter()-started}
        running[task] = process
        atomic_json(args.output/"STATE.json", state)

    pending = []
    try:
        for task in commands:
            if task in state["tasks"] and state["tasks"][task]["status"] == "complete":
                old = state["tasks"][task]
                if any(data_identity(Path(row["path"])) != row for row in old["payload"]):
                    raise ValueError("A committed finalization task changed before resuming")
                old["reused_after_exact_revalidation"] = True
            else:
                pending.append(task)
        while pending or running:
            while pending and len(running) < (1 if args.mode == "serial" else 2):
                launch(pending.pop(0))
            for task, process in list(running.items()):
                code = process.poll()
                if code is None:
                    continue
                process.wait()
                streams.pop(task).close()
                del running[task]
                state["tasks"][task].update(returncode=code, child_process=read(args.output/(task+"_process.json")))
                if code:
                    raise RuntimeError("An owned finalization DAG child failed: "+task)
                payload = task_payload(args.output, task)
                state["tasks"][task].update(status="complete", payload=payload, end_seconds=time.perf_counter()-started)
                atomic_json(args.output/"STATE.json", state)
                if task == "evaluate" and args.interrupt_after_evaluation:
                    state.update(status="interrupted_after_completed_evaluation", publication_created=False)
                    atomic_json(args.output/"STATE.json", state)
                    return 86
            if running:
                time.sleep(.02)
        if sources != [data_identity(p) for p in source_paths] or identity != [data_identity(p) for p in inputs]:
            raise ValueError("A frozen DAG input or implementation changed during finalization")
        if set(state["tasks"]) != {"evaluate", "archive"} or any(row["status"] != "complete" for row in state["tasks"].values()):
            raise ValueError("Finalization publication cannot proceed with an incomplete dependency")
        state.update(status="complete_both_validated_before_publication", publication_created=True,
            whole_makespan_seconds=time.perf_counter()-started,
            conservative_sum_of_task_peak_RSS_bytes=sum(r["child_process"]["peak_tree_RSS_bytes"] for r in state["tasks"].values()),
            exact_original_task_outputs=True)
        atomic_json(args.output/"PUBLICATION.json", {"status": "both_complete_local_artifacts_available", "tasks": state["tasks"],
            "external_preview_service_or_production_modified": False, "counted": False})
        atomic_json(args.output/"RESULT.json", state)
    except BaseException as error:
        state.update(status="failed_no_publication", error=repr(error), publication_created=False)
        atomic_json(args.output/"STATE.json", state)
        raise
    finally:
        for task, process in running.items():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
        for stream in streams.values():
            stream.close()
    atomic_json(args.output/"STATE.json", state)
    print({"status": state["status"], "makespan_seconds": state["whole_makespan_seconds"]}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
