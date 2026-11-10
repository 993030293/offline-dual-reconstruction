"""Time a fresh raw-to-preview run and preserve full evaluation and sensor checks."""

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback
import urllib.request

import psutil

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parent
OLD = ROOT / "02_code/gaussian_full_raw_ab_20261008"
sys.path.insert(0, str(OLD))
from atomic_json_windows import atomic_json, read_json
from run_side import windows_to_wsl
from runtime import data_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--route", choices=("reference", "sparse"), required=True)
    parser.add_argument("--case", default="box1", choices=("box1", "box2"))
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--node", type=Path, required=True)
    parser.add_argument("--playwright-package", type=Path, required=True)
    args = parser.parse_args()
    args.campaign = args.campaign.resolve()
    args.campaign.mkdir(parents=True, exist_ok=False)
    cases = read_json(ROOT / "04_outputs/gaussian_full_raw_ab_20261008/CASES.json")["cases"]
    case = next(row for row in cases if row["case"] == args.case)
    raw = Path(case["raw"])
    before_raw = (raw.stat().st_size, raw.stat().st_mtime_ns)
    atomic_json(args.campaign / "CASES.json", {"cases": [{**case, "output": str(args.campaign / args.case)}]})
    identities = [data_identity(path) for path in [Path(__file__), CODE / "run_pipeline_side.py",
        CODE / "augmentation_graph.py", CODE / "run_native_sparse.py", CODE / "pack_preview.cjs",
        OLD / "run_side.py", OLD / "run_initial.py", OLD / "run_sfm.py", OLD / "train_variant.py"]]
    atomic_json(args.campaign / "CONTRACT.json", {"route": args.route, "case": args.case,
        "sources": identities, "raw": str(raw), "raw_bytes": before_raw[0], "raw_mtime_ns": before_raw[1],
        "fresh_private_inputs": True, "historical_derived_inputs_used": False,
        "both_routes_use_current_optimized_initial_and_12000_iteration_gsplat": True,
        "common_declared_renderer_recovery": True, "all_failed_metric_attempts_counted": True,
        "GPU_and_CPU_formal_sides_run_serially": True,
        "OS_file_cache_uncontrolled_no_manual_flush_or_warmup": True,
        "primary_seconds": "owned uninterrupted raw initial start through complete native packing and full viewer readiness",
        "quality_sensor_audit_seconds_also_recorded_and_in_full_total": True})
    side = args.campaign / args.case / "optimized"
    started = time.monotonic()
    state = {"status": "running", "route": args.route, "case": args.case,
             "pid": os.getpid(), "process_birth": psutil.Process().create_time(), "stages": []}
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    server = None

    def publish():
        state["elapsed_seconds"] = time.monotonic()-started
        atomic_json(args.campaign / "STATE.json", state)

    def execute(label, command, budget, result=None):
        state["current_stage"] = label
        publish()
        receipt = args.campaign / (label+"_process.json")
        guarded = [sys.executable, str(OLD / "run_owned_task.py"), "--receipt", str(receipt),
                   "--log", str(args.campaign / (label+".log")), "--maximum-seconds", str(budget),
                   "--maximum-rss-gib", "32", "--", *map(str, command)]
        print(json.dumps({"stage": label, "route": args.route, "status": "starting"}), flush=True)
        began = time.monotonic()
        done = subprocess.run(guarded, env=env)
        measured = read_json(receipt)
        state["stages"].append({"stage": label, "controller_call_seconds": time.monotonic()-began,
                                "receipt": str(receipt), "status": measured["status"],
                                "elapsed_seconds": measured["elapsed_seconds"], "returncode": done.returncode})
        publish()
        if done.returncode or measured["status"] != "completed":
            raise RuntimeError("A fully receipted stage failed: "+label)
        return read_json(result) if result else None

    try:
        execute("pipeline", [sys.executable, CODE / "run_pipeline_side.py", "--route", args.route,
                "--case", args.case, "--mode", "optimized", "--campaign", args.campaign,
                "--render-near-recovery"], 14000, side / "training/STATE.json")
        state["raw_to_model_seconds"] = time.monotonic()-started
        execute("preview_staging", [sys.executable, OLD / "build_raw_viewer.py", "--campaign", args.campaign,
                "--case", args.case, "--mode", "optimized"], 300)
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", args.port))
        server_stream = (args.campaign / "preview_server.log").open("x", encoding="utf-8")
        server = subprocess.Popen([sys.executable, str(OLD / "serve_raw_viewer.py"), "--root",
            str(args.campaign / "viewer"), "--port", str(args.port)], stdout=server_stream,
            stderr=subprocess.STDOUT, env=env, creationflags=subprocess.CREATE_NO_WINDOW)
        state["preview_server_identity"] = {"pid": server.pid, "birth": psutil.Process(server.pid).create_time()}
        publish()
        url = f"http://127.0.0.1:{args.port}/{args.case}-optimized/cases/packed/export.html"
        deadline = time.monotonic()+20
        while True:
            if server.poll() is not None:
                raise RuntimeError("The owned preview server exited before publication")
            try:
                with urllib.request.urlopen(url, timeout=1) as reply:
                    if reply.status == 200:
                        break
            except (OSError, TimeoutError):
                if time.monotonic() >= deadline:
                    raise
                time.sleep(.1)
        packing = args.campaign / "native_packing"
        packing.mkdir()
        pack_command = [sys.executable, OLD / "gpu_lease.py", "--receipt", args.campaign / "packing_GPU_LEASE.json",
            "--label", args.route+"_native_packing", "--", args.node, CODE / "pack_preview.cjs",
            args.playwright_package, url, packing]
        execute("native_packing", pack_command, 2730,
                packing / "RESULT.json")
        state["raw_to_preview_seconds"] = time.monotonic()-started
        state["primary_pipeline_finished"] = True
        publish()
        execute("localization", [sys.executable, OLD / "localize_evaluation.py", "--side", side,
                "--mode", "optimized", "--output", side / "localization"], 1200,
                side / "localization/LOCALIZATION.json")
        render = ["wsl.exe", "-d", "E26-C2-LIV-20260907-134802", "--", "env", "PYTHONUTF8=1",
            "PYTHONDONTWRITEBYTECODE=1", "/opt/e26_thin_surface_v1/venv/bin/python",
            windows_to_wsl(OLD / "render_evaluation.py"), "--side", windows_to_wsl(side), "--runtime",
            windows_to_wsl(ROOT / "04_outputs/box2_3dgs_display_v1_20260930"), "--output", windows_to_wsl(side / "evaluation")]
        execute("evaluation", [sys.executable, OLD / "gpu_lease.py", "--receipt", args.campaign / "render_GPU_LEASE.json",
            "--label", args.route+"_full24_evaluation", "--", *render], 300,
            side / "evaluation/RESULT.json")
        execute("scale_consistency", [sys.executable, OLD / "audit_sensor_scale.py", "--side", side, "--raw", raw,
                "--output", side / "scale_consistency"], 300, side / "scale_consistency/RESULT.json")
        if before_raw != (raw.stat().st_size, raw.stat().st_mtime_ns):
            raise ValueError("The immutable original recording identity changed")
        state["raw_to_preview_plus_full_QA_seconds"] = time.monotonic()-started
        state["status"] = "complete_raw_model_preview_and_QA_comparison_pending"
        state["raw_identity_unchanged"] = True
        publish()
        atomic_json(args.campaign / "RESULT.json", state)
        print(json.dumps({k: state[k] for k in ["status", "raw_to_model_seconds", "raw_to_preview_seconds",
              "raw_to_preview_plus_full_QA_seconds"]}), flush=True)
    except BaseException:
        state.update(status="failed_full_run", traceback=traceback.format_exc())
        publish()
        atomic_json(args.campaign / "FAILURE.json", state)
        raise
    finally:
        if server is not None:
            if server.poll() is None:
                server.terminate()
                server.wait(timeout=15)
            server_stream.close()


if __name__ == "__main__":
    main()
