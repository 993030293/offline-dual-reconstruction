"""Verify actual host file dispatch with the original complete Gaussian save proof."""

import time

STARTED = time.perf_counter()

import argparse
import gc
import inspect
import json
from pathlib import Path
import shutil
import sys

import psutil

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"
V2 = ROOT / "02_code/gaussian_pipeline_optimization_v2_20261005"
sys.path[:0] = [str(V1), str(V2)]
from runtime import activate_reference, atomic_json, data_identity
from mapped_image_linux_guard import LinuxRSSGuard
import profile_streamed_gaussian_ply_002 as saved_checker
from streamed_gaussian_ply_oracle_002 import NAMES, streamed_ply_model
from native_host_gaussian_files import host_ply_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("generic", "cpu", "host"), required=True)
    for name in ("runtime", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("model", "native-proof"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args()
    args.output.resolve().relative_to((ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/hc001").resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    if shutil.disk_usage(args.output).free < 20 * 1024**3:
        raise RuntimeError("The original disk reserve is unavailable")
    process = psutil.Process()
    atomic_json(args.output / "PROCESS_IDENTITY.json", {
        "scope": "WSL", "pid": process.pid, "creation_time": process.create_time(), "counted": False})
    with LinuxRSSGuard(args.output, atomic_json) as resource_guard:
        reference = activate_reference(args.runtime)
        import torch
        import plyfile
        from scene.gaussian_model import GaussianModel
        from vectorized_gaussian_read import vectorized_gaussian_read
        torch.set_num_threads(4)
        sources = [Path(__file__), Path(__file__).with_name("native_host_gaussian_files.py"),
            Path(__file__).with_name("native_host_file_worker.py"),
            Path(__file__).with_name("profile_streamed_gaussian_ply_002.py"),
            Path(__file__).with_name("streamed_gaussian_ply_oracle_002.py"),
            Path(__file__).with_name("mapped_image_linux_guard.py"),
            V1 / "runtime.py", V1 / "ply_serialization.py", V2 / "vectorized_gaussian_read.py",
            Path(sys.modules[GaussianModel.__module__].__file__), Path(plyfile.__file__)]
        inputs = [path for path in (args.model, args.native_proof) if path is not None]
        identities = [data_identity(path) for path in sources + inputs]
        copies = args.output / "source_copies"
        copies.mkdir()
        for index, path in enumerate(sources):
            shutil.copyfile(path, copies / (f"{index:02d}_" + path.name))
        atomic_json(args.output / "CONTRACT.json", {"sources_inputs": identities, "reference": reference,
            "mode": args.mode, "all_original_complete_byte_and_state_guards_kept": True, "counted": False})
        events = []
        if args.mode == "generic":
            source = inspect.getsource(saved_checker.generic)
            namespace = dict(saved_checker.__dict__)
            namespace["streamed_ply_model"] = lambda base: host_ply_model(base, events)
            exec(compile(source, "<original_generic_with_owned_host_factory>", "exec"), namespace)
            namespace["generic"](args.output, GaussianModel, torch)
            generic = json.loads((args.output / "RESULT.json").read_text(encoding="utf-8"))
            checks = {
                "original_generic_48_all_bytes_and_error_rows": len(generic["rows"]) == 48 and all(generic["checks"].values()),
                "twelve_real_host_helpers_successful": len(events) == 12 and all(row.get("status") == "completed" for row in events),
                "all_source_inputs_unchanged": identities == [data_identity(path) for path in sources + inputs]}
            result = {"status": "host_generic_passed" if all(checks.values()) else "host_generic_failed",
                "checks": checks, "original_generic_result": generic, "host_events": events,
                "complete_child_seconds": time.perf_counter() - STARTED, "counted": False}
            atomic_json(args.output / "HOST_GENERIC_RESULT.json", result)
        else:
            if args.model is None or args.native_proof is None:
                raise ValueError("The real complete model and independent unmodified native proof are required")
            cls = streamed_ply_model(GaussianModel) if args.mode == "cpu" else host_ply_model(GaussianModel, events)
            model = cls(3)
            with vectorized_gaussian_read(True):
                model.load_ply(str(args.model))
            torch.cuda.synchronize()
            free, _ = torch.cuda.mem_get_info()
            budget = min(12 * 1024**3, free - 2 * 1024**3)
            if budget <= 0:
                raise RuntimeError("The original CUDA reserve is unavailable")
            torch.cuda.reset_peak_memory_stats()
            before = time.perf_counter()
            if identities != [data_identity(path) for path in sources + inputs]:
                raise RuntimeError("A complete source or input changed before saving")
            snapshot, previous_rng = saved_checker.state(model, torch), saved_checker.rng(torch)
            point = args.output / "point_cloud.ply"
            model.save_ply(str(point))
            checks = {
                "real_current_model_526951_vertices": len(model._xyz) == 526951,
                "all_parameter_bits_grad_layout_versions": saved_checker.state_exact(model, snapshot, torch),
                "all_RNG_unchanged": saved_checker.rng_exact(previous_rng, saved_checker.rng(torch), torch),
                "current_point_and_reference_all_bytes": saved_checker.file_exact(point, args.output / "native_oracle.ply"),
                "unmodified_original_native_proof_all_bytes": saved_checker.file_exact(point, args.native_proof),
                "complete_source_input_unchanged": identities == [data_identity(path) for path in sources + inputs]}
            roundtrip = GaussianModel(3)
            with vectorized_gaussian_read(True):
                roundtrip.load_ply(str(point))
            checks["unmodified_loader_all_parameter_bits"] = all(
                torch.equal(getattr(model, name).view(torch.int32), getattr(roundtrip, name).view(torch.int32)) for name in NAMES)
            checks["CUDA_original_budget"] = torch.cuda.max_memory_reserved() <= budget
            checks["actual_host_worker_selected_and_exited"] = args.mode == "cpu" or (
                len(events) == 1 and events[0].get("status") == "completed" and events[0]["interop_process_returncode"] == 0)
            original_events = list(model.ply_serialization_events)
            payload = data_identity(point)
            del model, snapshot, roundtrip
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - before
            resource_guard.sample()
            checks["WSL_original_RSS_budget"] = resource_guard.row["peak_tree_RSS_bytes"] <= resource_guard.maximum_bytes
            result = {"status": "complete_host_save_verified" if all(checks.values()) else "required_host_save_guard_failed",
                "mode": args.mode, "checks": checks, "complete_phase_seconds": elapsed,
                "complete_child_seconds": time.perf_counter() - STARTED,
                "original_serialization_events": original_events, "host_events": events,
                "WSL_peak_RSS_bytes": resource_guard.row["peak_tree_RSS_bytes"],
                "CUDA_peak_reserved_bytes": torch.cuda.max_memory_reserved(), "point_payload": payload, "counted": False}
            atomic_json(args.output / "RESULT.json", result)
        print(json.dumps({key: result[key] for key in ("status", "checks", "complete_child_seconds")}), flush=True)
        if not all(result["checks"].values()):
            raise AssertionError("An original complete host-file save guard failed")


if __name__ == "__main__":
    main()
