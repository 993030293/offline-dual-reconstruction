"""Profile the unchanged admitted Gaussian save before selecting a new mechanism."""

import time

STARTED = time.perf_counter()

import argparse
import cProfile
import gc
import json
from pathlib import Path
import pstats
import shutil
import sys

import psutil

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"
V2 = ROOT / "02_code/gaussian_pipeline_optimization_v2_20261005"
sys.path[:0] = [str(V1), str(V2)]
from runtime import activate_reference, atomic_json, data_identity
from mapped_image_linux_guard import LinuxRSSGuard
from profile_streamed_gaussian_ply_002 import file_exact, rng, rng_exact, state, state_exact
from streamed_gaussian_ply_oracle_002 import NAMES, admission, streamed_ply_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runtime", "model", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.output.resolve().relative_to((ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/dp001").resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    if shutil.disk_usage(args.output).free < 20 * 1024**3:
        raise RuntimeError("The original free-disk reserve is unavailable")
    process = psutil.Process()
    atomic_json(args.output / "PROCESS_IDENTITY.json", {
        "scope": "WSL", "pid": process.pid, "creation_time": process.create_time(),
        "mode": "unchanged_R127_source_cost_diagnostic", "counted": False})
    with LinuxRSSGuard(args.output, atomic_json) as resource_guard:
        reference = activate_reference(args.runtime)
        import numpy as np
        import plyfile
        import torch
        from scene.gaussian_model import GaussianModel
        from vectorized_gaussian_read import vectorized_gaussian_read

        torch.set_num_threads(4)
        sources = [Path(__file__), Path(__file__).with_name("mapped_image_linux_guard.py"),
            Path(__file__).with_name("profile_streamed_gaussian_ply_002.py"),
            Path(__file__).with_name("streamed_gaussian_ply_oracle_002.py"),
            V1 / "runtime.py", V1 / "ply_serialization.py", V2 / "vectorized_gaussian_read.py",
            Path(sys.modules[GaussianModel.__module__].__file__), Path(plyfile.__file__)]
        identities = [data_identity(path) for path in sources + [args.model]]
        copies = args.output / "source_copies"
        copies.mkdir()
        for index, path in enumerate(sources):
            shutil.copyfile(path, copies / (f"{index:02d}_" + path.name))
        atomic_json(args.output / "CONTRACT.json", {
            "reference": reference, "sources_inputs": identities, "counted": False,
            "profile_scope": "unchanged_R127_save_with_cProfile_diagnostic_only",
            "native_reference_once_is_required_first_diagnostic_cost": True,
            "parameter_RNG_source_roundtrip_and_all_bytes_guards_retained": True})
        model = streamed_ply_model(GaussianModel)(3)
        with vectorized_gaussian_read(True):
            model.load_ply(str(args.model))
        torch.cuda.synchronize()
        free, _ = torch.cuda.mem_get_info()
        budget = min(12 * 1024**3, free - 2 * 1024**3)
        if budget <= 0:
            raise RuntimeError("The original CUDA reserve is unavailable")
        torch.cuda.reset_peak_memory_stats()
        phase_started = time.perf_counter()
        if identities != [data_identity(path) for path in sources + [args.model]]:
            raise RuntimeError("A source or complete input changed before the diagnostic")
        snapshot, previous_rng = state(model, torch), rng(torch)
        supported, reason = admission(model, GaussianModel)
        if not supported:
            raise RuntimeError("The real current model is outside the admitted save scope: " + reason)
        profiler = cProfile.Profile()
        point = args.output / "point_cloud.ply"
        save_started = time.perf_counter()
        profiler.runcall(model.save_ply, str(point))
        save_seconds = time.perf_counter() - save_started
        profiler.dump_stats(str(args.output / "save_profile.pstats"))
        stats = pstats.Stats(profiler)
        calls = [{"file": key[0], "line": key[1], "function": key[2],
            "primitive_calls": value[0], "calls": value[1], "self_seconds": value[2],
            "cumulative_seconds": value[3]} for key, value in stats.stats.items()]
        calls.sort(key=lambda row: row["cumulative_seconds"], reverse=True)
        checks = {
            "real_current_model_526951_vertices": len(model._xyz) == 526951,
            "all_parameter_bits_grad_layout_versions_readonly": state_exact(model, snapshot, torch),
            "all_RNG_unchanged": rng_exact(previous_rng, rng(torch), torch),
            "current_complete_byte_oracle_equal": file_exact(point, args.output / "native_oracle.ply"),
            "complete_source_input_unchanged": identities == [data_identity(path) for path in sources + [args.model]],
            "CUDA_original_budget": torch.cuda.max_memory_reserved() <= budget}
        roundtrip = GaussianModel(3)
        with vectorized_gaussian_read(True):
            roundtrip.load_ply(str(point))
        checks["original_loader_all_six_parameter_bits"] = all(
            torch.equal(getattr(model, name).view(torch.int32), getattr(roundtrip, name).view(torch.int32))
            for name in NAMES)
        events = list(model.ply_serialization_events)
        peak_before_native = resource_guard.row["peak_tree_RSS_bytes"]
        del roundtrip, snapshot
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        complete_diagnostic_phase_seconds = time.perf_counter() - phase_started
        native_started = time.perf_counter()
        native_point = args.output / "independent_unmodified_native.ply"
        GaussianModel.save_ply(model, str(native_point))
        torch.cuda.synchronize()
        checks["unmodified_original_writer_all_file_bytes"] = file_exact(point, native_point)
        checks["all_RNG_unchanged_after_native"] = rng_exact(previous_rng, rng(torch), torch)
        native_seconds = time.perf_counter() - native_started
        checks["complete_source_input_still_unchanged"] = identities == [data_identity(path) for path in sources + [args.model]]
        payload = data_identity(point)
        logical = len(model._xyz) * len(model.construct_list_of_attributes()) * 4
        del model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        resource_guard.sample()
        checks["original_WSL_RSS_budget"] = resource_guard.row["peak_tree_RSS_bytes"] <= resource_guard.maximum_bytes
        result = {"status": "current_complete_save_profile_verified" if all(checks.values()) else "required_diagnostic_guard_failed",
            "checks": checks, "save_profile_seconds": save_seconds,
            "complete_diagnostic_phase_seconds": complete_diagnostic_phase_seconds,
            "independent_native_diagnostic_seconds": native_seconds,
            "complete_child_seconds": time.perf_counter() - STARTED,
            "logical_full_matrix_bytes_not_memory_metric": logical,
            "observed_peak_before_original_tuple_oracle_bytes": peak_before_native,
            "WSL_peak_RSS_bytes": resource_guard.row["peak_tree_RSS_bytes"],
            "CUDA_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "events": events, "point_payload": payload, "profile_calls": calls,
            "profile_overhead_not_formal_cost_evidence": True, "new_optimization_count": 0}
        atomic_json(args.output / "RESULT.json", result)
        print(json.dumps({key: result[key] for key in ("status", "checks", "save_profile_seconds",
            "complete_diagnostic_phase_seconds", "independent_native_diagnostic_seconds", "WSL_peak_RSS_bytes")}), flush=True)
        if not all(checks.values()):
            raise AssertionError("A current complete save profile guard failed")


if __name__ == "__main__":
    main()
