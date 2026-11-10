"""Profile the current complete native evaluation with reversible operator labels."""

import argparse
import json
from pathlib import Path
import sys
import time

STARTED = time.perf_counter()
ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity
import evaluate_interior_ssim
import evaluate_view_major


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", type=Path, required=True)
    parser.add_argument("--localization", type=Path, required=True)
    parser.add_argument("--cuda-events", action="store_true")
    args = parser.parse_args()
    output = args.pair.parent
    original_activate = evaluate_interior_ssim.activate_reference
    original_argv = sys.argv
    installed = {}
    native_sources = []
    calls = {"render": 0, "get_features": 0, "get_scaling": 0, "get_rotation": 0, "get_opacity": 0}
    profiler = None
    device_events = []

    def activate(runtime):
        nonlocal profiler
        metadata = original_activate(runtime)
        import torch
        import gaussian_renderer
        import diff_gaussian_rasterization
        from scene.gaussian_model import GaussianModel
        activities = [torch.profiler.ProfilerActivity.CPU]
        if not args.cuda_events:
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        profiler = torch.profiler.profile(activities=activities,
                                          record_shapes=True, profile_memory=True)
        profiler.__enter__()

        def measured(name, function, *values, **options):
            if args.cuda_events:
                begin = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                begin.record()
            with torch.profiler.record_function("R139/" + name):
                result = function(*values, **options)
            if args.cuda_events:
                end.record()
                device_events.append((name, begin, end))
            return result

        installed.update(module=gaussian_renderer, model_class=GaussianModel, render=gaussian_renderer.render, properties={})
        for name in ("get_features", "get_scaling", "get_rotation", "get_opacity"):
            descriptor = getattr(GaussianModel, name)
            installed["properties"][name] = descriptor

            def getter(model, descriptor=descriptor, name=name):
                calls[name] += 1
                return measured(name, descriptor.fget, model)

            setattr(GaussianModel, name, property(getter, descriptor.fset, descriptor.fdel, descriptor.__doc__))

        def render(*values, **options):
            calls["render"] += 1
            return measured("native_complete_render", installed["render"], *values, **options)

        gaussian_renderer.render = render
        paths = [Path(__file__), Path(evaluate_view_major.__file__), Path(evaluate_interior_ssim.__file__),
                 Path(gaussian_renderer.__file__), Path(diff_gaussian_rasterization.__file__)]
        native_sources.extend(data_identity(path) for path in paths)
        return metadata

    failure = None
    evaluate_interior_ssim.activate_reference = activate
    try:
        sys.argv = [str(CODE / "evaluate_view_major.py"), "--mode", "view", "--pair", str(args.pair),
                    "--localization", str(args.localization)]
        evaluate_view_major.main()
    except BaseException as error:
        failure = repr(error)
        raise
    finally:
        evaluate_interior_ssim.activate_reference, sys.argv = original_activate, original_argv
        if installed:
            installed["module"].render = installed["render"]
            for name, descriptor in installed["properties"].items():
                setattr(installed["model_class"], name, descriptor)
        if profiler is not None:
            profiler.__exit__(None, None, None)
        if failure is not None:
            atomic_json(output / "PROFILE_FAILURE.json", {"status": "failed_profiling", "error": failure,
                        "calls": calls, "elapsed_seconds": time.perf_counter() - STARTED, "new_optimization_count": 0})
    import torch
    averages = []
    for event in profiler.key_averages():
        averages.append({"name": event.key, "count": event.count, "cpu_total_us": event.cpu_time_total,
                         "cpu_self_us": event.self_cpu_time_total, "device_total_us": event.device_time_total,
                         "device_self_us": event.self_device_time_total, "cpu_memory_usage": event.cpu_memory_usage,
                         "device_memory_usage": event.device_memory_usage})
    kernels = [{"name": event.name, "duration_us": event.device_time_total, "device_index": event.device_index}
               for event in profiler.events() if str(event.device_type).endswith("CUDA")]
    event_rows = []
    if args.cuda_events:
        torch.cuda.synchronize()
        event_rows = [{"name": name, "elapsed_ms": begin.elapsed_time(end)} for name, begin, end in device_events]
    measurement = json.loads((output / "measurement/RESULT.json").read_text())
    evaluation = json.loads((args.pair / "evaluation/EVALUATION.json").read_text())
    checks = {"complete48_native_renders_and_each_property": all(value == 48 for value in calls.values()),
              "all_original96_input_model_RNG_cache_guards": all(measurement["checks"].values()),
              "original_complete24_quality_guard": evaluation["development_metric_guard_passed"]
                  and evaluation["all_frames_included_in_denominator"] == 24,
              "native_global_method_descriptors_restored": installed["module"].render is installed["render"]
                  and all(getattr(installed["model_class"], name) is value for name, value in installed["properties"].items()),
              "wrapper_original_bindings_restored": evaluate_interior_ssim.activate_reference is original_activate and sys.argv is original_argv,
              "actual_CUDA_measurement_available": (len(event_rows) == 240 and all(row["elapsed_ms"] > 0 for row in event_rows))
                  if args.cuda_events else len(kernels) > 0,
              "all_current_native_source_files_unchanged": native_sources == [data_identity(row["path"]) for row in native_sources]}
    result = {"status": "complete_native_profile_passed" if all(checks.values()) else "failed_verification", "checks": checks,
              "calls": calls, "averages": averages, "cuda_kernel_events": kernels, "native_sources": native_sources,
              "cuda_stream_interval_events": event_rows, "CUDA_measurement_method": "stream_timing_events" if args.cuda_events else "profiler_kernel_events",
              "cuda_stream_interval_scope": "Complete operator stream intervals include launch gaps; property intervals are nested within render intervals and must not be added to renders",
              "measured_profile_kernel_total_us": sum(row["duration_us"] for row in kernels),
              "nonoverlap_sum_scope": "CUDA kernel events only; nested CPU region device_total values must not be added",
              "elapsed_seconds": time.perf_counter() - STARTED, "new_optimization_count": 0,
              "profile_timing_not_formal_performance": True}
    atomic_json(output / "PROFILE.json", result)
    print(json.dumps({"status": result["status"], "checks": checks, "calls": calls,
                      "kernel_ms": result["measured_profile_kernel_total_us"] / 1000}), flush=True)
    if not all(checks.values()):
        raise AssertionError("A complete native profiling protection failed")


if __name__ == "__main__":
    main()
