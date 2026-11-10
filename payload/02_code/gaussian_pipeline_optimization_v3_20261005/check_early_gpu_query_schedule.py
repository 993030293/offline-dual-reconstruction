"""Validate real process drain and private-hook restoration without claiming GPU science."""

import argparse
import json
from pathlib import Path
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"),
               str(ROOT / "02_code/gaussian_pipeline_optimization_v2_20261005"),
               str(ROOT / "02_code/box2_lio_guided_sfm_v1_20261002")]
from runtime import atomic_json, data_identity
from early_gpu_query_provider import EarlyGPUQueryProvider, early_gpu_backend
import feature_localization


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.resolve().relative_to(ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/gsf001/bootstrap_dag_001")
    args.output.mkdir(parents=True, exist_ok=False)
    checks, traces = {}, []
    sources = [data_identity(path) for path in (Path(__file__), Path(__file__).with_name("early_gpu_query_provider.py"),
                                                Path(__file__).with_name("localize_early_gpu_query.py"))]
    snapshots = args.output / "source_snapshot"
    snapshots.mkdir()
    for number, row in enumerate(sources):
        (snapshots / (str(number) + "_" + Path(row["path"]).name)).write_bytes(Path(row["path"]).read_bytes())
    atomic_json(args.output / "CONTRACT.json", {"sources": sources, "fixture_scope": "real_process_and_private_binding_lifecycle_only",
        "GPU_features_math_or_cost_not_claimed": True, "counted": False})
    original = feature_localization.FeatureBank.extra
    for index, error_type in enumerate((None, RuntimeError, KeyboardInterrupt, SystemExit)):
        output = args.output / ("normal" if error_type is None else error_type.__name__)
        provider = EarlyGPUQueryProvider(output, command=[sys.executable, "-c", "import time;time.sleep(.05)"])
        entered = False
        try:
            with early_gpu_backend([args.output / "original_query.jpg"], output, gpu=True, producer=provider):
                entered = feature_localization.FeatureBank.extra is not original
                if error_type:
                    raise error_type("registered_early_GPU_consumer_failure")
        except (RuntimeError, KeyboardInterrupt, SystemExit) as error:
            if type(error) is not error_type or str(error) != "registered_early_GPU_consumer_failure":
                raise
        owner = provider.trace["broker"]["profile_owner"]
        try:
            exited = psutil.Process(owner["pid"]).create_time() != owner["creation_time"]
        except psutil.NoSuchProcess:
            exited = True
        name = "normal" if error_type is None else error_type.__name__
        checks[name + "_real_child_drained_and_native_hook_restored"] = entered and exited and provider.child.returncode == 0
        checks[name + "_private_trace_and_original_failure_preserved"] = provider.trace["binding_restored"] and provider.trace["broker"]["closed"]
        traces.append(provider.trace)
    output = args.output / "producer_failure"
    provider = EarlyGPUQueryProvider(output, command=[sys.executable, "-c", "raise SystemExit(7)"])
    rejected = False
    try:
        with early_gpu_backend([args.output / "original_query.jpg"], output, gpu=True, producer=provider):
            provider.prepare((args.output / "original_query.jpg",))
    except RuntimeError as error:
        rejected = "CPU fallback is forbidden" in str(error)
    checks["real_nonzero_producer_error_propagated_no_CPU_fallback"] = rejected and provider.child.returncode == 7
    checks["real_nonzero_producer_binding_restored_and_closed"] = provider.trace["binding_restored"] and provider.trace["broker"]["closed"]
    traces.append(provider.trace)
    checks["all_frozen_new_sources_and_native_extra_restored"] = sources == [data_identity(Path(row["path"])) for row in sources]
    checks["native_extra_ownership_original_at_end"] = feature_localization.FeatureBank.extra is original
    result = {"status": "real_process_scheduling_generic_passed_complete_GPU_consumer_pending" if all(checks.values()) else "failed_verification",
              "checks": checks, "sources": sources, "traces": traces, "counted": False}
    atomic_json(args.output / "RESULT.json", result)
    print({"status": result["status"], "checks": checks, "counted": False}, flush=True)
    if not all(checks.values()):
        raise AssertionError("An original early producer process or binding contract failed")


if __name__ == "__main__":
    main()
