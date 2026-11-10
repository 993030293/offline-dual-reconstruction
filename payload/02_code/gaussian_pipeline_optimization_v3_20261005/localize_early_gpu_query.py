"""Overlap the unchanged GPU producer with full incumbent localization startup."""

import argparse
import inspect
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).parent
OWN = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/gsf001/bootstrap_dag_001"
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity
from early_gpu_query_provider import EarlyGPUQueryProvider, early_gpu_backend


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("cpu_process", "gpu"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(OWN)
    output.mkdir(parents=True, exist_ok=False)
    producer, failure = None, None
    original_argv = sys.argv
    bootstrap_sources = [data_identity(path) for path in (Path(__file__), CODE / "early_gpu_query_provider.py", CODE / "localize_gpu_query_sift.py")]
    snapshots = output / "bootstrap_source_snapshot"
    snapshots.mkdir()
    for number, row in enumerate(bootstrap_sources):
        (snapshots / (str(number) + "_" + Path(row["path"]).name)).write_bytes(Path(row["path"]).read_bytes())
    try:
        if args.mode == "gpu":
            producer = EarlyGPUQueryProvider(output / "gpu_feature_batch")
        producer_started_before_heavy_imports = time.perf_counter()
        import localize_gpu_query_sift as original
        from array_match_geometry import array_match_geometry
        source = inspect.getsource(original.main)
        changes = {
            "    args.output.mkdir(parents=True, exist_ok=False)": "    _require_owned_bootstrap(args.output)",
            'paths = [Path(__file__), CODE/"native_sift_process.py",':
                'paths = [Path(__file__), CODE/"early_gpu_query_provider.py", CODE/"localize_gpu_query_sift.py", CODE/"array_match_geometry.py", CODE/"native_sift_process.py",',
            "with owned_atomic_json(args.output) as write_trace, query_backend(":
                "with owned_atomic_json(args.output) as write_trace, array_match_geometry() as geometry_trace, query_backend(",
            "        final_guard_seconds = time.perf_counter()-guard_started":
                '        checks["all48_common_R120_prepared_geometry_routes"] = len(geometry_trace["routes"]) == 48 and all(route == "owned_exact_array" for route in geometry_trace["routes"]) and geometry_trace["binding_restored"]\n        final_guard_seconds = time.perf_counter()-guard_started',
            '"trace": trace, "write_trace": write_trace, "worker":':
                '"trace": trace, "write_trace": write_trace, "common_array_geometry_trace": geometry_trace, "worker":',
        }
        for anchor, replacement in changes.items():
            if source.count(anchor) != 1:
                raise ValueError("The original full cold localization bootstrap anchor changed: " + anchor)
            source = source.replace(anchor, replacement)

        def require_owned(directory):
            if Path(directory).resolve() != output or not snapshots.is_dir():
                raise ValueError("The original consumer no longer belongs to its unique bootstrap directory")

        def backend(paths, directory, gpu):
            return early_gpu_backend(paths, directory, gpu=gpu, producer=producer)

        namespace = dict(vars(original))
        namespace.update(__file__=str(Path(__file__)), OWNED=OWN, _require_owned_bootstrap=require_owned,
                         array_match_geometry=array_match_geometry, query_backend=backend)
        (output / "COMPOSED_FULL_ENTRY.py").write_text(source, encoding="utf-8")
        sys.argv = [str(Path(__file__)), "--mode", args.mode, "--output", str(output)]
        exec(compile(source, "<early-owned-GPU-query-complete-entry>", "exec"), namespace)
        namespace["main"]()
        result = json.loads((output / "RESULT.json").read_text(encoding="utf-8"))
        if not all(result["checks"].values()) or bootstrap_sources != [data_identity(Path(row["path"])) for row in bootstrap_sources]:
            raise ValueError("The original full localization guard or bootstrap source changed")
        atomic_json(output / "BOOTSTRAP.json", {"status": "complete_current_incumbent_or_early_GPU_bootstrap_passed",
            "mode": args.mode, "sources": bootstrap_sources, "original_full_result": str(output / "RESULT.json"),
            "producer_started_before_heavy_imports_monotonic": producer_started_before_heavy_imports,
            "trace": producer.trace if producer else None, "complete_entry_before_last_publication_seconds": time.perf_counter() - started,
            "old_complete_query_stage_sum_is_diagnostic_not_acceptance": True,
            "acceptance_phase_is_parent_full_Popen_through_actual_exit_envelope": True,
            "common_R120_R124_CPU_and_R90_R92_R94_R102_unchanged": True, "counted": False})
        print({"status": "complete_current_incumbent_or_early_GPU_bootstrap_passed", "mode": args.mode,
               "complete_entry_seconds": time.perf_counter() - started,
               "first_GPU_wait_seconds": producer.trace["first_wait_seconds"] if producer else None}, flush=True)
    except BaseException as error:
        failure = {"type": type(error).__name__, "message": str(error)}
        atomic_json(output / "BOOTSTRAP_FAILURE.json", {"status": "failed", "failure": failure,
            "trace": producer.trace if producer else None, "elapsed_seconds": time.perf_counter() - started, "counted": False})
        raise
    finally:
        sys.argv = original_argv
        code = producer.drain() if producer else None
        if producer:
            producer.trace["broker"]["closed"] = True
        atomic_json(output / "BOOTSTRAP_CLOSE.json", {"source_and_argv_scope": "only_owned_complete_entry",
            "argv_restored": sys.argv is original_argv, "producer_drained_returncode": code,
            "original_consumer_failure": failure, "trace": producer.trace if producer else None,
            "elapsed_seconds": time.perf_counter() - started, "counted": False})
        if code and failure is None:
            raise RuntimeError("An early owned GPU producer failed before complete closure")


if __name__ == "__main__":
    main()
