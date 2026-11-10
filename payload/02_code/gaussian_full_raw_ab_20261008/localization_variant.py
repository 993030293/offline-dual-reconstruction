"""Compare preserved localization mechanisms on one complete fresh map and raw queries."""

import argparse
from contextlib import ExitStack
import inspect
import json
from pathlib import Path
import sys
import textwrap
import time

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"
V2 = ROOT / "02_code/gaussian_pipeline_optimization_v2_20261005"
V3 = ROOT / "02_code/gaussian_pipeline_optimization_v3_20261005"
sys.path[:0] = [str(V1), str(V2), str(V3)]
from runtime import data_identity
from atomic_json_windows import atomic_json
import localize_evaluation

MECHANISMS = ("eager_original", "eager_metadata", "lazy_original", "lazy_metadata", "bytes", "indexed",
              "integer", "prepared", "process", "strict", "ordered")


def ordered_main(function, pipeline):
    source = textwrap.dedent(inspect.getsource(function))
    start = "        for source in evaluation:\n"
    end = "        if not np.array_equal(parameters, camera.params) or identities != [data_identity(path) for path in sources]:\n"
    branch = '            if not source["coarse_pose_valid"]:\n'
    if source.count(start) != 1 or source.count(end) != 1 or source.count(branch) != 1:
        raise ValueError("The current ordered raw-query preparation boundary changed")
    prefix, body = source.split(start)
    body, suffix = body.split(end)
    preparation, calculation = body.split(branch)
    preparation = textwrap.dedent(preparation)
    producer = "        def prepare(source):\n"+textwrap.indent(preparation, "            ")
    producer += "            return source, before, name, image_path, row\n"
    replacement = producer+"        for source, before, name, image_path, row in owned_pipeline.iterate(evaluation, prepare):\n"+branch+calculation
    namespace = dict(function.__globals__)
    namespace["owned_pipeline"] = pipeline
    exec(compile(prefix+replacement+end+suffix, "<ordered_current_raw_reference_preparation>", "exec"), namespace)
    return namespace["main"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mechanism", choices=MECHANISMS, required=True)
    parser.add_argument("--compat-adapter", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    paths = [Path(__file__), Path(localize_evaluation.__file__), V2 / "byte_sift_matching.py",
        V3 / "native_observation_index.py", V3 / "array_match_geometry.py", V3 / "integer_sift_matching.py",
        V3 / "integer_sift_worker.py", V3 / "native_sift_process.py", V3 / "strict_sift_encoding.py",
        V3 / "query_preparation_pipeline.py"]
    if args.compat_adapter:
        paths.append(Path(__file__).with_name("localization_contract_adapter.py"))
    identities = [data_identity(path) for path in paths]
    atomic_json(args.output / "CONTRACT.json", {"sources": identities, "mechanism": args.mechanism,
        "full_raw_denominator": 24, "current_own_map_only": str(args.side), "cache_state": "fresh private outputs",
        "all_original_feature_geometry_pose_thresholds_retained": True, "quality_acceptance_not_inferred_from_oracle": True})
    old_store, old_argv = localize_evaluation.SiftFeatureStore, sys.argv
    trace = {}
    eager = args.mechanism.startswith("eager_")
    original_neighbors = args.mechanism.endswith("_original")

    def store(database, names, mode, capacity):
        return old_store(database, names, "eager" if eager else "lazy", capacity)

    try:
        localize_evaluation.SiftFeatureStore = store
        with ExitStack() as stack:
            advanced = args.mechanism in ("integer", "prepared", "process", "strict", "ordered")
            if args.mechanism in ("bytes", "indexed") or advanced:
                from byte_sift_matching import byte_sift_matching
                trace["byte"] = stack.enter_context(byte_sift_matching())
            if args.mechanism == "indexed" or advanced:
                from native_observation_index import native_observation_index
                trace["index"] = stack.enter_context(native_observation_index(args.output / "index"))
            if advanced:
                import integer_sift_matching
                if args.mechanism == "strict":
                    from strict_sift_encoding import StrictSIFTEncoder, receipt
                    encoder = StrictSIFTEncoder(True)
                    original_encoding = integer_sift_matching.bytes_exact
                    integer_sift_matching.bytes_exact = encoder
                    stack.callback(setattr, integer_sift_matching, "bytes_exact", original_encoding)
                trace["integer"] = stack.enter_context(integer_sift_matching.integer_sift_matching(args.output / "integer_worker.log"))
            if args.mechanism in ("prepared", "process", "strict", "ordered"):
                from array_match_geometry import array_match_geometry
                trace["prepared"] = stack.enter_context(array_match_geometry())
            if args.mechanism == "process":
                if args.compat_adapter:
                    from localization_contract_adapter import exact_worker_options
                    trace["worker_options_adapter"] = stack.enter_context(exact_worker_options())
                from native_sift_process import native_sift_process
                evaluations = json.loads((args.side / "initial/EVALUATION_FRAMES.json").read_text())["rows"]
                queries = [args.side / "initial/evaluation_images" / row["image_name"] for row in evaluations if row["coarse_pose_valid"]]
                trace["sift_process"] = stack.enter_context(native_sift_process(queries))
            entry = localize_evaluation.main
            if args.mechanism == "ordered":
                from query_preparation_pipeline import OrderedQueryPreparation
                pipeline = OrderedQueryPreparation(True, 2)
                stack.callback(pipeline.close)
                trace["ordered"] = pipeline.trace
                entry = ordered_main(entry, pipeline)
            if args.compat_adapter and (args.mechanism == "indexed" or advanced):
                from localization_contract_adapter import exact_transient_training_view
                trace["descriptor_composition_adapter"] = stack.enter_context(exact_transient_training_view())
            sys.argv = [str(Path(localize_evaluation.__file__)), "--side", str(args.side), "--output", str(args.output / "localization"),
                        "--mode", "baseline" if original_neighbors else "optimized"]
            entry()
            if args.mechanism == "strict":
                trace["strict_encoding"] = receipt(encoder)
        if identities != [data_identity(path) for path in paths]:
            raise ValueError("A preserved localization mechanism changed during the current trial")
        result = json.loads((args.output / "localization/LOCALIZATION.json").read_text())
        if len(result["rows"]) != 24:
            raise ValueError("A localization trial removed a failed raw evaluation frame")
        atomic_json(args.output / "RESULT.json", {"status": "complete24_current_queries_processed", "mechanism": args.mechanism,
            "explicit_compatibility_adapter": args.compat_adapter,
            "localization_status": result["status"], "localization_summary": result["summary"], "trace": trace,
            "sources_unchanged": True, "elapsed_seconds": time.perf_counter()-started,
            "module_guard_pass_is_not_all24_quality_acceptance": True})
    except BaseException as error:
        atomic_json(args.output / "FAILURE.json", {"status": "failed_current_interface_or_oracle", "error": repr(error),
            "mechanism": args.mechanism, "elapsed_seconds": time.perf_counter()-started, "partial_trace": trace})
        raise
    finally:
        localize_evaluation.SiftFeatureStore = old_store
        sys.argv = old_argv


if __name__ == "__main__":
    main()
