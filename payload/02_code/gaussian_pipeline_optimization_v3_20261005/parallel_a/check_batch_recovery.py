"""Inject entry cancellation and frozen-cache misses before opening raw resources."""

import argparse
from pathlib import Path
import sys

import run_selection_case
import run_blob_case
import quality_engine
import selection_engine
import run_initial_pipeline
import localize_with_byte_sift
import byte_sift_matching
import sift_feature_store
from runtime import atomic_json, data_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quality-cache", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), Path(run_selection_case.__file__), Path(run_blob_case.__file__),
        Path(__file__).with_name("spatial_novelty.py"), Path(__file__).with_name("feature_blob_views.py")]
    sources = [data_identity(path) for path in paths]
    original_argv = sys.argv
    original_selection_main = run_selection_case.run_raw_calibration.main
    original_blob_main = run_blob_case.localize_prepared_queries.main
    original_write = quality_engine.ScoreCache.write
    original_choose = selection_engine.choose_batched
    original_entry_choose = run_initial_pipeline.choose_batched
    original_open = Path.open
    original_atomic = run_initial_pipeline.atomic_json
    original_byte, original_entry_byte = byte_sift_matching.byte_sift_matching, localize_with_byte_sift.byte_sift_matching
    original_store = tuple(getattr(sift_feature_store.SiftFeatureStore, key) for key in ("__init__", "__getitem__", "close", "decode"))
    rows = []
    def cancel():
        raise KeyboardInterrupt("injected_owned_entry_cancel_before_raw_SQL_or_GPU_open")
    def miss():
        quality_engine.ScoreCache.write(None, "injected_missing_key", {})
    try:
        for label, target, arguments, failure in (
            ("selector_cancel", run_selection_case, ["--mode", "hash", "--quality-cache", str(args.quality_cache)], cancel),
            ("selector_frozen_cache_miss", run_selection_case, ["--mode", "hash", "--quality-cache", str(args.quality_cache)], miss),
            ("blob_consumer_cancel", run_blob_case, ["--mode", "view"], cancel)):
            output = args.output/label
            sys.argv = [str(Path(target.__file__)), *arguments, "--output", str(output)]
            if target is run_selection_case:
                run_selection_case.run_raw_calibration.main = failure
            else:
                run_blob_case.localize_prepared_queries.main = failure
            caller_argv = sys.argv
            caught = None
            try:
                target.main()
            except (KeyboardInterrupt, ValueError) as error:
                caught = {"type": type(error).__name__, "message": str(error)}
            restored = (quality_engine.ScoreCache.write is original_write and selection_engine.choose_batched is original_choose
                and run_initial_pipeline.choose_batched is original_entry_choose and sys.argv is caller_argv
                and Path.open is original_open and run_initial_pipeline.atomic_json is original_atomic
                and byte_sift_matching.byte_sift_matching is original_byte and localize_with_byte_sift.byte_sift_matching is original_entry_byte
                and tuple(getattr(sift_feature_store.SiftFeatureStore, key) for key in ("__init__", "__getitem__", "close", "decode")) == original_store)
            expected = "ValueError" if label.endswith("cache_miss") else "KeyboardInterrupt"
            rows.append({"case": label, "caught": caught, "exact_exception_and_no_partial_output": caught is not None
                and caught["type"] == expected and not output.exists(), "all_original_bindings_and_argv_restored": restored,
                "injection_before_actual_RAW_SQL_GPU_resources": True})
    finally:
        run_selection_case.run_raw_calibration.main = original_selection_main
        run_blob_case.localize_prepared_queries.main = original_blob_main
        sys.argv = original_argv
    checks = {"all3_cancel_miss_owned_entry_routes_restored": len(rows) == 3 and all(row["exact_exception_and_no_partial_output"]
        and row["all_original_bindings_and_argv_restored"] for row in rows), "sources_unchanged": sources == [data_identity(path) for path in paths]}
    atomic_json(args.output/"RESULT.json", {"status": "passed" if all(checks.values()) else "failed", "checks": checks, "rows": rows,
        "sources": sources, "hard_kill_or_WSL_cancellation_not_tested": True, "counted": False})
    if not all(checks.values()):
        raise ValueError("An entry cancellation or frozen-cache miss leaked a binding or artifact")


if __name__ == "__main__":
    main()
