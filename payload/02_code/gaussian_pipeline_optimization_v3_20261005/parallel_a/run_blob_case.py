"""Measure the entire pinned localization entry with immutable BLOB views."""

import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
V1 = ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"
V2 = ROOT/"02_code/gaussian_pipeline_optimization_v2_20261005"
V3 = ROOT/"02_code/gaussian_pipeline_optimization_v3_20261005"
sys.path[:0] = [str(V1), str(V2), str(V3)]
from runtime import atomic_json, data_identity
import localize_prepared_queries
import byte_sift_matching
import sift_feature_store
from feature_blob_views import feature_blob_execution


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("copy", "view"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.perf_counter()
    paths = [Path(__file__), Path(__file__).with_name("feature_blob_views.py"),
        Path(byte_sift_matching.__file__), Path(sift_feature_store.__file__)]
    sources = [data_identity(path) for path in paths]
    preparation = time.perf_counter()-started
    old = sys.argv
    try:
        with feature_blob_execution(args.mode == "view") as trace:
            sys.argv = [str(Path(localize_prepared_queries.__file__)), "--preparation-mode", "parallel",
                "--output", str(args.output), "--reference-mode", "cache", "--reference-cache-root",
                str(ROOT/"04_outputs/gaussian_pipeline_optimization_v2_20261005/final_route_checks_001/reference_cache"),
                "--feature-mode", "cache", "--feature-cache-root",
                str(ROOT/"04_outputs/gaussian_pipeline_optimization_v2_20261005/final_route_checks_001/feature_cache"),
                "--export", str(ROOT/"04_outputs/gaussian_pipeline_optimization_v1_20261005/new_map_export_003"),
                "--database", str(ROOT/"04_outputs/gaussian_pipeline_optimization_v1_20261005/p23_track_completion_001/database.db"),
                "--initial-frames", str(ROOT/"04_outputs/gaussian_pipeline_optimization_v2_20261005/final_raw_cold_001/initial"),
                "--reference-run", str(ROOT/"04_outputs/box2_lio_guided_sfm_v1_20261002T014330")]
            localize_prepared_queries.main()
        before = time.perf_counter()
        if sources != [data_identity(path) for path in paths] or not trace["all_function_bindings_restored"]:
            raise ValueError("An original source or BLOB function binding changed")
        if trace["instances"] != trace["all_owned_store_closes"] or trace["instances"] == 0:
            raise ValueError("Not every actual store closed inside the full entry")
        if args.mode == "view" and trace["view_calls"] == 0:
            raise ValueError("The actual immutable BLOB view branch did not execute")
        closing = time.perf_counter()-before
        phase = preparation+closing+sum(trace[key] for key in (
            "setup_seconds", "restore_seconds", "decode_binding_setup_seconds", "decode_binding_restore_seconds",
            "initialization_seconds", "all_get_seconds", "close_seconds"))
        atomic_json(args.output/"BLOB_READ.json", {"mode": args.mode, "sources": sources, "trace": trace,
            "complete_read_stage_seconds": phase, "source_preparation_seconds": preparation,
            "final_source_verification_seconds": closing, "decode_not_double_counted_in_get": True,
            "unchanged_current_R90_R92_R94_R102": True, "counted": False})
    finally:
        sys.argv = old


if __name__ == "__main__":
    main()
