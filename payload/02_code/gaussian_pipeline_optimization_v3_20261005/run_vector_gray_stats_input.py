"""Compose exact vector moments with the existing full RAW and corner consumer."""

import argparse
from contextlib import nullcontext
from pathlib import Path
import sys
import time

import cv2

import run_exact_raw_gray_input as accepted
import exact_bayer_gray
from planar_corner import planar_corner_execution, ORIGINAL_CORNER
from vector_gray_stats import vector_gray_stats_execution

ROOT = accepted.ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("current", "vector"), required=True)
    for name in ("library", "gray-library", "corner-library", "output", "quality-cache"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    paths = [Path(__file__), Path(__file__).with_name("vector_gray_stats.py"), Path(__file__).with_name("planar_corner.py"),
             Path(accepted.__file__), Path(exact_bayer_gray.__file__), args.library.resolve(strict=True),
             args.gray_library.resolve(strict=True), args.corner_library.resolve(strict=True)]
    sources = [accepted.data_identity(path) for path in paths]
    old_argv, old_factory = sys.argv, exact_bayer_gray.compile_quality
    try:
        before = time.perf_counter()
        with planar_corner_execution(args.corner_library) as corner_trace:
            corner_setup = time.perf_counter() - before
            before = time.perf_counter()
            with (vector_gray_stats_execution(args.library) if args.mode == "vector" else nullcontext()) as vector_trace:
                vector_setup = time.perf_counter() - before
                sys.argv = [str(Path(accepted.__file__)), "--mode", "fused", "--library", str(args.gray_library),
                            "--output", str(args.output), "--quality-cache", str(args.quality_cache)]
                accepted.main()
                before = time.perf_counter()
            vector_restore = time.perf_counter() - before
            before = time.perf_counter()
        corner_restore = time.perf_counter() - before
        if sources != [accepted.data_identity(path) for path in paths]:
            raise ValueError("A frozen complete vector input source or DLL changed")
        if exact_bayer_gray.compile_quality is not old_factory or cv2.goodFeaturesToTrack is not ORIGINAL_CORNER:
            raise ValueError("The original complete quality ownership was not restored")
        if not (corner_trace["original_factory_restored"] and corner_trace["workspace"]["DLL_unloaded"]
                and corner_trace["workspace"]["native_calls"] == 7338 and not corner_trace["workspace"]["fallback_calls"]
                and not corner_trace["workspace"]["retained_plane_bytes"] and not corner_trace["workspace"]["active_calls"]):
            raise ValueError("The incumbent full7338 corner denominator or close contract failed")
        if args.mode == "vector" and not (vector_trace["original_factory_restored"] and vector_trace["engine"]["DLL_unloaded"]
                and vector_trace["engine"]["native_calls"] == 7338 and not vector_trace["engine"]["fallback_calls"]
                and not vector_trace["engine"]["retained_output_bytes"] and not vector_trace["engine"]["active_calls"]):
            raise ValueError("The full7338 native vector denominator or close contract failed")
        accepted.atomic_json(args.output / "CORNER_INPUT.json", {"sources": sources, "mode": args.mode, "trace": corner_trace,
            "additional_corner_lifecycle_seconds": corner_setup + corner_restore, "original_factory_restored": True,
            "original_cv2_module_unchanged": True, "counted": False})
        accepted.atomic_json(args.output / "VECTOR_INPUT.json", {"sources": sources, "mode": args.mode, "trace": vector_trace,
            "additional_statistics_lifecycle_seconds": vector_setup + vector_restore, "original_factory_restored": True,
            "counted": False})
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
