"""Run the complete current RAW initial pipeline with exact scoring input fusion."""

import argparse
import importlib
from pathlib import Path
import sys
import time

import cv2

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"
sys.path[:0] = [str(V1), str(ROOT / "02_code/box2_initial_pose_quality_v2_20261002")]
from runtime import atomic_json, data_identity
import run_initial_pipeline as entry
import quality_engine
import run_raw_histogram_quality as original
from exact_bayer_gray import raw_gray_quality_execution
from native_quality_workspace import native_quality_workspace_execution


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("native", "fused"), required=True)
    for name in ("library", "output", "quality-cache"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    baseline = importlib.import_module("initial_selection")
    sys.path.insert(0, str(baseline.TOOLS))
    from color_contract import demosaic_rggb8_to_bgr
    paths = [Path(__file__), Path(__file__).with_name("exact_bayer_gray.py"), Path(__file__).with_name("native_quality_workspace.py"),
             Path(original.__file__), V1 / "run_initial_pipeline.py", V1 / "quality_engine.py", args.library.resolve(strict=True)]
    sources = [data_identity(path) for path in paths]
    original_score, original_argv, original_threads = entry.score_messages, sys.argv, cv2.getNumThreads()
    if original_score is not quality_engine.score_messages:
        raise ValueError("The original complete RAW scoring alias has a different owner")
    binding_seconds = 0.
    calls = []

    def raw_score(messages, decoder, converter, scorer, *parameters, **keywords):
        if converter is not demosaic_rggb8_to_bgr or scorer is not quality_engine.compact_quality:
            raise ValueError("The exact gray adapter received a different original complete scoring consumer")
        calls.append({"workers": keywords.get("workers"), "capacity": keywords.get("capacity")})
        return original_score(messages, decoder, lambda raw: raw, scorer, *parameters, **keywords)

    trace = None
    try:
        context = (native_quality_workspace_execution(True) if args.mode == "native" else
                   raw_gray_quality_execution(args.library, demosaic_rggb8_to_bgr))
        with context as trace:
            before = time.perf_counter()
            if args.mode == "fused":
                entry.score_messages = raw_score
            binding_seconds += time.perf_counter()-before
            sys.argv = [str(Path(original.__file__)), "--quality-mode", "histogram", "--quality-cache", str(args.quality_cache),
                        "--output", str(args.output)]
            original.main()
            before = time.perf_counter()
            expected = raw_score if args.mode == "fused" else original_score
            if entry.score_messages is not expected:
                raise ValueError("Another writer changed the owned original RAW scoring alias")
            entry.score_messages = original_score
            binding_seconds += time.perf_counter()-before
        if sources != [data_identity(path) for path in paths]:
            raise ValueError("An original full initial scoring or export source changed")
        if args.mode == "fused" and calls != [{"workers": 4, "capacity": 16}]:
            raise ValueError("The original complete RAW quality consumer was omitted or duplicated")
        atomic_json(args.output / "FUSED_INPUT.json", {"sources": sources, "mode": args.mode, "trace": trace,
            "additional_input_lifecycle_seconds": trace["setup_seconds"]+trace["restore_seconds"]+binding_seconds,
            "binding_calls": calls, "original_quality_alias_restored": entry.score_messages is original_score,
            "JPEG_color_function_and_original_Q78_R99_Q50_Q53_P18_unchanged": True, "counted": False})
    finally:
        entry.score_messages = original_score
        sys.argv = original_argv
        cv2.setNumThreads(original_threads)


if __name__ == "__main__":
    main()
