"""Compose the private planar corner path around the current complete RAW consumer."""

import argparse
from contextlib import nullcontext
from pathlib import Path
import sys
import time

import cv2

import run_exact_raw_gray_input as accepted
import exact_bayer_gray
from planar_corner import planar_corner_execution,ORIGINAL_CORNER

ROOT=accepted.ROOT


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",choices=("current","planar"),required=True)
    for name in ("library","gray-library","output","quality-cache"):
        parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args()
    paths=[Path(__file__),Path(__file__).with_name("planar_corner.py"),Path(accepted.__file__),Path(exact_bayer_gray.__file__),
        args.library.resolve(strict=True),args.gray_library.resolve(strict=True)]
    sources=[accepted.data_identity(path) for path in paths]
    old_argv,old_factory=sys.argv,exact_bayer_gray.compile_quality
    setup=restore=0.
    trace=None
    try:
        before=time.perf_counter()
        context=planar_corner_execution(args.library) if args.mode=="planar" else nullcontext()
        with context as trace:
            setup=time.perf_counter()-before
            sys.argv=[str(Path(accepted.__file__)),"--mode","fused","--library",str(args.gray_library),
                "--output",str(args.output),"--quality-cache",str(args.quality_cache)]
            accepted.main()
            before=time.perf_counter()
        restore=time.perf_counter()-before
        if sources!=[accepted.data_identity(path) for path in paths]:
            raise ValueError("An original complete planar input source or DLL changed")
        if exact_bayer_gray.compile_quality is not old_factory or cv2.goodFeaturesToTrack is not ORIGINAL_CORNER:
            raise ValueError("The original complete corner namespace ownership was not restored")
        if args.mode=="planar" and (not trace["original_factory_restored"] or not trace["workspace"]["DLL_unloaded"] or trace["workspace"]["native_calls"]!=7338 or trace["workspace"]["fallback_calls"] or trace["workspace"]["retained_plane_bytes"] or trace["workspace"]["active_calls"]):
            raise ValueError("The original full7338 planar denominator or closure failed")
        accepted.atomic_json(args.output/"CORNER_INPUT.json",{"sources":sources,"mode":args.mode,"trace":trace,
            "additional_corner_lifecycle_seconds":setup+restore,"original_factory_restored":True,
            "original_cv2_module_unchanged":True,"counted":False})
    finally:
        sys.argv=old_argv


if __name__=="__main__":
    main()
