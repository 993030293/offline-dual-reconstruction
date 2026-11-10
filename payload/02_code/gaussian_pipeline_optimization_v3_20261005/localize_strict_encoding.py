"""Keep the full fusion science common and vary only checked CPU encoding."""

import argparse
import inspect
from pathlib import Path
import sys
import time

STARTED = time.perf_counter()
import localize_geometry_sift_fusion as preserved
from strict_encoding_owned_json_write import owned_atomic_json, COMPOSED_SOURCE
from strict_sift_encoding import StrictSIFTEncoder, receipt

OUT = preserved.ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/oup019/R171"


def compose(enabled):
    source = inspect.getsource(preserved.main)
    edits = [
        ('paths = [Path(__file__), CODE/"geometry_sift_fusion_node.py",',
         'paths = [Path(__file__), CODE/"localize_geometry_sift_fusion.py",\n'
         '        *[CODE/name for name in ("strict_sift_encoding.py", "native_sift_encoding.cpp",\n'
         '            "strict_encoding_owned_json_write.py", "strict_encoding_node.py", "codec_localization_node.py",\n'
         '            "sift_process_node_config.py", "run_sift_process_campaign_005.py")],\n'
         '        ROOT/"04_outputs/gaussian_pipeline_optimization_v3_20261005/oup019/build0/native_sift_encoding.dll",\n'
         '        CODE/"geometry_sift_fusion_node.py",'),
        ('original, argv, workers = integer_sift_matching.DistanceWorker, sys.argv, []',
         'original, argv, workers = integer_sift_matching.DistanceWorker, sys.argv, []\n'
         '    original_bytes = integer_sift_matching.bytes_exact\n'
         '    encoder = StrictSIFTEncoder(ENCODING_ENABLED)\n'
         '    globals()["ENCODER_INSTANCE"] = encoder'),
        ('integer_sift_matching.DistanceWorker = observe',
         'integer_sift_matching.DistanceWorker = observe\n'
         '        integer_sift_matching.bytes_exact = encoder'),
        ('<= 32*1024**3', '<= 16*1024**3'),
        ('        atomic_json(args.output/"RESULT.json",',
         '        checks["all_original_complete_byte_conversion_calls_measured"] = bool(encoder.calls)\n'
         '        checks["all_supported_native_or_original_six_output_routes"] = all(\n'
         '            call["status"] == "completed" and len(call["actual_successful_numpy_outputs"])\n'
         '                == (1 if call["route"] == "native_single_scan" else 6) for call in encoder.calls)\n'
         '        atomic_json(args.output/"RESULT.json",'),
        ('"mode": args.mode, "checks": checks, "localization":',
         '"encoding_mode": "native" if ENCODING_ENABLED else "raw", "encoding": encoder_receipt(encoder),\n'
         '            "actual_array_output_bytes": encoder_receipt(encoder)["actual_recorded_successful_output_bytes"],\n'
         '            "mode": args.mode, "checks": checks, "localization":'),
        ('        integer_sift_matching.DistanceWorker = original',
         '        integer_sift_matching.DistanceWorker = original\n'
         '        integer_sift_matching.bytes_exact = original_bytes'),
        ('"argv_restored": sys.argv is argv, "new_optimization_count": 0}',
         '"argv_restored": sys.argv is argv, "bytes_exact_restored": integer_sift_matching.bytes_exact is original_bytes,\n'
         '            "new_optimization_count": 0}'),
    ]
    for before, after in edits:
        if source.count(before) != 1:
            raise ValueError("The original full fusion source boundary changed: " + before[:80])
        source = source.replace(before, after)
    namespace = dict(preserved.main.__globals__)
    namespace.update(OWNED=OUT, __file__=str(Path(__file__).resolve()), STARTED=STARTED,
        ENCODING_ENABLED=enabled, StrictSIFTEncoder=StrictSIFTEncoder, encoder_receipt=receipt,
        owned_atomic_json=owned_atomic_json, COMPOSED_SOURCE=COMPOSED_SOURCE)
    exec(compile(source, str(Path(__file__)), "exec"), namespace)
    return namespace["main"], source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoding-mode", choices=("raw", "native"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    old = sys.argv
    try:
        sys.argv = [str(Path(__file__)), "--mode", "process", "--output", str(args.output)]
        function, _ = compose(args.encoding_mode == "native")
        function()
    finally:
        sys.argv = old


if __name__ == "__main__":
    main()
