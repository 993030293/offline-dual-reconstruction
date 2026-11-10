"""Compose accepted raw streaming, exact pairing and bounded export in one guarded entry."""

import argparse
from contextlib import contextmanager, nullcontext
import inspect
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity
import run_initial_pipeline as native
from indexed_candidate_pairs import candidate_pairs
from parallel_raw_export import prepared_exports
from encoded_jpeg_cache import cached_jpeg_encoding
from single_copy_raw_reader import single_copy_sensor_reader
from windowed_raw_reader import windowed_sensor_reader


@contextmanager
def combined_raw_execution(cached):
    original = native.main
    source = inspect.getsource(original)
    pair_before = '''    positions = np.asarray([row["C_W_m"] for row in selected])
    pairs = set()
    for i, row in enumerate(selected):
        for j in range(i+1, min(len(selected), i+9)):
            pairs.add((i, j))
        for j in np.argsort(np.linalg.norm(positions-positions[i], axis=1))[:17]:
            if abs(int(j)-i) > 8:
                pairs.add(tuple(sorted((i, int(j)))))
'''
    export_before = '''    for exported, (frame, message) in enumerate(image_messages(selected_by_id), 1):
        stamp, _, raw = decode_bayer(message["payload"])
        row = selected_by_id[frame]
        if int(stamp) != row["camera_header_ns"]:
            raise ValueError("Raw image identity changed before export")
        ok, encoded = cv2.imencode(".jpg", demosaic_rggb8_to_bgr(raw),
                                 [cv2.IMWRITE_JPEG_QUALITY, baseline.POLICY["jpeg_quality"]])
'''
    export_after = '''    for exported, (frame, stamp, ok, encoded) in enumerate(_effective_exports(
            image_messages(selected_by_id), decode_bayer, demosaic_rggb8_to_bgr, baseline.POLICY["jpeg_quality"]), 1):
        row = selected_by_id[frame]
        if int(stamp) != row["camera_header_ns"]:
            raise ValueError("Raw image identity changed before export")
'''
    if source.count(pair_before) != 1 or source.count(export_before) != 1:
        raise ValueError("The single guarded raw-entry composition no longer matches the native source")
    trace = {"pair_calls": [], "export_calls": [], "single_compiled_source_preserves_both_adapters": True}
    generators = []

    def pairs(selected):
        before = time.perf_counter()
        values, row = candidate_pairs(selected, True)
        trace["pair_calls"].append({**row, "seconds": time.perf_counter()-before})
        return values

    def exports(messages, decoder, converter, quality):
        row = {}
        trace["export_calls"].append(row)
        generator = prepared_exports(messages, decoder, converter, quality, workers=4, capacity=8,
            encode_in_workers=not cached, trace=row)
        generators.append(generator)
        return generator

    namespace = dict(native.__dict__)
    namespace.update(_effective_pairs=pairs, _effective_exports=exports)
    source = source.replace(pair_before, "    pairs = _effective_pairs(selected)\n").replace(export_before, export_after)
    exec(compile(source, "<accepted_single_source_raw_execution_combination>", "exec"), namespace)
    native.main = namespace["main"]
    try:
        yield trace
    finally:
        for generator in generators:
            generator.close()
        native.main = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["cold", "repeat", "mapped-repair"], required=True)
    parser.add_argument("--quality-cache", type=Path, required=True)
    parser.add_argument("--jpeg-cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if bool(args.jpeg_cache) != (args.mode == "repeat"):
        raise ValueError("Only the repeated export route can use its explicit exact JPEG cache")
    args.output.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), Path(native.__file__)]+[Path(__file__).with_name(name) for name in
        ("indexed_candidate_pairs.py", "parallel_raw_export.py", "encoded_jpeg_cache.py", "single_copy_raw_reader.py", "windowed_raw_reader.py")]
    sources = [data_identity(path) for path in paths]
    atomic_json(args.output/"CONTRACT.json", {"status": "frozen_before_actual_combination", "sources": sources,
        "mode": args.mode, "quality_cache": str(args.quality_cache), "JPEG_cache": str(args.jpeg_cache) if args.jpeg_cache else None,
        "JPEG_cache_ownership_main_thread_only": True, "workers": 4, "capacity": 8,
        "all7338_raw_only_decisions1002_JPEG11583_pairs_unchanged": True, "selected_frame_budget_reduction": False,
        "mapped_branch_is_optional_repair_not_default": True, "new_optimization_count": 0})
    reader = windowed_sensor_reader if args.mode == "mapped-repair" else single_copy_sensor_reader
    cache = cached_jpeg_encoding(args.jpeg_cache) if args.jpeg_cache else nullcontext(None)
    original_argv = sys.argv
    try:
        with reader(True) as input_trace, cache as store, combined_raw_execution(bool(args.jpeg_cache)) as execution:
            sys.argv = [str(Path(native.__file__)), "--output", str(args.output/"initial"), "--cache", str(args.quality_cache),
                "--workers", "4", "--scorer", "compact", "--selector", "batched", "--pose-mode", "batched", "--reader", "indexed"]
            native.main()
        exports = execution["export_calls"]
        if len(execution["pair_calls"]) != 1 or len(exports) != 1 or not exports[0]["closed"] or (
                exports[0]["submitted"], exports[0]["consumed"]) != (1002, 1002):
            raise ValueError("The actual combined raw entry omitted an accepted adapter or selected image")
        if sources != [data_identity(path) for path in paths]:
            raise ValueError("An accepted raw combination source changed during execution")
        atomic_json(args.output/"COMBINATION.json", {"status": "actual_raw_combination_completed_original_output_oracle_pending",
            "sources": sources, "input": input_trace, "execution": execution, "JPEG_cache": store.receipt() if store else None,
            "new_optimization_count": 0})
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
