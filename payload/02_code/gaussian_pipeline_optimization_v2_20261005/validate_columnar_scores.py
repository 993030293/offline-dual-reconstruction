"""Verify all original score rows plus immutable numeric-table misses and corruption."""

import argparse
import json
from pathlib import Path
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"
sys.path.insert(0, str(V1))
from runtime import atomic_json, data_identity
from quality_engine import FIELDS, ScoreCache
import run_initial_pipeline
from columnar_quality_scores import ColumnarScoreCache, checked_record, columnar_score_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "artifact", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    contract = {"fixture": "strict-keyed-image-quality-lifecycle"}
    cache = ColumnarScoreCache(ScoreCache, args.source, contract, args.artifact, args.output/"misses", FIELDS)
    paths = sorted(args.source.glob("*.json"))
    checks = {"all7338_source_rows": len(paths) == 7338}
    originals = [checked_record(path, FIELDS) for path in paths]
    checks["all_original_quality_dicts_exact"] = all(cache.read(key) == values for key, values, _ in originals)
    keys = [key for key, _, _ in originals]
    with ThreadPoolExecutor(max_workers=4) as pool:
        recovered = list(pool.map(cache.read, keys))
    checks["four_worker_reads_preserve_order_and_values"] = recovered == [values for _, values, _ in originals]
    checks["all_numeric_table_values_read_only"] = not cache.table.flags.writeable and not cache.keys.flags.writeable
    key = cache.key(b"unseen-independent-source-score-fixture")
    checks["unseen_key_is_a_miss"] = key not in cache.lookup and cache.read(key) is None
    sample = originals[0][1]
    cache.write(key, sample)
    checks["new_score_written_to_independent_miss_directory_only"] = cache.read(key) == sample and not (args.source/(key+".json")).exists()
    try:
        cache.write(keys[0], sample)
        checks["committed_score_overwrite_rejected"] = False
    except ValueError:
        checks["committed_score_overwrite_rejected"] = True
    bad = args.output/"corrupt_table"
    shutil.copytree(args.artifact, bad)
    with (bad/"scores.npy").open("r+b") as stream:
        stream.seek(-1, 2)
        value = stream.read(1)
        stream.seek(-1, 2)
        stream.write(bytes([value[0] ^ 1]))
    try:
        ColumnarScoreCache(ScoreCache, args.source, contract, bad, args.output/"bad_misses", FIELDS)
        checks["actual_corrupt_numeric_payload_rejected"] = False
    except ValueError:
        checks["actual_corrupt_numeric_payload_rejected"] = True
    incomplete = args.output/"incomplete_table"
    incomplete.mkdir()
    shutil.copyfile(args.artifact/"MANIFEST.json", incomplete/"MANIFEST.json")
    try:
        ColumnarScoreCache(ScoreCache, args.source, contract, incomplete, args.output/"incomplete_misses", FIELDS)
        checks["incomplete_table_rejected"] = False
    except FileNotFoundError:
        checks["incomplete_table_rejected"] = True
    try:
        ColumnarScoreCache(ScoreCache, args.output, contract, args.artifact, args.output/"wrong_source_misses", FIELDS)
        checks["wrong_source_directory_rejected"] = False
    except ValueError:
        checks["wrong_source_directory_rejected"] = True
    original_factory = run_initial_pipeline.ScoreCache
    try:
        with columnar_score_cache(run_initial_pipeline, args.artifact, args.output/"context_misses", FIELDS):
            raise RuntimeError("Intentional numeric-score adapter restoration check")
    except RuntimeError:
        pass
    checks["entry_factory_restored_after_error"] = run_initial_pipeline.ScoreCache is original_factory
    sources = [Path(__file__), Path(__file__).with_name("columnar_quality_scores.py"), V1/"quality_engine.py"]
    result = {"status": "lifecycle_passed" if all(checks.values()) else "failed_verification", "checks": checks,
        "cache": cache.receipt(), "sources": [data_identity(p) for p in sources], "artifact": str(args.artifact),
        "source": str(args.source), "counted": False, "actual_full_raw_entry_pairs_pending": True}
    atomic_json(args.output/"RESULT.json", result)
    print(json.dumps({"status": result["status"], "checks": checks}), flush=True)
    if not all(checks.values()):
        raise ValueError("The actual immutable numeric quality-score lifecycle did not pass")


if __name__ == "__main__":
    main()
