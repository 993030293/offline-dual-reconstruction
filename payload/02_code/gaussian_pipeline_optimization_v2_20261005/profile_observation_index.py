"""Measure cold geometry-index construction against real localization accesses."""

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import pycolmap

from observation_index import FrozenObservationIndex

V1 = Path(__file__).resolve().parents[1]/"gaussian_pipeline_optimization_v1_20261005"
sys.path.insert(0, str(V1))
from runtime import atomic_json, data_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-localization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.source_localization/"H_LOCALIZATION.json"
    localization = json.loads(source.read_text(encoding="utf-8"))
    trace = json.loads((args.source_localization/"FEATURE_ACCESS_TRACE.json").read_text(encoding="utf-8"))
    model_path = Path(localization["map_source"])
    sources = [source, args.source_localization/"FEATURE_ACCESS_TRACE.json", *sorted(model_path.glob("*.bin")),
               Path(__file__), Path(__file__).with_name("observation_index.py")]
    identities = [data_identity(p) for p in sources]
    atomic_json(args.output/"CONTRACT.json", {"maximum_seconds": 120, "maximum_RSS_GiB": 8,
        "inputs": identities, "three_pairs": True, "cold_build_included": True,
        "scope": "geometry access only; not BF matching speed or pose accuracy", "counted": False})
    model = pycolmap.Reconstruction(model_path)
    accesses = trace["access_image_ids"]
    if trace["query_count"] != 24 or len(localization["rows"]) != 24:
        raise ValueError("The original24 localization denominator changed")
    records, baseline_values = [], {}
    for run in range(3):
        timings = {}
        for mode in (["baseline", "index"] if run % 2 == 0 else ["index", "baseline"]):
            table = FrozenObservationIndex(model) if mode == "index" else None
            values = []
            started = time.perf_counter()
            for image_id in accesses:
                if table is None:
                    points = model.images[image_id].points2D
                    pixels = np.asarray([p.xy for p in points], dtype=np.float64).reshape(-1, 2)
                    ids = np.asarray([int(p.point3D_id) for p in points], dtype=np.uint64)
                    valid = np.asarray([p.has_point3D() for p in points], dtype=bool)
                else:
                    pixels, ids, valid = table.get(model, image_id)
                values.append((pixels, ids, valid))
            timings[mode] = time.perf_counter()-started
            if mode == "baseline":
                baseline_values[run] = values
            else:
                baseline_values[(run, "index")] = values
                table.verify_unchanged()
                timings["index_receipt"] = table.receipt()
        timings["all_geometry_values_exact"] = all(
            all(np.array_equal(x, y) for x, y in zip(a, b))
            for a, b in zip(baseline_values[run], baseline_values[(run, "index")]))
        records.append(timings)
        baseline_values.clear()
    baseline = statistics.median(x["baseline"] for x in records)
    candidate = statistics.median(x["index"] for x in records)
    reduction = 1-candidate/baseline
    checks = {"three_cold_pairs": len(records) == 3, "exact_geometry": all(x["all_geometry_values_exact"] for x in records),
              "improves10percent": reduction >= .1, "sources_unchanged": identities == [data_identity(p) for p in sources]}
    result = {"status": "profile_passed_real_matching_pending" if all(checks.values()) else "profile_failed",
        "checks": checks, "pairs": records, "median_seconds": {"baseline": baseline, "index": candidate},
        "reduction": reduction, "actual_access_count": len(accesses), "counted": False,
        "boundary": "Baseline materializes full IDs for an independent array oracle; actual match stage must be separately timed before accepting speed"}
    atomic_json(args.output/"RESULT.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
