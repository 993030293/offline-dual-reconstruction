"""Compare every scientific field of a complete support screening pair."""

import argparse
import json
from pathlib import Path
import time

from machine_json import exact_json_values
from check_grid_support import ROOT
from runtime import atomic_json


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", type=Path, required=True)
    args = parser.parse_args()
    args.pair.resolve().relative_to(ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/sg001")
    results = {mode: json.loads((args.pair/mode/"RESULT.json").read_text(encoding="utf-8"))
               for mode in ("native", "segmented")}
    checks = {}
    for key in ("rows", "sources", "inputs", "model_before", "model_after"):
        exact_json_values(results["native"][key], results["segmented"][key])
        checks[key + "_exact"] = True
    checks["both_original_guards_passed"] = all(all(result["checks"].values()) for result in results.values())
    result = {"status": "complete1016_screening_pair_exact" if all(checks.values()) else "failed_screening_pair",
              "checks": checks, "all1016_and_five_complete_fields": True,
              "working_seconds": {mode: value["working_support_seconds"] for mode, value in results.items()},
              "complete_phase_seconds": {mode: value["complete_phase_seconds"] for mode, value in results.items()},
              "single_pair_not_formal_acceptance": True, "new_optimization_count": 0,
              "comparison_seconds": time.perf_counter()-started}
    atomic_json(args.pair/"COMPARISON.json", result)
    print(json.dumps(result))
    if not all(checks.values()):
        raise AssertionError("The complete screening pair did not preserve its original scientific fields")


if __name__ == "__main__":
    main()
