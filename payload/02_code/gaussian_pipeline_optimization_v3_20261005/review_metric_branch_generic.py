"""Review saved exact branch oracles against the original fallback contract."""

import argparse
import json
from pathlib import Path
import time

STARTED = time.perf_counter()
from check_metric_branch_dag import LABELS, close_geometry, read
import pycolmap
from runtime import atomic_json, data_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    original = read(args.input / "RESULT.json")
    if original["status"] != "required_generic_guard_failed" or len(original["completed"]) != 2:
        raise AssertionError("The saved two-fixture checker failure changed")
    cases = []
    for number in range(2):
        folder = args.input / ("fixture_" + str(number))
        left, right = read(folder / "serial/RESULT.json"), read(folder / "dag/RESULT.json")
        checks = {"status_equal": left["status"] == right["status"] == "metric_interface_verified",
            "native_guard_equal": left["development_guard_passed"] == right["development_guard_passed"],
            "native_delta_equal": left["common_observation_p95_delta_px"] == right["common_observation_p95_delta_px"],
            "native_selection_equal": left["selected"] == right["selected"],
            "original_native_select_rule": right["selected"] == (LABELS[1] if right["common_observation_p95_delta_px"] <= .05 else LABELS[0])}
        branches = {}
        for label in LABELS:
            control, candidate = folder / "serial", folder / "dag/branches" / label
            names = {"cameras.bin", "images.bin", "points3D.bin", "frames.bin", "rigs.bin"}
            a, b = control / (label+"_INITIAL_MODEL"), candidate / (label+"_INITIAL_MODEL")
            checks[label+"_complete_initial"] = ({p.name for p in a.iterdir()} == names
                and {p.name for p in b.iterdir()} == names and all((a/n).read_bytes() == (b/n).read_bytes() for n in names))
            checks[label+"_complete_effective_options"] = read(control / (label+"_EFFECTIVE_OPTIONS.json")) == read(candidate / (label+"_EFFECTIVE_OPTIONS.json"))
            equivalent, exact, maximum = close_geometry(pycolmap.Reconstruction(left["results"][label]["model"]),
                                                       pycolmap.Reconstruction(right["results"][label]["model"]))
            checks[label+"_complete_final_geometry"] = equivalent and all(exact.values())
            branches[label] = {"maximum_absolute_difference": maximum, "all_final_fields_bit_exact": all(exact.values())}
        close = read(folder / "dag/DAG_RESOURCE_CLOSE.json")
        checks["normal_owned_close"] = not close["alive_owned_processes"] and close["cancellation_kills"] == 0
        cases.append({"fixture": number, "checks": checks, "branches": branches,
                      "native_weak_guard_passed": right["development_guard_passed"], "selected": right["selected"]})
    passed = all(all(case["checks"].values()) for case in cases)
    atomic_json(args.output / "RESULT.json", {"status": "saved_generic_native_interface_passed" if passed else "saved_generic_review_failed",
        "cases": cases, "algorithm_source_modified": False, "native_solver_rerun": False,
        "checker_issue": "Both algorithms selected the original allowed visual fallback; checker incorrectly required weak admission for every synthetic input",
        "original_failed_checker_receipt_preserved": str(args.input / "RESULT.json"),
        "original_source_snapshots_still_present": read(args.input / "CONTRACT.json")["sources"],
        "review_source": data_identity(Path(__file__)), "elapsed_seconds": time.perf_counter()-STARTED, "counted": False})
    if not passed:
        raise AssertionError("Saved complete original interface review failed")


if __name__ == "__main__":
    main()
