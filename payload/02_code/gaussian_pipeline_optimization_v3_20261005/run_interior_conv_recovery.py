"""Validate real consumer faults and complete independent-directory recovery."""

import argparse
import json
from pathlib import Path
import shutil
import time

import psutil

import run_interior_conv_campaign as campaign


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-name", choices=("r0", "r1"), default="r0")
    args = parser.parse_args()
    output = campaign.OWN / args.output_name
    output.mkdir(parents=True, exist_ok=False)
    owner = psutil.Process()
    campaign.atomic_json(output / "PROCESS_IDENTITY.json", {"pid": owner.pid, "creation_time": owner.create_time()})
    paths, inputs = campaign.identity_set()
    paths.extend([Path(__file__), campaign.CODE / "fault_interior_conv_work.py"])
    sources, identities = [campaign.data_identity(p) for p in paths], [campaign.data_identity(p) for p in inputs]
    for ordinal, path in enumerate(paths):
        shutil.copyfile(path, output / (f"SOURCE_{ordinal:02d}_" + path.name))
    campaign.atomic_json(output / "CONTRACT.json", {"sources": sources, "inputs": identities,
        "cases": [["native", "error"], ["interior", "error"], ["interior", "cancel"]],
        "complete_fresh_both_modes_recovery": True, "new_optimization_count": 0})
    rows = []
    for mode, fault in (("native", "error"), ("interior", "error"), ("interior", "cancel")):
        folder = output / (mode + "_" + fault)
        campaign.prepare(folder)
        campaign.invoke(folder, mode, script="fault_interior_conv_work.py", extra=("--fault", fault), expected_code=86)
        result = campaign.read(folder / "FAULT_RESULT.json")
        if result["status"] != "expected_failure_cleanup_verified" or not all(result["checks"].values()):
            raise AssertionError("A real original consumer fault was not restored")
        rows.append({"mode": mode, "fault": fault, "folder": str(folder), "all_checks": True})
    for mode in ("native", "interior"):
        folder = output / (mode + "_recovered")
        campaign.prepare(folder)
        campaign.invoke(folder, mode)
        rows.append({"mode": mode, "fault": None, "folder": str(folder), "all_checks": True})
    comparison = campaign.compare(output / "native_recovered", output / "interior_recovered")
    original_comparison = campaign.compare(campaign.OWN / "c1/p0/native", output / "interior_recovered")
    checks = {"all_three_real_error_cancel_cleanup_cases": len(rows) == 5 and all(r["all_checks"] for r in rows),
              "fresh_full24_both_modes_recovery_exact": all(comparison["checks"].values()),
              "fresh_recovery_equals_original_complete_outputs": all(original_comparison["checks"].values()),
              "all_original_sources_unchanged": sources == [campaign.data_identity(p) for p in paths],
              "all_original_inputs_unchanged": identities == [campaign.data_identity(p) for p in inputs]}
    campaign.atomic_json(output / "RESULT.json", {"status": "complete_error_cancel_and_new_directory_recovery_verified" if all(checks.values()) else "failed",
        "checks": checks, "rows": rows, "comparison": comparison, "original_comparison": original_comparison,
        "sources": sources, "inputs": identities, "complete_seconds": time.perf_counter() - started, "new_optimization_count": 0})
    if not all(checks.values()):
        raise AssertionError("An original complete consumer recovery or input protection failed")
    print(json.dumps({"status": "complete_error_cancel_and_new_directory_recovery_verified", "checks": checks}), flush=True)


if __name__ == "__main__":
    main()

