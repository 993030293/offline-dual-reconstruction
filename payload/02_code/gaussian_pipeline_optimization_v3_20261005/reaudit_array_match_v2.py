"""Recheck saved complete pairs under the authorized V2 engineering policy."""

import datetime
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

STARTED = time.perf_counter()
ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).parent
sys.path[:0] = [str(CODE), str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"),
    str(ROOT / "02_code/gaussian_pipeline_optimization_v2_20261005")]

import cv2
import psutil

from array_match_geometry import array_match_geometry
from check_array_match_geometry import prepare, prepared_factory
from check_parallel_neighbor_geometry import fixture
from observation_index import indexed_match
from run_array_match_campaign import compare_pair
from runtime import atomic_json, data_identity

OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/am001"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    output = OUT / "v2r0"
    output.mkdir(exist_ok=False)
    campaign = read(OUT / "c0/RESULT.json")
    closure = read(OUT / "NEGATIVE_CLOSURE_001.json")
    generic = read(OUT / "g0/RESULT.json")
    checks = {
        "V1_negative_preserved": campaign["status"] == "failed_verification"
            and not campaign["checks"]["complete_phase_reduces10percent"]
            and closure["new_optimization_count"] == 0,
        "all180_generic_and16_original_guards": generic["status"] == "passed"
            and len(generic["rows"]) == 180 and len(generic["checks"]) == 16
            and all(generic["checks"].values()),
        "three_fresh_original_interleaved_orders": campaign["orders"] == [
            ["incumbent", "array"], ["array", "incumbent"], ["incumbent", "array"]],
    }
    files, windows, linux = {}, {}, {}
    for item in [*campaign["sources"], *generic["sources"]]:
        files[item["path"]] = item
    comparisons = []
    for trial in range(3):
        folder = OUT / f"c0/t{trial}"
        protected = compare_pair(folder / "b", folder / "c")
        comparisons.append(protected)
        checks[f"t{trial}_actual_full_outputs_exact_now"] = protected["all_protected"]
        for label in ("b", "c"):
            node = read(folder / f"n{label}/NODE.json")
            side = read(folder / f"{label}/RESULT.json")
            contract = read(folder / f"{label}/CONTRACT.json")
            bindings = read(folder / f"{label}/BINDING_RESTORATION.json")
            checks[f"t{trial}_{label}_all24_48_cold_native_snapshot_close"] = (
                all(side["checks"].values()) and contract["cold_private_caches"]
                and bindings["worker_restored"] and bindings["argv_restored"]
                and node["status"] == "completed" and node["final_source_unchanged"]
                and not node["live_Windows_after_wait"]
                and not any(item["alive"] for item in node["actual_Linux_after_wait"]))
            for item in [*contract["sources"], *contract["inputs"]]:
                if item["path"] in files and files[item["path"]] != item:
                    raise ValueError("A saved source or input has inconsistent identities")
                files[item["path"]] = item
            for item in [node["owner"], *node["guard"]["owned_process_identities"]]:
                windows[(item["pid"], item["creation_time"])] = item
            for item in node["Linux_resource_receipts"]:
                linux[(item["pid"], item["creation_time"])] = {
                    "pid": item["pid"], "creation_time": item["creation_time"]}
    leases = [read(OUT / name) for name in ("GENERIC_LEASE_001.json", "CAMPAIGN_LEASE_001.json")]
    checks["both_original_leases_released"] = all(
        item["returncode"] == 0 and item.get("lease_released_utc") for item in leases)
    for lease in leases:
        for prefix in ("guard", "child"):
            item = {"pid": lease[prefix + "_pid"], "creation_time": lease[prefix + "_creation_time"]}
            windows[(item["pid"], item["creation_time"])] = item
    checks["all_original_source_and_input_bytes_current"] = all(
        data_identity(item["path"]) == item for item in files.values())
    live = []
    for item in windows.values():
        try:
            if psutil.Process(item["pid"]).create_time() == item["creation_time"]:
                live.append(item)
        except psutil.NoSuchProcess:
            pass
    checks["all46_owned_Windows_instances_gone_now"] = len(windows) == 46 and not live
    code = "import json,psutil; rows=json.loads(" + repr(json.dumps(list(linux.values()))) + "); live=[]\n"
    code += "for i in rows:\n try:\n  p=psutil.Process(i['pid'])\n  if p.create_time()==i['creation_time'] and p.status()!=psutil.STATUS_ZOMBIE: live.append(i)\n except psutil.NoSuchProcess: pass\nprint(json.dumps(live))"
    live_linux = json.loads(subprocess.check_output(["wsl", "-d", "E26-C2-LIV-20260907-134802", "--",
        "/opt/e26_thin_surface_v1/venv/bin/python", "-B", "-c", code], text=True, timeout=30))
    checks["all6_owned_Linux_instances_gone_now"] = len(linux) == 6 and not live_linux

    import native_observation_index
    hook, fit = native_observation_index.indexed_match, cv2.findFundamentalMat
    cancellation_rows = []
    try:
        values = fixture(89, 1., 1., "irregular", 4)
        for kind in (KeyboardInterrupt, SystemExit):
            calls, errors = [], []
            payload = prepare(values)
            def cancel(*args):
                calls.append(1)
                raise kind("owned_array_cancel")
            cv2.findFundamentalMat = cancel
            for candidate in (False, True):
                try:
                    with prepared_factory(payload):
                        if candidate:
                            with array_match_geometry() as trace:
                                native_observation_index.indexed_match(*values)
                        else:
                            indexed_match(*values)
                except (KeyboardInterrupt, SystemExit) as error:
                    errors.append((type(error).__name__, str(error)))
            cancellation_rows.append({"kind": kind.__name__, "same_exception_no_retry":
                len(calls) == 2 and len(errors) == 2 and errors[0] == errors[1],
                "hook_restored": native_observation_index.indexed_match is hook and trace["binding_restored"]})
    finally:
        cv2.findFundamentalMat = fit
    checks["both_native_cancel_exceptions_no_retry_and_private_hook_restored"] = all(
        row["same_exception_no_retry"] and row["hook_restored"] for row in cancellation_rows)
    checks["full_original_end_checks_and_hook_errors_precede_any_reuse"] = all(closure["checks"].values())

    original_compare_fee = sum(pair["complete_compare_seconds"] for pair in campaign["pairs"])
    original_wrapper_fee = leases[1]["child_elapsed_seconds"] - sum(
        sum(pair["raw_whole_seconds"].values()) for pair in campaign["pairs"]) - original_compare_fee
    if original_wrapper_fee < 0:
        raise ValueError("The complete campaign wrapper fee cannot be negative")
    first_fee = closure["all_first_fees_with_final_close_seconds"] + original_wrapper_fee
    result = {"status": "V2_reaudit_verified_cost_and_outer_close_pending" if all(checks.values()) else "failed_verification",
        "checks": checks, "source": data_identity(Path(__file__)), "comparisons": comparisons,
        "cancellation_rows": cancellation_rows, "original_campaign_wrapper_fee_not_previously_allocated": original_wrapper_fee,
        "complete_prior_first_fees_seconds": first_fee, "raw_pairs": campaign["pairs"],
        "source_input_identities_checked": len(files), "owned_Windows_checked": len(windows),
        "owned_Linux_checked": len(linux), "actual_live_Windows": live, "actual_live_Linux": live_linux,
        "review_seconds_inside_script": time.perf_counter() - STARTED,
        "owner": {"pid": psutil.Process().pid, "creation_time": psutil.Process().create_time()},
        "review_kind": "root_self_review_no_subagents", "new_optimization_count": 0,
        "original_negative_closure": str(OUT / "NEGATIVE_CLOSURE_001.json"),
        "policy": str(ROOT / "research/2026-10-05-gaussian-optimization-round3-50/ENGINEERING_ACCEPTANCE_POLICY_V2_20261006.md"),
        "external_full_object_GT_training_preview": "unrun_not_claimed",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    atomic_json(output / "REVIEW.json", result)
    print(json.dumps({key: result[key] for key in ("status", "checks", "source_input_identities_checked",
        "complete_prior_first_fees_seconds", "review_seconds_inside_script")}), flush=True)
    if not all(checks.values()):
        raise AssertionError("A V2 source, output or recovery protection failed")


if __name__ == "__main__":
    main()
