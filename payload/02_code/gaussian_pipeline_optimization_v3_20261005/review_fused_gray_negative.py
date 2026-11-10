"""Self-review the negative complete RAW screen and close its owned resources."""

import argparse
import ast
import json
import msvcrt
from pathlib import Path
import statistics
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"
sys.path.insert(0, str(V1))
from runtime import atomic_json, data_identity

OWN = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/gm001"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def exited(pid, creation):
    try:
        process = psutil.Process(pid)
        return abs(process.create_time() - creation) > .000001 or not process.is_running()
    except psutil.NoSuchProcess:
        return True


def snapshot_source(path, identities):
    prefix, separator, name = path.name.partition("_")
    if separator and prefix.isdecimal():
        index = int(prefix)
        return [identities[index]] if index < len(identities) and Path(identities[index]["path"]).name == name else []
    return [row for row in identities if Path(row["path"]).name == path.name]


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(OWN)
    output.mkdir(parents=True, exist_ok=False)
    process = psutil.Process()
    atomic_json(output / "PROCESS_IDENTITY.json", {"scope": "Windows", "pid": process.pid, "creation_time": process.create_time()})
    generic, pilot = load(OWN / "g0/RESULT.json"), load(OWN / "p0/RESULT.json")
    checks = {"generic12_exact_quality_integer_and_closure_checks": len(generic["checks"]) == 12 and all(generic["checks"].values()),
              "all_three_original128_RAW_pairs_and_registered_checks": len(pilot["pairs"]) == 3 and all(all(pair["checks"].values()) for pair in pilot["pairs"]),
              "component_negative_same_preregistered5percent_no_formal_count": pilot["status"] == "component_screen_negative_below5percent" and pilot["reduction"] < .05 and pilot["formal_pairs_executed"] == pilot["accepted_count_contribution"] == 0}
    sources, snapshot_records = [], []
    for name in ("b0", "g0", "p0"):
        result = load(OWN / name / "RESULT.json")
        for identity in result["sources"]:
            current = data_identity(Path(identity["path"]))
            sources.append({"node": name, "path": identity["path"], "unchanged": current == identity})
        snapshot = OWN / name / "source_snapshot"
        for path in snapshot.iterdir():
            if name == "r0":
                continue
            candidates = snapshot_source(path, result["sources"])
            snapshot_records.append({"node": name, "name": path.name, "one_executed_source": len(candidates) == 1,
                "bytes_match_current_executed_source": len(candidates) == 1 and path.read_bytes() == Path(candidates[0]["path"]).read_bytes()})
    checks["all_executed_source_DLL_compiler_and_original_inputs_unchanged"] = all(row["unchanged"] for row in sources)
    checks["all_executed_text_snapshots_match_current_original_sources"] = all(row["one_executed_source"] and row["bytes_match_current_executed_source"] for row in snapshot_records)
    fixture = [{"path": "check_fused_gray_moments.py"}, {"path": "fused_gray_moments.py"}]
    checks["failed_suffix_collision_fixed_with_exact_index_and_filename"] = snapshot_source(Path("0_check_fused_gray_moments.py"), fixture) == [fixture[0]] and snapshot_source(Path("1_fused_gray_moments.py"), fixture) == [fixture[1]] and snapshot_source(Path("1_check_fused_gray_moments.py"), fixture) == []
    contract = load(OWN / "p0/CONTRACT.json")
    raw = Path(contract["original_raw_path"]).stat()
    checks["original_RAW_size_mtime_and128_membership_preserved"] = (raw.st_size, raw.st_mtime_ns) == (contract["original_raw_size"], contract["original_raw_mtime_ns"]) and len(contract["selected_ids"]) == 128
    all_cache_bytes, all_rows, traces = [], [], []
    for trial in range(3):
        folder = OWN / "p0" / ("t" + str(trial))
        left, right = folder / "native/quality", folder / "fused/quality"
        names = sorted(path.name for path in left.iterdir())
        all_cache_bytes.append(len(names) == 128 and names == sorted(path.name for path in right.iterdir()) and all(
            (left / name).read_bytes() == (right / name).read_bytes() for name in names))
        left_rows = load(folder / "native/SCORES.json")["original_full_metadata_and_quality"]
        right_rows = load(folder / "fused/SCORES.json")["original_full_metadata_and_quality"]
        all_rows.append(len(left_rows) == 128 and left_rows == right_rows and [row["camera_frame"] for row in left_rows] == contract["selected_ids"])
        for mode in ("native", "fused"):
            row = load(folder / mode / "TRACE.json")
            workspace = row["execution"]["workspace"]["workspace"]
            closed = row["execution"]["fused_builder_restored"] and row["execution"]["workspace"]["original_builder_restored"] and row["histogram"]["aliases_restored"] and workspace["closed"] and workspace["owned_plane_bytes_after_close"] == 0
            if mode == "fused":
                engine = row["execution"]["engine"]
                closed &= engine["DLL_unloaded"] and engine["active_calls"] == 0 and engine["calls"] == 128 and engine["owned_output_bytes_after_close"] == 0
            traces.append(closed and row["detail"]["submitted"] == 128 and row["detail"]["cache_hits"] == 0)
    checks["all_three128_raw_metadata_quality_and_persistent_cache_bytes_rechecked"] = all(all_cache_bytes) and all(all_rows)
    checks["all_six_original_denominators_aliases_pools_native_planes_and_DLL_closed"] = all(traces) and load(OWN / "p0/RESOURCE_CLOSE.json")["OpenCV_threads_restored"]
    medians = {mode: statistics.median(pair["modes"][mode]["complete_phase_seconds"] for pair in pilot["pairs"]) for mode in ("native", "fused")}
    checks["original_three_pair_medians_and_negative_reduction_recomputed"] = medians == pilot["medians"] and 1 - medians["fused"] / medians["native"] == pilot["reduction"]
    leases, instances = [], []
    for name in ("LEASE_B0.json", "LEASE_G0.json", "LEASE_P0.json", "LEASE_R0.json"):
        lease = load(OWN / name)
        leases.append(lease)
        for kind in ("guard", "child"):
            instances.append({"receipt": name, "kind": kind, "pid": lease[kind + "_pid"], "creation_time": lease[kind + "_creation_time"],
                "actual_owned_instance_exited": exited(lease[kind + "_pid"], lease[kind + "_creation_time"])})
    checks["all_eight_actual_owned_Windows_instances_and_four_prior_leases_closed"] = all(row["actual_owned_instance_exited"] for row in instances) and all(row["returncode"] == (1 if row["owner"] == "root_R144_review_001" else 0) and row.get("lease_released_utc") for row in leases)
    state = load(ROOT / "research/2026-10-05-gaussian-optimization-round3-50/CURRENT_STATE.json")
    receipts = []
    for path in state["acceptance_receipts"]:
        row = load(ROOT / path)
        counts = [row[key] for key in ("new_optimization_count", "additional_accepted_count", "accepted_count_contribution") if key in row]
        receipts.append({"path": path, "one_mechanism": bool(counts) and all(count == 1 for count in counts)})
    checks["current22_unique_existing_acceptance_receipts_preserved"] = state["additional_accepted_count"] == len(receipts) == len(set(state["acceptance_receipts"])) == len(set(state["accepted_mechanisms"])) == 22 and all(row["one_mechanism"] for row in receipts)
    files = ["fused_gray_moments.py", "build_native_gray_moments.py", "check_fused_gray_moments.py", "profile_fused_gray_moments.py", "review_fused_gray_negative.py"]
    for name in files:
        ast.parse(Path(__file__).with_name(name).read_text(encoding="utf-8"), filename=name)
    checks["all_five_new_python_sources_parse_without_execution"] = True
    checks["unrun_full7338_external_and_new_training_not_present"] = not any((OWN / name).exists() for name in ("c0", "c1", "e0", "train", "ACCEPTANCE.json"))
    receipt = {"status": "negative_component_self_review_passed" if all(checks.values()) else "failed_verification",
        "review_type": "separate_root_self_review_no_independent_agent", "checks": checks, "sources": sources,
        "executed_text_snapshot_records": snapshot_records, "actual_Windows_instances": instances,
        "all22_actual_acceptance_receipts": receipts, "component_medians": medians, "component_reduction": pilot["reduction"],
        "known_build_generic_and_component_lease_child_seconds": sum(row["child_elapsed_seconds"] for row in leases),
        "self_review_seconds_before_publication": time.perf_counter() - started,
        "not_run": ["wrong_ABI_injection", "full7338_original_entry", "formal_three_fresh_whole_pairs", "actual_RAW_failure_partial_cache_recovery", "public360_component", "new_training", "GT", "unseen_object_chain", "production_or_preview"],
        "accepted_count_contribution": 0, "parent_count": 22,
        "own_review_guard_child_and_OS_lock_require_post_exit_check": True}
    atomic_json(output / "RESULT.json", receipt)
    print({"status": receipt["status"], "checks": len(checks), "sources": len(sources), "snapshots": len(snapshot_records),
           "known_previous_child_seconds": receipt["known_build_generic_and_component_lease_child_seconds"], "parent_count": 22}, flush=True)
    if not all(checks.values()):
        raise AssertionError(json.dumps({name: value for name, value in checks.items() if not value}))


if __name__ == "__main__":
    main()
