"""Close a negative complete screen with current source and actual process evidence."""

import argparse
import json
from pathlib import Path
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).parent
OWN = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/gsf001/bootstrap_dag_001"
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity
from gpu_query_node import linux_identity_check, remaining


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def walk(value):
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from walk(item)


def main():
    began = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.relative_to(OWN.resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    own = {"pid": psutil.Process().pid, "creation_time": psutil.Process().create_time()}
    paths = [OWN / "g0/RESULT.json", OWN / "c1/RESULT.json", OWN / "c1/t0/INVARIANTS.json",
             OWN / "GENERIC_LEASE_001.json", OWN / "FIRST_PAIR_LEASE_001.json"]
    for name in ("cpu", "gpu"):
        leaf = OWN / "c1/t0" / name
        paths.extend(leaf / filename for filename in ("RESULT.json", "BOOTSTRAP.json", "BOOTSTRAP_CLOSE.json", "BINDING_RESTORATION.json"))
        paths.append(OWN / "c1/t0" / ("n_" + name) / "NODE.json")
    paths.append(OWN / "c1/t0/gpu/gpu_feature_batch/RESULT.json")
    values = [read(path) for path in paths]
    win, identities, linux = {}, {}, {}
    for value in values:
        for row in walk(value):
            if "pid" in row and "creation_time" in row and row.get("scope") != "WSL":
                win[(row["pid"], row["creation_time"])] = {key: row[key] for key in ("pid", "creation_time")}
            for prefix in ("guard", "child"):
                if prefix + "_pid" in row and prefix + "_creation_time" in row:
                    item = {"pid": row[prefix + "_pid"], "creation_time": row[prefix + "_creation_time"]}
                    win[(item["pid"], item["creation_time"])] = item
            if all(key in row for key in ("path", "bytes", "sha256")):
                item = {key: row[key] for key in ("path", "bytes", "sha256")}
                previous = identities.setdefault(item["path"], item)
                if previous != item:
                    raise ValueError("A saved source or input differs between complete branches")
    for name in ("cpu", "gpu"):
        node = read(OWN / "c1/t0" / ("n_" + name) / "NODE.json")
        for row in node["Linux_resource_receipts"]:
            item = {key: row[key] for key in ("pid", "creation_time")}
            linux[(item["pid"], item["creation_time"])] = item
            win.pop((item["pid"], item["creation_time"]), None)
    for item in (data_identity(Path(__file__)),):
        identities[item["path"]] = item
    live_windows = remaining(list(win.values()))
    live_linux = [linux_identity_check(item) for item in linux.values()]
    source_matches = list(identities.values()) == [data_identity(Path(item["path"])) for item in identities.values()]
    snapshot_count = 0
    campaign = read(OWN / "c1/RESULT.json")
    for number, item in enumerate(campaign["sources"]):
        saved = OWN / "c1/source_snapshot" / (str(number) + "_" + Path(item["path"]).name)
        identity = data_identity(saved)
        if identity["bytes"] != item["bytes"] or identity["sha256"] != item["sha256"]:
            raise ValueError("A complete campaign executed source snapshot differs")
        snapshot_count += 1
    for name in ("cpu", "gpu"):
        leaf = OWN / "c1/t0" / name
        for number, item in enumerate(read(leaf / "BOOTSTRAP.json")["sources"]):
            identity = data_identity(leaf / "bootstrap_source_snapshot" / (str(number) + "_" + Path(item["path"]).name))
            if identity["bytes"] != item["bytes"] or identity["sha256"] != item["sha256"]:
                raise ValueError("An actual bootstrap source snapshot differs")
            snapshot_count += 1
    generic, invariants = read(OWN / "g0/RESULT.json"), read(OWN / "c1/t0/INVARIANTS.json")
    pair = campaign["pairs"][0]
    leases = [read(OWN / name) for name in ("GENERIC_LEASE_001.json", "FIRST_PAIR_LEASE_001.json")]
    checks = {"only_one_preregistered_fresh_complete_pair_not_replaced": len(campaign["pairs"]) == 1
            and pair["order"] == ["cpu_process", "gpu"] and len(campaign["runs"]) == 2,
        "all12_scheduling_generic_guards": len(generic["checks"]) == 12 and all(generic["checks"].values()),
        "all_original_complete_scientific_and_format_guards": invariants["all_protected"] and all(pair["checks"].values()),
        "negative_original_complete_cost_stop_preserved": campaign["status"] == "closed_negative_first_raw_complete_cost_screen"
            and pair["raw_complete_reduction"] < .05,
        "all_current_saved_source_and_input_identities": source_matches,
        "all_saved_executed_source_snapshots": snapshot_count == 19,
        "all_saved_actual_Windows_instances_gone": not live_windows,
        "all_saved_actual_Linux_instances_gone": not any(item["alive"] for item in live_linux),
        "both_complete_previous_leases_released": all(item["returncode"] == 0 and item.get("lease_released_utc") for item in leases),
        "no_acceptance_count_or_formal_quality_claim": campaign["new_optimization_count"] == 0
            and campaign["quality_formal_external_recovery"] == "not_run_after_cost_screen_negative"}
    if not all(checks.values()):
        raise AssertionError("A current source, owned process or negative scope guard failed")
    raw_sum = sum(pair["raw_complete_seconds"].values())
    first_campaign_seconds = leases[1]["child_elapsed_seconds"]
    root_fee = max(0., first_campaign_seconds - raw_sum)
    result = {"status": "negative_complete_screen_current_sources_and_processes_closed",
        "review_kind": "separate_root_self_review_no_subagents", "independent_review": False,
        "checks": checks, "new_optimization_count": 0, "parent_additional_accepted_count": 22,
        "raw_fresh_complete_seconds": pair["raw_complete_seconds"], "raw_complete_reduction": pair["raw_complete_reduction"],
        "old_negative_family_measured_fees_seconds": 75.030935,
        "new_generic_measured_fee_seconds": leases[0]["child_elapsed_seconds"],
        "fresh_campaign_measured_seconds": first_campaign_seconds,
        "fresh_campaign_outside_two_complete_side_envelopes_seconds": root_fee,
        "measured_family_exposure_before_this_review_seconds": 75.030935 + sum(item["child_elapsed_seconds"] for item in leases),
        "no_formal_three_pair_amortized_speed_claim": True,
        "current_source_input_identities": len(identities), "executed_source_snapshots": snapshot_count,
        "actual_Windows_instances_checked": len(win), "actual_Linux_instances_checked": len(linux),
        "actual_live_Windows": live_windows, "actual_Linux": live_linux, "own_process": own,
        "receipt_inputs": list(map(str, paths)), "sources": list(identities.values()),
        "formal_three_pairs_quality_three_seeds_actual_GPU_cancel_recovery_external_training_GT_preview": "not_run_after_first_cost_negative",
        "production_modified": False, "review_before_last_publication_seconds": time.perf_counter() - began,
        "review_outer_lease_and_own_exit_pending_separate_observation": True}
    atomic_json(args.output / "RESULT.json", result)
    print(json.dumps({key: result[key] for key in ("status", "checks", "actual_Windows_instances_checked", "actual_Linux_instances_checked", "current_source_input_identities")}), flush=True)


if __name__ == "__main__":
    main()
