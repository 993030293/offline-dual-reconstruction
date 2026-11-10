"""Audit frozen complete quality consumers, source lineage, measured fees and resource closure."""

import argparse
import json
from pathlib import Path
import statistics
import time

import psutil

ROOT = Path(__file__).resolve().parents[2]
OWN = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/native_quality_workspace_001"
DOC = ROOT / "research/2026-10-05-gaussian-optimization-round3-50"
import sys
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def gone(row):
    try:
        return abs(psutil.Process(row["pid"]).create_time() - row["creation_time"]) >= .001
    except psutil.NoSuchProcess:
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lease", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(OWN)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    current = psutil.Process()
    ancestry = [{"pid": process.pid, "creation_time": process.create_time()}
                for process in [current] + current.parents()]
    atomic_json(output / "EXECUTION.json", {"pid": current.pid, "creation_time": current.create_time(), "actual_self_and_parents": ancestry})
    (output / "REVIEWER_SOURCE.py").write_bytes(Path(__file__).read_bytes())
    state = {"status": "review_running", "checks": {}, "review_kind": "separate_root_self_review_no_subagents",
             "sources_and_inputs": [], "snapshots": [], "process_instances": [], "leases": [], "counted": False}

    def require(name, condition):
        state["checks"][name] = bool(condition)
        if not condition:
            raise AssertionError(name)

    try:
        owner = load(args.lease)
        require("review_owns_actual_guard_child_or_parent_launcher", owner["status"] == "running" and owner[
            "owner"] == "root_R141_complete_cost_source_resource_review_001" and any(row["pid"] == owner["child_pid"]
            and abs(row["creation_time"] - owner["child_creation_time"]) < .001 for row in ancestry))
        generic = load(OWN / "g0/RESULT.json")
        pilot = load(OWN / "p0/RESULT.json")
        formal = load(OWN / "c1/RESULT.json")
        protocol = load(OWN / "c1/PROTOCOL.json")
        supplement = load(OWN / "g1/RESULT.json")
        require("generic8_original37_layout6_unsupported_four_threads_and_closure", len(generic["checks"]) == 8 and all(generic["checks"].values()))
        require("pilot_all_three_original128_RAW_cold_pairs_complete_and_positive", len(pilot["pairs"]) == 3 and all(all(row["checks"].values())
            for row in pilot["pairs"]) and pilot["reduction"] >= .05 and pilot["formal_pairs_executed"] == 0)
        require("three_original_fresh_pairs_same_frozen_source_and_schedule", len(formal["pairs"]) == 3 and len(formal["runs"]) == 6
            and [row["order"] for row in formal["pairs"]] == protocol["schedule"] == [["native", "workspace"], ["workspace", "native"], ["native", "workspace"]]
            and all(all(row["checks"].values()) for row in formal["pairs"]) and formal["sources"] == protocol["sources"])
        require("original5percent_each_whole1point1_and_full_denominators_preserved", protocol["minimum_complete_phase_reduction"] == .05
            and protocol["each_whole_ratio"] == 1.1 and protocol["whole_median_nonregression"] and
            (protocol["raw"], protocol["eligible"], protocol["selected"], protocol["pairs"]) == (7338, 6797, 1002, 11583))
        require("supplement33_errors23_partial128_recovery360_external_full_quality", len(supplement["checks"]) == 33
            and all(supplement["checks"].values()) and len(supplement["failure_recovery_pairs"]) == 3
            and len(supplement["real_directory_errors"]) == 2 and len(supplement["external_pairs"]) == 3)
        source_specs = ((OWN / "g0", generic["sources"]), (OWN / "p0", pilot["sources"]),
                        (OWN / "c1", protocol["sources"]), (OWN / "g1", supplement["sources"]))
        for folder, rows in source_specs:
            require(folder.name + "_all_current_source_identities", rows == [data_identity(Path(row["path"])) for row in rows])
            state["sources_and_inputs"].extend(rows)
            files = sorted((folder / "source_snapshot").iterdir())
            require(folder.name + "_all_source_snapshot_count", len(files) == len(rows))
            for snapshot in files:
                row = rows[int(snapshot.name.split("_", 1)[0])]
                actual = data_identity(snapshot)
                require(folder.name + "_snapshot_" + str(len(state["snapshots"])), all(actual[key] == row[key] for key in ("bytes", "sha256")))
                state["snapshots"].append(str(snapshot))
        require("all360_current_external_full_image_identities", supplement["input_images"] == [data_identity(Path(row["path"]))
            for row in supplement["input_images"]] and len(supplement["input_images"]) == 360)
        state["sources_and_inputs"].extend(supplement["input_images"])
        raw = Path(supplement["original_raw_path"]).stat()
        require("original_RAW_size_mtime_and_128_role_ids_unchanged", (raw.st_size, raw.st_mtime_ns) == (
            supplement["original_raw_size"], supplement["original_raw_mtime_ns"]) and supplement["original_selected128_ids"] == load(OWN / "p0/CONTRACT.json")["selected_ids"])
        require("pilot_and_supplement_all_native_aliases_threads_and_planes_closed", load(OWN / "p0/RESOURCE_CLOSE.json")["OpenCV_threads_restored"]
            and all(load(OWN / "g1/RESOURCE_CLOSE.json")[name] for name in ("OpenCV_threads_restored", "original_aliases_restored", "no_new_worker_alive")))
        costs = []
        for name in ("GENERIC_LEASE_001.json", "PILOT_LEASE_001.json", "FIRST_PAIR_LEASE_001.json",
                     "REMAINING_PAIRS_LEASE_001.json", "SUPPLEMENT_LEASE_001.json"):
            lease = load(OWN / name)
            require(name + "_terminal_success_release", lease["status"] == "child_exited" and lease["returncode"] == 0
                and bool(lease["lease_released_utc"]) and not lease["lock_wait_in_algorithm_cost"])
            costs.append(lease["child_elapsed_seconds"])
            state["leases"].append({"path": str(OWN / name), "actual_child_seconds": lease["child_elapsed_seconds"]})
            for prefix in ("guard", "child"):
                state["process_instances"].append({"pid": lease[prefix + "_pid"], "creation_time": lease[prefix + "_creation_time"]})
        require("original_guarded_campaigns_did_not_overlap", all(load(OWN / left)["lease_released_utc"] <= load(OWN / right)["created_utc"]
            for left, right in zip(("GENERIC_LEASE_001.json", "PILOT_LEASE_001.json", "FIRST_PAIR_LEASE_001.json", "REMAINING_PAIRS_LEASE_001.json"),
                                  ("PILOT_LEASE_001.json", "FIRST_PAIR_LEASE_001.json", "REMAINING_PAIRS_LEASE_001.json", "SUPPLEMENT_LEASE_001.json"))))
        peaks = {mode: [] for mode in ("native", "workspace")}
        files = 0
        owned_bytes = 0
        for path in OWN.rglob("*"):
            if path.is_file():
                files += 1
                owned_bytes += path.stat().st_size
        require("independent_output20GiB_and_current_free20GiB_budget", owned_bytes <= 20 * 1024**3 and psutil.disk_usage(str(OWN)).free >= 20 * 1024**3)
        for run in formal["runs"]:
            require("complete_run_" + str(len(state["process_instances"])) + "_returned_and_recorded_instance_closed", run["status"] == "completed"
                and run["returncode"] == 0 and gone(run))
            state["process_instances"].append({"pid": run["pid"], "creation_time": run["creation_time"]})
            process = load(OWN / "c1" / ("t" + str(run["trial"])) / (run["mode"] + "_PROCESS.json"))
            require("bounded_consumer_" + str(run["trial"]) + "_" + run["mode"] + "_original_time_RSS_exit", process["status"] == "completed"
                and process["returncode"] == 0 and process["maximum_seconds"] == 600 and process["elapsed_seconds"] <= 600
                and process["peak_tree_RSS_bytes"] <= process["maximum_RSS_bytes"] == 8 * 1024**3)
            state["process_instances"].extend(process["owned_process_identities"])
            peaks[run["mode"]].append(process["peak_tree_RSS_bytes"])
        bounded = load(OWN / "g1_PROCESS.json")
        require("bounded_supplement_original_time_RSS_exit", bounded["status"] == "completed" and bounded["returncode"] == 0
            and bounded["elapsed_seconds"] <= 600 and bounded["peak_tree_RSS_bytes"] <= 8 * 1024**3)
        state["process_instances"].extend(bounded["owned_process_identities"])
        for folder in ("g1",):
            row = load(OWN / folder / "EXECUTION.json")
            state["process_instances"].append({"pid": row["pid"], "creation_time": row["creation_time"]})
        pilot_close = load(OWN / "p0/RESOURCE_CLOSE.json")
        state["process_instances"].append({"pid": pilot_close["actual_PID"], "creation_time": pilot_close["actual_creation_time"]})
        state["process_instances"] = list({(row["pid"], row["creation_time"]): row for row in state["process_instances"]}.values())
        require("all_registered_owned_Windows_instances_exited_by_creation_identity", all(gone(row) for row in state["process_instances"]))
        parent = load(DOC / "CURRENT_STATE.json")
        require("parent21_accepted_receipts_unique_candidate0_and_single_chat_scope", parent["additional_accepted_count"] == 21
            and len(set(parent["accepted_mechanisms"])) == len(set(parent["acceptance_receipts"])) == 21
            and parent["R141_new_optimization_count"] == 0 and not parent["production_modified"])
        # Complete leaf envelopes and actual comparison work are already charged in all six formal sides.
        leaf_seconds = sum(run["actual_complete_child_close_seconds"] for run in formal["runs"])
        comparison_seconds = sum(pair["comparison_seconds_charged_each_side"] for pair in formal["pairs"])
        shared = sum(costs) - leaf_seconds - comparison_seconds
        require("all_measured_shared_fees_positive_no_leaf_or_comparison_double_count", shared > 0
            and sum(invocation["measured_leaf_envelopes_seconds"] for invocation in formal["invocations"]) == leaf_seconds)
        adjusted = {name: {mode: value + shared / 6 for mode, value in formal["medians"][name].items()}
                    for name in ("phase_seconds", "whole_seconds")}
        reductions = {name: 1 - values["workspace"] / values["native"] for name, values in adjusted.items()}
        require("provisional_complete_phase5percent_and_whole_median_nonincrease", reductions["phase_seconds"] >= .05
            and reductions["whole_seconds"] >= 0 and all(pair["whole_seconds"]["workspace"] <= 1.1 * pair[
                "whole_seconds"]["native"] for pair in formal["pairs"]))
        state.update(status="complete_consumers_source_errors_external_cost_resource_self_review_passed_final_actual_review_fee_pending",
                     complete_fresh_pairs=3, owned_output_files=files, owned_output_bytes=owned_bytes,
                     peak_complete_tree_RSS_medians={mode: statistics.median(values) for mode, values in peaks.items()},
                     total_measured_campaign_envelopes_seconds=sum(costs), formal_leaf_envelopes_seconds=leaf_seconds,
                     actual_formal_comparison_seconds_already_charged= comparison_seconds,
                     known_shared_fees_before_review_publication_seconds=shared,
                     provisional_all_known_adjusted_medians=adjusted, provisional_reductions=reductions,
                     external_complete_medians=supplement["external_complete_medians"],
                     external_complete_reduction=supplement["external_complete_reduction"],
                     candidate_buffers_bound_by_actual_four_threads_no_object_shape_prior=True,
                     formal_cold_is_score_cache_fresh_not_OS_disk_flush=True,
                     not_run="production_integration_new_training_GT_unseen_object_whole_pipeline_or_new_preview",
                     review_elapsed_before_write_seconds=time.perf_counter() - started, accepted_count_contribution=0)
        atomic_json(output / "RESULT.json", state)
        print({"status": state["status"], "checks": len(state["checks"]), "owned_instances_exited": len(state["process_instances"]),
               "shared_fees_before_review_publication": shared, "provisional_reductions": reductions, "counted": False}, flush=True)
    except BaseException as error:
        state.update(status="failed", error_type=type(error).__name__, error=str(error), elapsed_seconds=time.perf_counter() - started)
        atomic_json(output / "FAILURE.json", state)
        raise


if __name__ == "__main__":
    main()
