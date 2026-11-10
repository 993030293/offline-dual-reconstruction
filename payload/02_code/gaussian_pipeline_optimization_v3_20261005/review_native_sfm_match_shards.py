"""Run a separate root audit of complete native matching and all retained costs."""

import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
import traceback

import native_sfm_match_shards as native
from check_native_sfm_match_shards import closed

OUT = native.OUT
DOC = native.ROOT / "research/2026-10-05-gaussian-optimization-round3-50"
READER_FEES = [
    ("73211f_failed_source_path", .1942635), ("924dbe_CLI_version_and_failed_policy_path", 4.0232717),
    ("408970_failed_source_escape", .1068682), ("b56fcf_corrected_source", .2116062),
    ("5d971c_source", .2431944), ("97cfd9_source", .2989299), ("7e52cb_source", .2695236),
    ("fb1d14_source", .3235299), ("475e32_source", .2545487), ("a086f8_source", .1592316),
    ("38a17b_source", .1892856), ("38a709_source", .244437), ("591653_hardware", 1.5100487),
    ("02e834_prior_birth_closure", .2539022), ("425f28_protocol_and_c1", .3786195),
    ("2474fc_c1_exact_and_closure", .2239439), ("543222_c2_exact_and_closure", .2384685),
    ("a0b444_source_and_failed_glob", .3011208), ("a572a9_lifecycle_source", .1716938),
    ("c61dd0_source_and_failed_public_path", .6421678), ("5dba41_board_text", .5114369),
    ("36e009_public_source", .3758519), ("6ed2ff_c3_public_and_disk", 1.4543753),
    ("9468a2_native_log_and_API", .3857473), ("925008_supplement_closure", .2964219),
    ("8ee5ac_full_recovery_closure", .3129839), ("08b1e4_current_text", .2013505),
    ("083c50_full_medians_and_feature_source", .2548259),
]


def history_rows(prefix):
    guard = native.read(OUT / (prefix + "_GUARD.json"))
    lease = native.read(OUT / (prefix + "_lease.json"))
    if guard["status"] != "completed" or guard["returncode"] != 0 or lease["returncode"] != 0 or not lease.get("lease_released_utc"):
        raise AssertionError("A prior actual native matching journal or lease is not terminal")
    rows = guard["owned_process_identities"] + [
        {"pid": lease["guard_pid"], "creation_time": lease["guard_creation_time"]},
        {"pid": lease["child_pid"], "creation_time": lease["child_creation_time"]}]
    if not all(closed(row) for row in rows):
        raise AssertionError("A prior matching process birth is still live")
    return guard, rows


def costs(extra_seconds=0.):
    guards = []
    for prefix in ("p0", "p1", "keys"):
        row = native.read(OUT.parent / (prefix + "_GUARD.json"))
        guards.append({"name": "009_" + prefix, "seconds": row["elapsed_seconds"], "returncode": row["returncode"]})
    for prefix in ("g0", "c1", "c2", "c3", "checks0", "r0"):
        row = native.read(OUT / (prefix + "_GUARD.json"))
        guards.append({"name": "R165_" + prefix, "seconds": row["elapsed_seconds"], "returncode": row["returncode"]})
    total = sum(row["seconds"] for row in guards) + sum(value for name, value in READER_FEES) + 15 + extra_seconds
    return {"actual_guard_costs": guards, "actual_known_source_readers_including_failures": [dict(name=name, seconds=value) for name, value in READER_FEES],
            "preregistered_metadata_allowance_seconds": 10, "preregistered_post_allowance_seconds": 5,
            "additional_actual_review_and_post_seconds": extra_seconds, "total_known_execution_plus_allowance_seconds": total,
            "shared_lease_wait_not_double_charged": True, "unmeasured_research_work_time_not_zero_or_total_ROI": True}


def scores(fee):
    pairs = []
    for prefix in ("c1", "c2", "c3"):
        value = native.read(OUT / prefix / "RESULT.json")
        b = next(row for row in value["entries"] if not row["candidate"])["complete_closed_features_to_ready_database_seconds"]
        c = next(row for row in value["entries"] if row["candidate"])["complete_closed_features_to_ready_database_seconds"]
        pairs.append({"name": prefix, "baseline_seconds": b, "candidate_seconds": c, "raw_whole_ratio": c / b,
                      "fee_adjusted_whole_ratio": (c + fee / 6) / (b + fee / 6)})
    b = statistics.median(row["baseline_seconds"] for row in pairs)
    c = statistics.median(row["candidate_seconds"] for row in pairs)
    return {"pairs": pairs, "baseline_median_seconds": b, "candidate_median_seconds": c,
            "raw_stage_reduction_of_medians": (b - c) / b,
            "raw_median_pair_reduction": 1 - statistics.median(row["raw_whole_ratio"] for row in pairs),
            "fee_per_side_seconds": fee / 6, "net_complete_stage_reduction": (b - c) / (b + fee / 6),
            "raw_whole_median_ratio": statistics.median(row["raw_whole_ratio"] for row in pairs),
            "fee_adjusted_whole_median_ratio": statistics.median(row["fee_adjusted_whole_ratio"] for row in pairs),
            "whole_endpoint_is_complete_matching_not_mapper_BA_or_training": True}


def main():
    started = time.perf_counter()
    output = OUT / "root_review0"
    if output.exists():
        raise FileExistsError("A fresh separate root review is required")
    output.mkdir()
    sources = [native.data_identity(Path(__file__)), native.data_identity(Path(native.__file__)),
               native.data_identity(Path(__file__).with_name("check_native_sfm_match_shards.py"))]
    (output / "sources").mkdir()
    for row in sources:
        shutil.copyfile(row["path"], output / "sources" / Path(row["path"]).name)
    try:
        identities = {}; births = []; normal = []; oracles = []
        for prefix in ("g0", "c1", "c2", "c3", "checks0", "r0"):
            guard, rows = history_rows(prefix); births.extend(rows)
            row = native.read(OUT / prefix / "RESULT.json")
            if not all(row["checks"].values()):
                raise AssertionError("A complete original matching or supplement check failed")
            for identity in row["sources"] + row["inputs"]:
                if identity["path"] in identities and identities[identity["path"]] != identity:
                    raise AssertionError("Frozen native matching scientific source versions differ")
                identities[identity["path"]] = identity
            if prefix != "checks0":
                normal.append(row)
                b = next(entry for entry in row["entries"] if not entry["candidate"])
                c = next(entry for entry in row["entries"] if entry["candidate"])
                oracles.append(native.compare_databases(b["database"], c["database"]))
                for entry in row["entries"]:
                    births.extend(native.read(Path(entry["database"]).parent / "NATIVE_PROCESSES.json"))
        for identity in identities.values():
            if native.data_identity(identity["path"]) != identity:
                raise AssertionError("A frozen native matching source or input changed")
        supplement = native.read(OUT / "checks0/RESULT.json")
        for row in supplement["components"].values():
            births.extend(row.get("closed_births", []))
        public_oracle = native.compare_databases(native.ROOT / "04_outputs/public_full_e26_scene_v3_20260929T141800Z/attempt02/01_visual_workspace/database.db", OUT / "checks0/public/left.db")
        for prefix in ("p0", "p1", "keys"):
            guard = native.read(OUT.parent / (prefix + "_GUARD.json"))
            births.extend(guard["owned_process_identities"])
        counter_proof = native.read(OUT.parent / "SOURCE_ORACLE_KEYS_001.json")
        cancelled = OUT / "checks0/shard_cancel"
        before_cancelled = {path.name: path.stat().st_size for path in cancelled.iterdir() if path.is_file()}
        with (output / "same_directory_rejection.log").open("x", encoding="utf-8") as log:
            rejection = subprocess.run([sys.executable, "-X", "utf8", "-B", str(Path(native.__file__)), str(cancelled), "AB"],
                cwd=native.ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=20, creationflags=subprocess.CREATE_NO_WINDOW)
        after_cancelled = {path.name: path.stat().st_size for path in cancelled.iterdir() if path.is_file()}
        snapshots = []
        for name in ("R165_NATIVE_MATCH_SHARDS_TASK.md", "R165_NATIVE_MATCH_SHARDS_RESULTS.md", "V6_NET_SPEED_POLICY.md", "SFM_MATCHING_COST_009_RESULTS.md", "NEXT_NATIVE_FEATURE_COST_010_BRIEF.md"):
            text = (DOC / name).read_text(encoding="utf-8-sig")
            (output / name).write_text(text, encoding="utf-8")
            snapshots.append({"path": str(DOC / name), "direct_text_lines": len(text.splitlines()), "copied_original_text": True})
        report = (DOC / "R165_NATIVE_MATCH_SHARDS_RESULTS.md").read_text(encoding="utf-8-sig")
        parent = native.read(DOC / "CURRENT_STATE.json")
        fee = costs(time.perf_counter() - started)
        measured = scores(fee["total_known_execution_plus_allowance_seconds"])
        gpu_rows = [row for prefix in ("g0", "c1", "c2", "c3", "r0") for side in ("b", "c") for row in native.read(OUT / prefix / side / "GPU_SAMPLES.json")]
        checks = {
            "all_five_normal_and_recovery_complete_typed_oracles_repeated_in_root": len(oracles) == 5 and all(row["complete_values_exact"] for row in oracles),
            "all1002_features_calibration_IDs_and11583_pairs_not_reduced": all(row["typed_complete_oracle"]["counts"]["images"] == 1002 and row["typed_complete_oracle"]["counts"]["matches"] == row["typed_complete_oracle"]["counts"]["two_view_geometries"] == 11583 for row in normal),
            "three_NEW_preregistered_formal_AB_BA_AB_orders_preserved": [native.read(OUT / name / "PROTOCOL.json")["order"] for name in ("c1", "c2", "c3")] == ["AB", "BA", "AB"],
            "generic_conflicts_NULL_BLOB_extra_columns_and_full_rollback_passed": all(supplement["components"]["generic"]["checks"].values()),
            "full_public360_all64620_native_columns_oracle_repeated": public_oracle["complete_values_exact"] and public_oracle["counts"]["matches"] == 64620,
            "both_real_native_errors_and_live_peer_drain_propagated": all(all(supplement["components"][name]["checks"].values()) for name in ("native_error", "shard_error")),
            "both_real_SIGINT_after_native_matching_became_active_drained": all(all(supplement["components"][name]["checks"].values()) for name in ("native_cancel", "shard_cancel")),
            "cancelled_same_directory_actually_rejected_without_changes": rejection.returncode != 0 and before_cancelled == after_cancelled and "fresh registered private matching" in (output / "same_directory_rejection.log").read_text(encoding="utf-8-sig"),
            "fresh_full_recovery_same_sources_inputs_and_denominator": all(native.read(OUT / "r0/RESULT.json")["checks"].values()),
            "all_previous_exact_native_guard_and_child_births_closed": all(closed(row) for row in births),
            "all_current_scientific_sources_inputs_and_execution_snapshots_preserved": all(native.data_identity(row["path"]) == row for row in sources) and all(Path(row["path"]).is_file() for row in identities.values()),
            "009_failed_raw_Counter_comparison_and_independent_correct_proof_preserved": native.read(OUT.parent / "p0_GUARD.json")["returncode"] != 0 and all(counter_proof["checks"].values()),
            "all_prior_guards_fees_failed_readers_plus_registered15_allowance_retained": len(fee["actual_guard_costs"]) == 9 and any(row["returncode"] != 0 for row in fee["actual_guard_costs"]) and fee["preregistered_metadata_allowance_seconds"] + fee["preregistered_post_allowance_seconds"] == 15,
            "V6_complete_stage_net_speed_at_least2percent_pending_actual_review_fee": measured["net_complete_stage_reduction"] >= .02,
            "raw_and_fee_adjusted_whole_medians_and_every_pair_within_original_guards": measured["raw_whole_median_ratio"] <= 1.01 and measured["fee_adjusted_whole_median_ratio"] <= 1.01 and all(row["raw_whole_ratio"] <= 1.1 and row["fee_adjusted_whole_ratio"] <= 1.1 for row in measured["pairs"]),
            "GPU_samples_and_all_prior_RSS_disk_guards_preserved": bool(gpu_rows) and all(row["usedMiB"] <= 12288 and row["freeMiB"] >= 2048 for row in gpu_rows) and all(native.read(OUT / (name + "_GUARD.json"))["peak_tree_RSS_bytes"] <= 4 * 1024 ** 3 for name in ("g0", "c1", "c2", "c3", "checks0", "r0")) and sum(path.stat().st_size for path in OUT.rglob("*") if path.is_file()) <= 8 * 1024 ** 3,
            "one_distinct_mechanism_and_parent_still35_until_actual_final_exit": parent["additional_accepted_count"] == 35 and len(parent["accepted_mechanisms"]) == 35 and parent["R165_new_optimization_count"] == 0,
            "report_text_matches_full_stage_boundaries_and_unrun_limits": all(value in report for value in ("14.913958300", "9.515632300", "36.196467%", "未部署生产流程", "未重新运行公开native matcher", "没有运行mapper", "923/1002", "净结果待最后费用回执")),
        }
        value = {"status": "separate_root_self_review_passed_pending_actual_exit0", "independent_agent_review": False,
                 "checks": checks, "complete_oracles": oracles, "public_oracle": public_oracle, "sources": sources,
                 "frozen_identities": list(identities.values()), "closed_previous_births": births, "direct_text_snapshots": snapshots,
                 "known_costs": fee, "scores_pending_actual_review_guard": measured,
                 "GPU_peak_whole_device_used_MiB_not_owned_peak": max(row["usedMiB"] for row in gpu_rows),
                 "same_directory_rejection_returncode": rejection.returncode, "root_count": 35, "new_optimization_count": 0}
        native.save(output / "REVIEW.json", value)
        print(json.dumps({"checks": checks, "provisional_net": measured["net_complete_stage_reduction"], "new_count": 0}), flush=True)
        if not all(checks.values()):
            raise AssertionError("The separate root native shard review failed")
    except BaseException as error:
        native.save(output / "FAILURE.json", {"error": repr(error), "traceback": traceback.format_exc(), "new_optimization_count": 0})
        raise


if __name__ == "__main__":
    main()
