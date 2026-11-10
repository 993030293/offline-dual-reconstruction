"""Recheck the full failed sorting consumer, source snapshots, and owned exits."""

import datetime
import json
from pathlib import Path
import re
import shutil
import sys

import psutil

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/lco001"
DOC = ROOT / "research/2026-10-05-gaussian-optimization-round3-50"
sys.path[:0] = [str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"),
    str(ROOT / "02_code/gaussian_pipeline_optimization_v2_20261005")]
from runtime import atomic_json, data_identity
from raw_initial_invariants import compare_initial_outputs


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    target = OUT / "NEGATIVE_V1_SELF_REVIEW_002.json"
    if target.exists():
        raise FileExistsError("The original negative review cannot be replaced")
    state, generic, pilot = read(OUT / "c0/RESULT.json"), read(OUT / "g0/RESULT.json"), read(OUT / "p0/RESULT.json")
    traces = read(OUT / "p0/PILOT_ORDER_ADAPTER.json")
    identities = {}
    for row in state["sources"] + generic["sources"] + [generic["library"]] + traces["sources"]:
        prior = identities.setdefault(row["path"], row)
        if prior != row:
            raise AssertionError("A current sorting source has two incompatible identities")
    checks = {
        "all_current_sources_and_compiled_library_match_executed_versions": all(data_identity(Path(row["path"])) == row for row in identities.values()),
        "all12_generic_conditions_and_no_acceptance": len(generic["checks"]) == 12 and all(generic["checks"].values()) and generic["new_optimization_count"] == 0,
        "three_same_process128_original_outputs_and_factory_guards": len(pilot["pairs"]) == 3 and all(all(pair["checks"].values()) for pair in pilot["pairs"])
            and len(traces["checks"]) == 6 and all(traces["checks"].values()),
        "one_only_new_full7338_pair_stopped_by_registered_cost": len(state["pairs"]) == 1 and state["status"] == "closed_negative_full_RAW_registered_cost_screen0"
            and state["net_complete_reduction_upper_bound_not_final_acceptance"] < .02 and not (OUT / "c0/t1").exists() and not (OUT / "c0/t2").exists(),
        "all_four_original_full_pair_conditions_passed": len(state["pairs"][0]["checks"]) == 4 and all(state["pairs"][0]["checks"].values()),
        "all_three_saved_full11_protected_invariants_passed": len(state["pairs"][0]["invariants"]) == 3
            and all(row["all_protected"] and len(row["checks"]) == 11 for row in state["pairs"][0]["invariants"].values()),
        "old_fees_and_whole_wall_not_component_selected": state["known_extra_costs"]["g0_actual_lease"] > 0 and state["known_extra_costs"]["p0_actual_lease"] > 0
            and state["primary_complete_affected_initial_entry_wall_including_all_compare_and_actual_exit"],
        "both_actual_full_priority_factories_and_all7338_calls_closed": all(all(row["priority"]["checks"].values()) for row in state["pairs"][0]["rows"].values()),
    }
    fresh = compare_initial_outputs(OUT / "c0/t0/native/initial", OUT / "c0/t0/lazy/initial")
    checks["current_full_decisions_K_poses1002_actual_JPEG_and_pairs_recomparison_exact"] = fresh["all_protected"]
    left, right = OUT / "c0/t0/native_quality", OUT / "c0/t0/lazy_quality"
    names = sorted(path.name for path in left.iterdir())
    checks["all_current7338_cache_files_bytes_exact"] = len(names) == 7338 and names == sorted(path.name for path in right.iterdir()) and all((left / name).read_bytes() == (right / name).read_bytes() for name in names)
    instances = []
    for prefix in ("g0", "p0", "c0_first"):
        lease, guard = read(OUT / (prefix + "_lease.json")), read(OUT / (prefix + "_GUARD.json"))
        checks[prefix + "_actual_guard_lease_completed0_and_released"] = lease["returncode"] == guard["returncode"] == 0 and bool(lease["lease_released_utc"])
        instances += guard["owned_process_identities"]
        instances += [{"pid": lease[key + "pid"], "creation_time": lease[key + "creation_time"]} for key in ("guard_", "child_")]
    for mode in ("native", "lazy"):
        guard = read(OUT / "c0/t0" / (mode + "_GUARD.json"))
        instances += guard["owned_process_identities"]
        instances += [state["pairs"][0]["rows"][mode]["owner"]]
        checks[mode + "_all_original_current_statistics_and_shared_DLL_restored"] = (
            read(OUT / "c0/t0" / mode / "VECTOR_INPUT.json")["trace"]["engine"]["native_calls"] == 7338
            and read(OUT / "c0/t0" / mode / "FUSED_INPUT.json")["original_quality_alias_restored"]
            and guard["returncode"] == 0)
    instances = list({(row["pid"], row["creation_time"]): row for row in instances}.values())
    alive = []
    for row in instances:
        try:
            if psutil.Process(row["pid"]).create_time() == row["creation_time"]:
                alive.append(row)
        except psutil.NoSuchProcess:
            pass
    checks["all_saved_actual_Windows_PID_birth_instances_ended"] = not alive
    checks["no_WSL_commands_in_recorded_full_initial_runs"] = all("wsl" not in row["command"] for row in state["runs"])
    checks["parent33_unique_and_R160_not_counted"] = read(DOC / "CURRENT_STATE.json")["additional_accepted_count"] == 33
    parent = read(DOC / "CURRENT_STATE.json")
    checks["unique33_mechanisms_and_old_R149_only_one"] = len(parent["accepted_mechanisms"]) == len(set(parent["accepted_mechanisms"])) == 33 and sum(name.split("_", 1)[0] == "R149" for name in parent["accepted_mechanisms"]) == 1 and not any("R160" in name for name in parent["accepted_mechanisms"])
    failed = read(OUT / "NEGATIVE_V1_SELF_REVIEW_001.json")
    failed_guard, failed_lease = read(OUT / "v1_negative_review_GUARD.json"), read(OUT / "v1_negative_review_lease.json")
    failed_source = data_identity(OUT / "negative_review_source_001.py")
    checks["first_review_identifier_bug_and_failed_source_preserved"] = failed["status"] == "failed_verification" and [key for key, value in failed["checks"].items() if not value] == ["unique33_mechanisms_and_old_R149_only_one"] and failed_guard["returncode"] == failed_lease["returncode"] == 1 and all(failed_source[key] == failed["source"][key] for key in ("bytes", "sha256"))
    report = (DOC / "R160_LAZY_CORNER_ORDER_V1_RESULTS.md").read_text(encoding="utf-8")
    task = (DOC / "R160_LAZY_CORNER_ORDER_TASK.md").read_text(encoding="utf-8")
    links = [Path(path) for path in re.findall(r"\]\((C:/[^)]+)\)", report)]
    checks["report_original_text_scope_cost_and_failures_verified"] = all(part in report for part in ("33/50", "0.070974%", "0.060567%", "只作组成项", "三fresh", "root自审", "尚未单独验证原因"))
    checks["all_report_links_existing_or_two_explicit_closing_targets"] = bool(links) and all(path.is_file() or path in (target, OUT / "FINAL_V1_POST_EXIT_001.json") for path in links)
    checks["task_preserves_registered_complete2percent_gate_and_stop"] = all(part in task for part in ("净>=2%", "先执行第1对", "不足即关闭计0", "六侧", "7338/6797/1002/11583"))
    checks["actual_output_and_disk_bounds_kept"] = sum(path.stat().st_size for path in OUT.rglob("*") if path.is_file()) <= 12 * 1024**3 and shutil.disk_usage(OUT).free >= 20 * 1024**3
    atomic_json(target, {"status": "verified_negative_first_full_RAW_heap_version" if all(checks.values()) else "failed_verification",
        "checks": checks, "current_full_recomparison": fresh, "Windows_saved_instances": instances, "alive_Windows": alive,
        "reviewer": {"pid": psutil.Process().pid, "creation_time": psutil.Process().create_time()},
        "source": data_identity(Path(__file__)), "source_count": len(identities),
        "original_report_read_as_text": True, "independent_agent_review": False,
        "new_optimization_count": 0, "parent_accepted_count": 33,
        "failed_first_review_fee_seconds": failed_lease["child_elapsed_seconds"],
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()})
    print(json.dumps({"checks": checks, "source_count": len(identities), "Windows_instances": len(instances), "new_count": 0}), flush=True)
    if not all(checks.values()):
        raise AssertionError("The separate complete sorting negative self-review failed")


if __name__ == "__main__":
    main()
