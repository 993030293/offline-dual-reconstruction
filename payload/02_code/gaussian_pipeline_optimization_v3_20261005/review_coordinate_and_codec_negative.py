"""Close two negative branches with source, science, cost and actual PID audits."""

import ast
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).parent
BASE = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005"
OUT = BASE / "oup018/negative_review0"
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def gone(identity):
    try:
        return psutil.Process(identity["pid"]).create_time() != identity["creation_time"]
    except psutil.NoSuchProcess:
        return True


def main():
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(Path(__file__), OUT / "REVIEW_SOURCE.py")
    coordinate = BASE / "oup017/R169"
    codec = BASE / "oup018/R170"
    c = read(coordinate / "c0/RESULT.json")
    s = read(codec / "c0/STATE.json")
    sources = {}
    for value in [c, read(coordinate / "g0/RESULT.json"), read(BASE / "oup018/a0/RESULT.json"),
            read(BASE / "oup018/b0/RESULT.json"), s, read(codec / "g1/RESULT.json"), read(codec / "g2/RESULT.json"),
            *[read(codec / f"c0/t{i}/{tag}/CONTRACT.json") for i in range(2) for tag in ("b", "c")]]:
        for identity in value.get("sources", []):
            if identity["path"].startswith("/mnt/c/"):
                path = "C:/" + identity["path"][7:]
            elif identity["path"].startswith("/"):
                continue
            else:
                path = identity["path"]
            current = data_identity(Path(path))
            exact = current["bytes"] == identity["bytes"] and current["sha256"] == identity["sha256"]
            sources[identity["path"]] = {"identity": identity, "current_exact": exact}
    windows, linux, node_rows = {}, {}, []
    for family in (coordinate, codec):
        for path in family.rglob("NODE.json"):
            node = read(path)
            node_rows.append({"path": str(path), "status": node["status"], "returncode": node["returncode"],
                              "Windows_closed": not node["live_Windows_after_wait"],
                              "Linux_closed": not any(row["alive"] for row in node["actual_Linux_after_wait"]),
                              "resource_peaks": node["resource_peaks"]})
            for row in node["guard"].get("owned_process_identities", []) + [node["owner"]]:
                windows[(row["pid"], row["creation_time"])] = row
            for row in node["Linux_resource_receipts"]:
                linux[(row["pid"], row["creation_time"])] = {"pid": row["pid"], "creation_time": row["creation_time"]}
    for path in codec.rglob("WSL_RESOURCE.json"):
        row = read(path)
        linux[(row["pid"], row["creation_time"])] = {"pid": row["pid"], "creation_time": row["creation_time"]}
    leases = []
    for family in (coordinate, BASE / "oup018", codec):
        for path in family.glob("*_LEASE.json"):
            if path.name == "NEGATIVE_REVIEW_LEASE.json":
                continue
            value = read(path)
            leases.append({"path": str(path), "elapsed_seconds": value["child_elapsed_seconds"],
                           "released": "lease_released_utc" in value, "returncode": value["returncode"]})
            for prefix in ("guard", "child"):
                if prefix+"_pid" in value and prefix+"_creation_time" in value:
                    row = {"pid": value[prefix+"_pid"], "creation_time": value[prefix+"_creation_time"]}
                    windows[(row["pid"], row["creation_time"])] = row
    linux_code = "import json,psutil; rows=json.loads(" + repr(json.dumps(list(linux.values()))) + "); out=[]\n"
    linux_code += ("for row in rows:\n alive=False\n try:\n  p=psutil.Process(row['pid']); "
                   "alive=p.create_time()==row['creation_time'] and p.status()!=psutil.STATUS_ZOMBIE\n"
                   " except psutil.NoSuchProcess: pass\n out.append(dict(row,alive=alive))\nprint(json.dumps(out))")
    actual_linux = json.loads(subprocess.check_output(["wsl", "-d", "E26-C2-LIV-20260907-134802", "--",
        "/opt/e26_thin_surface_v1/venv/bin/python", "-B", "-c", linux_code], text=True, timeout=30))
    original_invariants = [read(codec / f"c0/t{i}/INVARIANTS.json") for i in range(2)]
    numeric = read(codec / "c0/NUMERIC_DIFFERENCE_DIAGNOSTIC.json")
    prepared = CODE / "fault_codec_localization.py"
    ast.parse(prepared.read_text(encoding="utf-8"))
    parent = read(ROOT / "research/2026-10-05-gaussian-optimization-round3-50/CURRENT_STATE.json")
    checks = {
        "R169_three_original_fresh_pairs_and_complete15_fields_retained": len(c["pairs"]) == 3
            and all(all(row["checks"].values()) for row in c["pairs"]),
        "R169_original_negative_whole_guard_preserved": c["whole_ratios"]["candidate"] > 1.03
            if isinstance(c["whole_ratios"], dict) and "candidate" in c["whole_ratios"] else c["status"].startswith("closed_negative"),
        "R170_both_complete24_48_original_tolerances_and_matched_bits_passed": all(
            all(v for k, v in row["checks"].items() if k != "all24_complete_numeric_pose_centers_stats_exact")
            for row in original_invariants),
        "R170_second_pair_actual_science_extra_exact_and_whole_guard_failure_retained": len(s["pairs"]) == 2
            and not s["pairs"][1]["checks"]["all_original_protected_science_including_extra_exact_poses"]
            and not s["pairs"][1]["checks"]["each_actual_complete_parent_whole_within1_1"],
        "R170_third_pair_and_prepared_full_error_SIGINT_recovery_not_run": not (codec / "c0/t2").exists()
            and not list(codec.rglob("FAULT_RESULT.json")) and not (codec / "r0").exists(),
        "R170_numeric_diagnostic_not_threshold_change_or_root_cause_claim": numeric[
            "not_a_relaxation_of_the_frozen_guard_or_proof_of_root_cause"],
        "018_both_CPU_prototypes_complete48_and_eight_negative_checks": all(
            all(read(BASE / path)["checks"].values()) for path in ("oup018/a0/RESULT.json", "oup018/b0/RESULT.json")),
        "R170_g1_g2_all_actual_native_component_checks_passed": all(
            all(read(codec / path)["checks"].values()) for path in ("g1/RESULT.json", "g2/RESULT.json")),
        "all_relevant_current_Windows_sources_and_native_library_identities": all(row["current_exact"] for row in sources.values()),
        "all_original_nodes_actual_completed_and_saved_processes_closed": all(row["status"] == "completed"
            and row["returncode"] == 0 and row["Windows_closed"] and row["Linux_closed"] for row in node_rows),
        "all_prior_Windows_PID_birth_instances_ended": all(gone(row) for row in windows.values()),
        "all_prior_Linux_PID_birth_instances_ended": all(not row["alive"] for row in actual_linux),
        "all_prior_exclusive_leases_released_failures_retained": all(row["released"] for row in leases)
            and read(codec / "G0_LEASE.json")["returncode"] == 1
            and read(codec / "REMAINING_LEASE.json")["returncode"] == 1,
        "parent_effective_count_stays38_both_branches_zero": parent["additional_accepted_count"] == 38
            and c["new_optimization_count"] == s["new_optimization_count"] == 0,
    }
    result = {"status": "negative_branches_actual_closed_pending_text_and_writer_exit" if all(checks.values()) else "failed_verification",
              "checks": checks, "review_is_separate_root_self_review_not_independent": True,
              "sources": list(sources.values()), "nodes": node_rows, "Windows": list(windows.values()),
              "Linux": actual_linux, "leases": leases,
              "prototype_and_R170_known_actual_pipe_extra_bytes": sum(read(codec / path)["extra_actual_written_input_bytes"]
                    for path in ("g1/RESULT.json", "g2/RESULT.json")),
              "formal_four_sides_pipe_bytes_are_not_recharged_as_extra": True,
              "final_net_speed_work_acceptance_not_computed_due_frozen_guard_failure": True,
              "negative_review_seconds": time.perf_counter()-started, "new_optimization_count": 0,
              "prepared_but_unexecuted_full_fault_source": data_identity(prepared),
              "writer": {"pid": psutil.Process().pid, "creation_time": psutil.Process().create_time()}}
    atomic_json(OUT / "RESULT.json", result)
    print(json.dumps({key: result[key] for key in ("status", "checks", "negative_review_seconds")}), flush=True)
    if not all(checks.values()):
        raise AssertionError("A negative branch science, source or actual closure review failed")


if __name__ == "__main__":
    main()
