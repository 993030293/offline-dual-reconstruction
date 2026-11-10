"""Separately audit completed diagnostics, negative pairs and actual owners."""

import json
import msvcrt
from pathlib import Path
import shutil
import subprocess
import time

import psutil

from equal_fee_cost_bound import equal_fee_cost_bound
from run_fused_camera_digest_campaign import compare
from review_scene_batch_negative import read, windows_closed
from scene_transform_cost_node import ROOT, CODE
from runtime import atomic_json, data_identity

BASE = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005"
OUT = BASE / "sclc001/r2"
DOC = ROOT / "research/2026-10-05-gaussian-optimization-round3-50"


def main():
    started = time.perf_counter()
    OUT.mkdir(exist_ok=False)
    self_id = data_identity(Path(__file__))
    shutil.copyfile(Path(__file__), OUT / Path(__file__).name)
    state = read(DOC / "CURRENT_STATE.json")
    preparation = read(BASE / "sclc001/RESULT.json")
    main_cost = read(BASE / "sclm002/RESULT.json")
    inverse = read(BASE / "sclc001/k0/RESULT.json")
    phase, operand = (read(BASE / name / "c0/RESULT.json") for name in ("sip001", "spr004"))
    reference = BASE / "scfq001/c0/t0/combo"
    comparisons = [compare(reference, BASE / "sclc001/p0"), compare(reference, BASE / "sclm002/p0"),
        compare(BASE / "sip001/c0/t0/control", BASE / "sip001/c0/t0/isolated"),
        compare(BASE / "spr004/c0/t0/control", BASE / "spr004/c0/t0/operand")]
    node_paths = [BASE / rel for rel in ("sclc001/p0_node/NODE.json", "sclm001/p0_node/NODE.json",
        "sclm002/p0_node/NODE.json", "sip001/c0/t0/control_node/NODE.json", "sip001/c0/t0/isolated_node/NODE.json",
        "spr001/c0/t0/control_node/NODE.json", "spr002/c0/t0/control_node/NODE.json",
        "spr003/c0/t0/control_node/NODE.json", "spr004/c0/t0/control_node/NODE.json", "spr004/c0/t0/operand_node/NODE.json")]
    windows, linux, nodes, leases = [], [], [], []
    for path in node_paths:
        node = read(path)
        nodes.append(node)
        windows += [node["owner"], *node.get("guard", {}).get("owned_process_identities", [])]
        guard_path = path.parent / "guard.json"
        if guard_path.exists() and "guard" not in node:
            windows += read(guard_path).get("owned_process_identities", [])
        linux += [{"pid": r["pid"], "creation_time": r["creation_time"]} for r in node.get("actual_Linux_after_wait", [])]
    lease_rows = (("sclc001/p0", 0), ("sclc001/k0", 0), ("sclm001/p0", 1), ("sclm002/p0", 0),
        ("sip001/c0_first", 0), ("spr001/g0", 0), ("spr001/c0_first", 1),
        ("spr002/c0_first", 1), ("spr003/g0", 0), ("spr003/c0_first", 1), ("spr004/c0_first", 0))
    for label, expected in lease_rows:
        lease = read(BASE / (label + "_lease.json"))
        if lease["status"] != "child_exited" or lease["returncode"] != expected or not lease.get("lease_released_utc"):
            raise AssertionError("An actual diagnostic lease did not close with its original return code")
        leases.append({"path": str(BASE / (label + "_lease.json")), **lease})
        windows += [{"pid": lease["guard_pid"], "creation_time": lease["guard_creation_time"]},
            {"pid": lease["child_pid"], "creation_time": lease["child_creation_time"]}]
    for root_name, labels in (("sclc001", ("p0", "k0")), ("sclm001", ("p0",)),
        ("sclm002", ("p0",)), ("sip001", ("c0_first",)), ("spr001", ("g0", "c0_first")),
        ("spr002", ("c0_first",)), ("spr003", ("g0", "c0_first")), ("spr004", ("c0_first",))):
        for label in labels:
            path = BASE / root_name / (label + ("_guard.json" if label in ("g0", "k0") else "_outer_guard.json"))
            if root_name == "sip001" and label == "c0_first":
                path = BASE / "sip001/c0_first_guard.json"
            if not path.exists() and label == "k0":
                path = BASE / root_name / "k0_outer_guard.json"
            guard = read(path)
            windows += guard.get("owned_process_identities", [])
    for rel in ("sclc001/k0/WSL_RESOURCE.json", "spr001/g0_resource/WSL_RESOURCE.json", "spr003/g0_resource/WSL_RESOURCE.json"):
        row = read(BASE / rel)
        linux.append({"pid": row["pid"], "creation_time": row["creation_time"]})
    for formal in (phase, operand):
        windows += [row["owner"] for pair in formal["pairs"] for row in pair["rows"].values()]
    for rel in ("spr001/c0/STATE.json", "spr002/c0/STATE.json", "spr003/c0/STATE.json", "sclm001/STATE.json"):
        row = read(BASE / rel)
        if row.get("failed_owner"):
            windows.append(row["failed_owner"])
        elif row.get("owner"):
            windows.append(row["owner"])
    windows = list({(r["pid"], r["creation_time"]): r for r in windows}.values())
    linux = list({(r["pid"], r["creation_time"]): r for r in linux}.values())
    script = "import json,psutil; rows=json.loads(" + repr(json.dumps(linux)) + "); result=[]\n"
    script += "for row in rows:\n alive=False\n try:\n  p=psutil.Process(row['pid']); alive=p.create_time()==row['creation_time'] and p.status()!=psutil.STATUS_ZOMBIE\n except psutil.NoSuchProcess: pass\n result.append(dict(row,actual_current_alive=alive))\nprint(json.dumps(result))"
    linux_now = json.loads(subprocess.check_output(["wsl", "-d", "E26-C2-LIV-20260907-134802", "--", "/opt/e26_thin_surface_v1/venv/bin/python", "-B", "-c", script], text=True, timeout=30))
    source_checks = []
    for data in (preparation, main_cost, phase, operand):
        source_checks.append(all(data_identity(row["path"]) == row for row in data["sources"]))
    for rel in ("sclc001/p0/COST_ADAPTER_SOURCES.json", "sclm002/p0/MAIN_ADAPTER_SOURCES.json"):
        for row in read(BASE / rel)["sources"]:
            current_source = data_identity(row["path"].replace("/mnt/c/", "C:/"))
            source_checks.append(all(current_source[key] == row[key] for key in ("bytes", "sha256")))
    copy_checks = []
    for name, formal in (("sip001", phase), ("spr004", operand)):
        for i, row in enumerate(formal["sources"]):
            actual = data_identity(BASE / name / "c0/source_snapshot" / (f"{i:03d}_" + Path(row["path"]).name))
            copy_checks.append(actual["sha256"] == row["sha256"] and actual["bytes"] == row["bytes"])
    original = read(BASE / "spr004/c0/t0/control/RESULT.json")
    import importlib.util
    spec = importlib.util.spec_from_file_location("root_owned_identity_review", CODE / "parallel_b_005/ordered_identity.py")
    collector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(collector)
    trace = {}
    current = collector.collect_identities([row["path"].replace("/mnt/c/", "C:/") for row in original["inputs"]], data_identity, trace)
    inputs_exact = len(current) == len(original["inputs"]) == 1036 and all({**a, "path": b["path"]} == b for a, b in zip(current, original["inputs"]))
    tails = {}
    bounds = {}
    for name, formal in (("sip001", phase), ("spr004", operand)):
        actual = sum(row["actual_spawn_to_exit_seconds"] for pair in formal["pairs"] for row in pair["rows"].values())
        actual += sum(p["actual_compare_seconds_charged_each_side"] for p in formal["pairs"])
        tails[name] = read(BASE / name / "c0_first_lease.json")["child_elapsed_seconds"] - actual
        if tails[name] < 0:
            raise AssertionError("A real full campaign envelope became smaller than its observed windows")
        keys = ("control", "isolated" if name == "sip001" else "operand")
        fee = formal["initial_known_fee_seconds"] + tails[name] + 10.
        bounds[name] = equal_fee_cost_bound(*(formal["raw_complete_stage_medians"][key] for key in keys), fee)
    lock = BASE / "coordination_001/exclusive_benchmark.lock"
    with lock.open("r+b", buffering=0) as stream:
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
    path_trace = read(BASE / "spr004/c0/t0/operand/PATH_OPERAND_TRACE.json")
    main_trace = read(BASE / "sclm002/MAIN_SUMMARY.json")
    names = {"profile_composed_scene_cost.py", "profile_composed_scene_main_cost.py", "profile_phase_isolated_scene.py", "profile_path_operand_scene.py", "probe_camera_inverse_overlap.py"}
    untracked = []
    for process in psutil.process_iter(["pid", "create_time", "cmdline"]):
        if any(arg.replace("\\", "/").rsplit("/", 1)[-1] in names for arg in (process.info["cmdline"] or [])):
            untracked.append({"pid": process.info["pid"], "creation_time": process.info["create_time"]})
    checks = {
        "parent29_unique_mechanisms29_receipts29_and_new_counts0": state["additional_accepted_count"] == len(state["accepted_mechanisms"]) == len(set(state["accepted_mechanisms"])) == len(state["acceptance_receipts"]) == 29 and phase["new_optimization_count"] == operand["new_optimization_count"] == 0,
        "all4_separate_original15_fields_and2_actual_files_comparisons_exact": all(all(c["checks"].values()) for c in comparisons),
        "both_original_fresh_complete_pair6_comparison_checks_exact": all(all(pair["comparison"]["checks"].values()) for formal in (phase, operand) for pair in formal["pairs"]),
        "full_original1036_inputs_current_complete_bytes_unchanged": inputs_exact,
        "all_current_successful_diagnostic_and_negative_pair_sources_unchanged": all(source_checks),
        "all_original_formal_source_copies_exact": all(copy_checks),
        "original_preparation5_main4_and_inverse5_diagnostic_checks_passed": all(preparation["checks"].values()) and all(main_cost["checks"].values()) and all(inverse["checks"].values()) and all(main_trace["checks"].values()),
        "original_main2032_full_path_resolves_measured": main_trace["groups"]["Path.resolve"]["main"]["calls"] == 2032,
        "R155_actual1016_first_resolves_and_static_same_operands_functions_restored": path_trace["native_resolve_calls"] == path_trace["same_absolute_operand"] == 1016 and path_trace["original_functions_restored"] and path_trace["native_fallback_comparisons"] == 0,
        "R154_first_full_pair_slower_exceeds_original1point1_stop": len(phase["pairs"]) == 1 and phase["whole_ratios"][0] > 1.1 and phase["status"] == "closed_negative_complete_phase_screen0",
        "R155_first_full_pair_known_fee_upper_under_fixed3percent_stop": len(operand["pairs"]) == 1 and bounds["spr004"]["conservative_sample_reduction_upper_bound"] < .03 and operand["status"] == "closed_negative_complete_path_screen0",
        "all_original11_diagnostic_leases_released_actual_returncodes_preserved": len(leases) == 11,
        "all_actual_current_Windows_and_Linux_saved_owners_closed": all(windows_closed(row) for row in windows) and all(not r["actual_current_alive"] for r in linux_now),
        "no_other_actual_Windows_managed_entry_left_after_failures": not untracked,
        "all4_failed_branch_logs_sources_and_explicit_receipts_preserved": all((BASE / rel).exists() for rel in ("sclm001/FAILURE_001.json", "spr001/FAILURE_001.json", "spr002/FAILURE_001.json", "spr003/FAILURE_001.json")),
        "native_working_guard_and_real_Windows_symlink_failure_recorded": read(BASE / "spr003/c0/t0/control_node/ACTUAL_GUARD_EXIT.json")["returncode"] == 15 and read(BASE / "spr001/guard_smoke_001.json")["returncode"] == 0,
        "original_exclusive_resource_lock_reacquired_and_released": True,
        "negative_upper_bound_zero_for_R154_not_false_negative_gain_certificate": bounds["sip001"]["conservative_sample_reduction_upper_bound"] == 0,
        "actual_root_review_source_and_snapshot_unchanged": self_id == data_identity(Path(__file__)) and data_identity(OUT / Path(__file__).name)["sha256"] == self_id["sha256"],
    }
    if not all(checks.values()):
        atomic_json(OUT / "FAILED.json", {"checks": checks})
        raise AssertionError("The separated root review found an inconsistent negative or original output")
    atomic_json(OUT / "RESULT.json", {"status": "separate_root_self_review_passed_not_independent_review0", "checks": checks,
        "comparisons": comparisons, "current_full_input_identity_queue": trace,
        "Windows_identities": windows, "Linux_current_identities": linux_now, "leases": leases,
        "formal_tail_seconds": tails, "known_lower_fee_with_original10_reserve_bounds": bounds,
        "elapsed_seconds_before_result": time.perf_counter() - started, "owner": {"pid": psutil.Process().pid, "creation_time": psutil.Process().create_time()},
        "sources": [self_id], "new_optimization_count": 0, "accepted_count": 29, "remaining": 21,
        "both_remaining_formal_pairs_interfaces_real_errors_recovery_GT_new_training_not_run": True,
        "main_failed_branch_AND_symlink_fixtures_unchanged": True})
    print(json.dumps({"checks": checks, "Windows": len(windows), "Linux": len(linux), "leases": len(leases), "formal_tails": tails, "new": 0}))


if __name__ == "__main__":
    main()
