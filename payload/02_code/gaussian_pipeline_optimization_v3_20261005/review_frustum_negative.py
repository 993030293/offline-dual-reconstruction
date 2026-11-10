"""Review the failed private viewer candidate and its saved behavior evidence."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[2]
OWN = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/r145_001"
JOBS = ["g0", "g1", "g2", "p0", "t0", "g3", "t1", "p1", "d0", "d1"]


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def identity(path):
    content = path.read_bytes()
    return {"path": str(path), "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}


def exited(pid, creation):
    try:
        process = psutil.Process(pid)
        return abs(process.create_time() - creation) > .000001 or not process.is_running()
    except psutil.NoSuchProcess:
        return True


def save(path, payload):
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def equal_order(a, b):
    return Path(a["path"]).read_bytes() == Path(b["path"]).read_bytes()


def subsequence(left, right):
    a = np.fromfile(left["ordering"]["path"], dtype="<u4")
    b = np.fromfile(right["ordering"]["path"], dtype="<u4")
    if len(a) != left["active"] or len(b) != right["active"] or len(b) > len(a):
        return False
    if len(b) == 0:
        return True
    if len(np.unique(a)) != len(a):
        return False
    positions = np.empty(max(int(a.max()), int(b.max())) + 1, dtype=np.int64)
    positions.fill(-1)
    positions[a] = np.arange(len(a))
    selected = positions[b]
    return bool(np.all(selected >= 0) and np.all(np.diff(selected) > 0))


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    output = parser.parse_args().output.resolve()
    output.relative_to(OWN)
    output.mkdir(parents=True, exist_ok=False)
    process = psutil.Process()
    save(output / "PROCESS_IDENTITY.json", {"scope": "Windows", "pid": process.pid, "creation_time": process.create_time()})
    receipts = {name: load(OWN / name / "RESULT.json") for name in JOBS}
    checks = {}
    checks["first_fixture_and_caller_checker_failures_preserved"] = all(receipts[name]["status"] == "failed_verification" for name in ["g0", "g1"])
    checks["both_generic22_passed_without_acceptance"] = all(receipts[name]["status"] == "generic_passed_screen_pending" and len(receipts[name]["checks"]) == 22 and all(c["passed"] for c in receipts[name]["checks"]) for name in ["g2", "g3"])
    stable = receipts["p0"]["pairs"][0]
    checks["old_stable124_positive_is_not_acceptance"] = len(stable["rows"]) == 124 and all(r["exact"] and r["retained_order_subsequence"] for r in stable["rows"]) and stable["reduction"] == 1 - stable["candidateTriangles"] / stable["nativeTriangles"] and stable["reduction"] >= .2 and receipts["p0"]["newOptimizationCount"] == 0
    for name, failures in [("t0", 3), ("t1", 6)]:
        rows = receipts[name]["rows"]
        checks[name + "_actual_transition_failure_preserved"] = receipts[name]["status"] == "actual_transition_quality_changed_no_acceptance" and len(rows) == 9 and all(r["camera_exact"] for r in rows) and sum(not r["rgba_bytes_exact"] for r in rows) == failures
    screen = receipts["p1"]
    a, b = screen["episodes"]
    changed = []
    protected = ["name", "width", "height", "projection", "position", "direction", "count", "mode", "worldMatrix", "quaternion", "originalActive"]
    for left, right in zip(a["captures"], b["captures"]):
        changed.append({"name": left["name"], "camera_population_exact": all(left[k] == right[k] for k in protected),
            "rgba_bytes_exact": left["rgba_sha256"] == right["rgba_sha256"],
            "native_full_order_bytes_exact": equal_order(left["fullOrdering"], right["fullOrdering"]),
            "actual_draw_subsequence_exact": subsequence(left, right)})
    checks["V2_all124_camera_population_full_and_selected_order_rechecked"] = len(a["captures"]) == len(b["captures"]) == len(changed) == 124 and all(r["camera_population_exact"] and r["native_full_order_bytes_exact"] and r["actual_draw_subsequence_exact"] for r in changed)
    checks["V2_27_stable_RGBA_failures_and_whole_regression_preserved"] = screen["status"] == "failed_verification" and sum(not r["rgba_bytes_exact"] for r in changed) == 27 and b["elapsedSeconds"] > a["elapsedSeconds"] * 1.1
    gpu_rows = []
    for name in ["d0", "d1"]:
        result = receipts[name]
        for index, row in enumerate(result["rows"]):
            left = result["native"]["records"][index]["gpu"]
            right = result["candidate"]["records"][index]["gpu"]
            length = left["population"] * 16
            x = np.fromfile(left["displayPacked"]["path"], dtype=np.uint8, count=length)
            y = np.fromfile(right["displayPacked"]["path"], dtype=np.uint8, count=length)
            offsets = np.flatnonzero(x != y)
            uploads = []
            for entry, cpu, gpu, count in [(left, "fullCpuOrder", "fullGpuOrder", left["originalActive"]),
                    (right, "fullCpuOrder", "fullGpuOrder", right["originalActive"]),
                    (right, "maskedCpuOrder", "maskedGpuOrder", right["active"])]:
                cp = Path(entry[cpu]["path"]).read_bytes()
                gp = Path(entry[gpu]["path"]).read_bytes()[:count * 4]
                uploads.append(len(cp) == len(gp) == count * 4 and cp == gp)
            gpu_rows.append({"node": name, "range": row["range"], "bytes": length, "changed_bytes": len(offsets),
                "changed_byte_offsets_in_splat": np.bincount(offsets % 16, minlength=16).tolist(),
                "original_and_masked_GPU_uploads_exact": all(uploads),
                "readback_reproduces_report": len(x) == len(y) == length and len(offsets) == row["payload"]["changed"],
                "all_failed_pixel_modes_preserved": not row["selectedPixelsExact"] and not row["candidateFullPixelsExact"] and row["candidateFullVersusMaskedExact"]})
    checks["all_four_actual_GPU_payload_failures_and_exact_uploads_recomputed"] = len(gpu_rows) == 4 and all(r["original_and_masked_GPU_uploads_exact"] and r["readback_reproduces_report"] and r["all_failed_pixel_modes_preserved"] for r in gpu_rows)
    snapshots, inputs, runtimes = [], [], []
    current = {}
    for name, result in receipts.items():
        for row in result["sources"]:
            candidates = [x for x in result["sources"] if Path(x["path"]).name == Path(row["path"]).name]
            snap = OWN / name / "sources" / Path(row["path"]).name
            value = identity(snap)
            snapshots.append({"node": name, "source": row["path"], "snapshot": str(snap),
                "exact_executed_version": len(candidates) == 1 and value["bytes"] == row["bytes"] and value["sha256"] == row["sha256"]})
        for row in result["inputIdentities"]:
            path = Path(row["path"])
            value = current.setdefault(str(path), identity(path))
            inputs.append({"node": name, "path": str(path), "original_input_unchanged": value == row})
        derivation = load(OWN / name / "DERIVATION.json")
        runtimes.append({"node": name, "original_encoding": derivation["unchanged_original_SPKPK001_encoding"],
            "exact_native_runtime": identity(OWN / name / "native_runtime.js")["sha256"] == derivation["native_runtime_sha256"],
            "exact_candidate_runtime": identity(OWN / name / "candidate_runtime.js")["sha256"] == derivation["candidate_runtime_sha256"]})
    checks["all_executed_source_versions_and_runtime_bytes_preserved"] = all(r["exact_executed_version"] for r in snapshots) and all(r["exact_native_runtime"] and r["exact_candidate_runtime"] and r["original_encoding"] for r in runtimes)
    checks["all_original_assets_vendor_encoding_and_inputs_unchanged"] = all(r["original_input_unchanged"] for r in inputs)
    leases, instances, budgets = [], [], []
    for name in JOBS:
        lease = load(OWN / ("LEASE_" + name.upper() + ".json"))
        guard = load(OWN / (name.upper() + "_PROCESS.json"))
        leases.append(lease)
        ids = guard["owned_process_identities"] + [{"pid": lease[k + "_pid"], "creation_time": lease[k + "_creation_time"]} for k in ["guard", "child"]]
        for item in ids:
            instances.append({"node": name, **item, "actual_owned_instance_exited": exited(item["pid"], item["creation_time"])})
        budgets.append(guard["peak_tree_RSS_bytes"] <= 8 * 1024 ** 3 and guard["elapsed_seconds"] <= 900)
    checks["all10_lease_and_guard_terminal_states_correct"] = all(r["status"] == "child_exited" and r.get("lease_released_utc") and r["returncode"] == (1 if r["owner"] in ["root_R145_generic_001", "root_R145_generic_002", "root_R145_V2_screen_001"] else 0) for r in leases)
    checks["all_owned_prior_Windows_process_instances_actual_exited"] = all(r["actual_owned_instance_exited"] for r in instances)
    checks["all_measured_process_tree_budgets_preserved"] = all(budgets)
    state = load(ROOT / "research/2026-10-05-gaussian-optimization-round3-50/CURRENT_STATE.json")
    accepted = []
    for path in state["acceptance_receipts"]:
        record = load(ROOT / path)
        counts = [record[k] for k in ["new_optimization_count", "additional_accepted_count", "accepted_count_contribution"] if k in record]
        accepted.append({"path": path, "one_mechanism": bool(counts) and all(x == 1 for x in counts)})
    checks["all22_actual_unique_accepted_receipts_preserved"] = state["additional_accepted_count"] == len(accepted) == len(set(state["acceptance_receipts"])) == len(set(state["accepted_mechanisms"])) == 22 and all(r["one_mechanism"] for r in accepted)
    checks["no_candidate_failure_or_diagnostic_was_accepted"] = all(r["newOptimizationCount"] == 0 for r in receipts.values()) and not (OWN / "ACCEPTANCE.json").exists()
    checks["unrun_formal_recovery_external_and_training_not_present"] = not any((OWN / name).exists() for name in ["c0", "formal", "recovery", "e0", "train"])
    checks["private_storage_and_disk_budgets_preserved"] = sum(p.stat().st_size for p in OWN.rglob("*") if p.is_file()) < 30 * 1024 ** 3 and shutil.disk_usage(ROOT).free >= 20 * 1024 ** 3
    result = {"status": "negative_candidate_self_review_passed" if all(checks.values()) else "failed_verification",
        "review_type": "separate_root_self_review_no_independent_agent", "checks": checks,
        "source_snapshots": snapshots, "input_identities": inputs, "actual_runtime_identities": runtimes,
        "all_V2_stable_positions": changed, "actual_GPU_payload_rows": gpu_rows,
        "actual_Windows_instances": instances, "all22_actual_acceptance_receipts": accepted,
        "known10_prior_lease_child_seconds": sum(r["child_elapsed_seconds"] for r in leases),
        "old_R138_family_measured_child_seconds": 80.6468966,
        "source_read_intake_and_static_not_fully_metered_no_complete_cost_claim": True,
        "accepted_count_contribution": 0, "parent_count": 22,
        "not_run": ["formal_three_fresh_whole_pairs", "actual_error_new_context_complete_recovery", "external_object_chain", "GT", "new_training", "production_deployment"],
        "review_seconds_before_publication": time.perf_counter() - started,
        "own_review_guard_and_OS_lock_require_post_exit_check": True}
    save(output / "RESULT.json", result)
    print(json.dumps({"status": result["status"], "checks": len(checks), "snapshots": len(snapshots), "inputs": len(inputs),
        "prior_instances": len(instances), "seconds": result["review_seconds_before_publication"]}), flush=True)
    if not all(checks.values()):
        raise AssertionError(json.dumps({k: v for k, v in checks.items() if not v}))


if __name__ == "__main__":
    main()
