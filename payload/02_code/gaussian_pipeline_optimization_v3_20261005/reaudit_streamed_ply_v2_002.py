"""Audit saved complete Gaussian byte proofs and the new recovery supplement."""

import json
from pathlib import Path
import subprocess
import sys
import time

STARTED = time.perf_counter()
ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).parent
sys.path[:0] = [str(CODE), str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005")]

import psutil

from runtime import atomic_json, data_identity
from run_streamed_gaussian_ply_campaign_003 import file_exact, mounted

OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/dpv001"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    folder = OUT / "v2a1"
    folder.mkdir(exist_ok=False)
    campaign = read(OUT / "c1/RESULT.json")
    negative, final = read(OUT / "NEGATIVE_CLOSURE_001.json"), read(OUT / "FINAL_COST_001.json")
    supplement, first = read(OUT / "v2r1/RESULT.json"), read(OUT / "v2r0/RESULT.json")
    checks = {"V1_cost_negative_retained": final["new_optimization_count"] == 0
        and campaign["checks"]["complete_phase_at_least10percent"] is False,
        "original_negative_closure_guards_retained": all(negative["checks"].values()),
        "three_fresh_interleaved_original_orders": campaign["orders"] == [
            ["native", "stream"], ["stream", "native"], ["native", "stream"]],
        "new12_real_permission_cancel_and_recovery_pairs_verified": len(supplement["rows"]) == 12
            and all(supplement["checks"].values()) and all(all(row["checks"].values()) for row in supplement["rows"]),
        "first_empty_write_injection_failure_preserved": first["status"] == "failed_verification"
            and sum(not all(row["checks"].values()) for row in first["rows"]) == 4,
        "both_permission_supplement_identity_restorations_verified": first["effective_uid_after"] == first["original_uid"]
            and supplement["effective_uid_after"] == supplement["original_uid"]}
    windows, linux, files = {}, {}, {}
    for name in ("g0", "g1"):
        generic = read(OUT / name / "RESULT.json")
        checks[name + "_all48_complete_generic_rows"] = len(generic["rows"]) == 48 and all(
            all(row["checks"].values()) for row in generic["rows"]) and all(generic["checks"].values())
    for trial in range(3):
        native, stream = [OUT / f"c1/t{trial}_{mode}" for mode in ("native", "stream")]
        checks[f"t{trial}_all_complete_point_and_native_artifact_bytes_now"] = (
            file_exact(native / "point_cloud.ply", stream / "point_cloud.ply")
            and file_exact(native / "native_oracle.ply", stream / "native_oracle.ply")
            and file_exact(native / "point_cloud.ply", native / "native_oracle.ply")
            and file_exact(stream / "point_cloud.ply", stream / "native_oracle.ply"))
        for mode in ("native", "stream"):
            side = read(OUT / f"c1/t{trial}_{mode}/RESULT.json")
            checks[f"t{trial}_{mode}_full_parameters_rng_and_saved_point_identity_now"] = (
                all(side["checks"].values())
                and side["point_payload"]["path"] == mounted(OUT / f"c1/t{trial}_{mode}/point_cloud.ply")
                and all(data_identity(OUT / f"c1/t{trial}_{mode}/point_cloud.ply")[key] == side["point_payload"][key]
                    for key in ("bytes", "sha256")))
    for path in sorted(OUT.glob("*LEASE*.json")):
        if path.name == "V2_REVIEW_LEASE_002.json":
            continue
        lease = read(path)
        checks[path.name + "_released"] = bool(lease.get("lease_released_utc"))
        for prefix in ("guard", "child"):
            if prefix + "_pid" in lease:
                item = {"pid": lease[prefix + "_pid"], "creation_time": lease[prefix + "_creation_time"]}
                windows[(item["pid"], item["creation_time"])] = item
    nodes = [*OUT.glob("n*/NODE.json"), *OUT.glob("c*/*_node/NODE.json")]
    for path in sorted(nodes):
        node = read(path)
        expected_failure = path.parent.name == "nv2r0"
        checks[str(path.relative_to(OUT)) + "_closed_expected_status"] = (
            node["status"] == ("failed" if expected_failure else "completed")
            and node["final_source_unchanged"] and not node["live_Windows_after_wait"]
            and not any(item["alive"] for item in node["actual_Linux_after_wait"]))
        for item in [node["owner"], *node["guard"]["owned_process_identities"]]:
            windows[(item["pid"], item["creation_time"])] = item
        for item in node["actual_Linux_after_wait"]:
            linux[(item["pid"], item["creation_time"])] = {
                "pid": item["pid"], "creation_time": item["creation_time"]}
    contracts = [*OUT.glob("c*/t*_native/CONTRACT.json"), *OUT.glob("c*/t*_stream/CONTRACT.json"),
        OUT / "g0/CONTRACT.json", OUT / "g1/CONTRACT.json"]
    for path in contracts:
        for item in read(path)["sources_inputs"]:
            files[(item["path"], item["sha256"])] = item
    for item in [*first["sources"], *supplement["sources"]]:
        files[(item["path"], item["sha256"])] = item
    payload = {"processes": list(linux.values()), "files": list(files.values())}
    code = "import json,hashlib,pathlib,psutil; p=json.loads(" + repr(json.dumps(payload)) + "); out={'live':[],'bad_files':[]}\n"
    code += "for i in p['processes']:\n try:\n  q=psutil.Process(i['pid'])\n  if q.create_time()==i['creation_time'] and q.status()!=psutil.STATUS_ZOMBIE: out['live'].append(i)\n except psutil.NoSuchProcess: pass\n"
    code += "for i in p['files']:\n f=pathlib.Path(i['path']); h=hashlib.sha256()\n with f.open('rb') as s:\n  for b in iter(lambda:s.read(1024**2),b''): h.update(b)\n if f.stat().st_size!=i['bytes'] or h.hexdigest()!=i['sha256']: out['bad_files'].append(i['path'])\nprint(json.dumps(out))"
    actual = json.loads(subprocess.check_output(["wsl", "-d", "E26-C2-LIV-20260907-134802", "--",
        "/opt/e26_thin_surface_v1/venv/bin/python", "-B", "-c", code], text=True, timeout=30))
    checks["all_saved_Linux_instances_actually_gone_now"] = not actual["live"]
    checks["all_measured_source_and_input_bytes_preserved_now"] = not actual["bad_files"]
    checks["campaign_Windows_sources_and_original_model_current"] = all(
        data_identity(item["path"]) == item for item in campaign["sources_inputs"])
    live = []
    for item in windows.values():
        try:
            if psutil.Process(item["pid"]).create_time() == item["creation_time"]:
                live.append(item)
        except psutil.NoSuchProcess:
            pass
    checks["all_saved_Windows_instances_actually_gone_now"] = not live
    checks["new_Linux_resource_observer_joined_with_original_bound"] = (
        read(OUT / "v2r1/WSL_RESOURCE.json")["status"] == "completed"
        and read(OUT / "v2r1/WSL_RESOURCE.json")["peak_tree_RSS_bytes"] <= 32 * 1024**3)
    result = {"status": "V2_functional_protection_verified_final_cost_pending" if all(checks.values()) else "failed_verification",
        "checks": checks, "raw_pairs": campaign["pairs"],
        "complete_prior_first_fees_seconds": final["complete_shared_first_fees_including_review_and_campaign_wrapper_seconds"],
        "new_recovery_fees_including_first_failed_test_seconds": sum(read(OUT / name)["child_elapsed_seconds"]
            for name in ("V2_RECOVERY_LEASE_001.json", "V2_RECOVERY_LEASE_002.json")),
        "new_empty_write_diagnostic_reported_tool_wall_seconds": .8,
        "new_reaudit_first_path_failure_fee_seconds": read(OUT / "V2_REVIEW_LEASE_001.json")["child_elapsed_seconds"],
        "current_review_lease_closed_separately": str(OUT / "V2_REVIEW_LEASE_002.json"),
        "review_seconds_inside_script": time.perf_counter() - STARTED,
        "owned_Windows_checked": len(windows), "owned_Linux_checked": len(linux),
        "actual_live_Windows": live, "actual_Linux_and_file_checks": actual, "sources_inputs_checked": len(files),
        "actual_peak_RSS_bytes": final["actual_peak_RSS_bytes"], "actual_RSS_reduction": final["actual_RSS_reduction"],
        "owner": {"pid": psutil.Process().pid, "creation_time": psutil.Process().create_time()},
        "review_kind": "root_self_review_no_subagents", "new_optimization_count": 0,
        "source": data_identity(Path(__file__)), "external_object_Gaussian_and_GT_or_new_training_or_preview": "unrun_unverified"}
    atomic_json(folder / "REVIEW.json", result)
    print(json.dumps({key: result[key] for key in ("status", "checks", "new_recovery_fees_including_first_failed_test_seconds",
        "review_seconds_inside_script", "owned_Windows_checked", "owned_Linux_checked", "sources_inputs_checked")}), flush=True)
    if not all(checks.values()):
        raise AssertionError("A full saved output, source, recovery or actual owner check failed")


if __name__ == "__main__":
    main()

