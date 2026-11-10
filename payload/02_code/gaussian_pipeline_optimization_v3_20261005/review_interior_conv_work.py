"""Perform a separate root review of actual complete outputs, work and closure."""

import json
import msvcrt
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import time

import psutil

import run_interior_conv_campaign as campaign


def identity(path):
    return campaign.data_identity(path)


def trace_elements(record):
    rows = record["rows"]
    total = sum(row.get("output_elements", 0) for row in rows
                if row.get("completed") and row.get("device", "").startswith("cuda"))
    if total != record["successful_CUDA_output_elements"]:
        raise AssertionError("A native successful convolution census does not recompute")
    if sum(row.get("output_bytes", 0) for row in rows
           if row.get("completed") and row.get("device", "").startswith("cuda")) != record["successful_CUDA_output_bytes"]:
        raise AssertionError("A native successful convolution byte census does not recompute")
    return total


def historical_work_upper():
    old = campaign.ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/is001"
    full = campaign.read(old / "c0/RESULT.json")
    components = []
    for row in full["rows"]:
        folder = Path(row["measurement"]).parent
        folder = campaign.windows(folder)
        measurement = campaign.read(folder / "RESULT.json")
        if not all(measurement["checks"].values()):
            raise AssertionError("An original historical complete SSIM protection failed")
        bound = 0
        for ordinal, item in enumerate(measurement["rows"]):
            batch, channels, height, width = item["shape"]
            area = height * width
            if row["mode"] == "interior" and ordinal % 2:
                area = 20 * (height + width)
            bound += batch * channels * area * 5
        components.append({"kind": "original_complete_side", "source": str(folder / "RESULT.json"),
                           "mode": row["mode"], "upper_elements_not_historical_census": bound})
    generic = campaign.read(old / "g0/RESULT.json")
    bound = 0
    for row in generic["rows"]:
        batch, channels, height, width = row.get("shape", [1, 3, 620, 812])
        baseline = batch * channels * height * width * 5
        candidate = batch * channels * (20 * (height + width) if row.get("interior_reused") else height * width) * 5
        bound += baseline + candidate
    components.append({"kind": "original113_generic_rows_including_CPU_and_error_overbudget",
                       "source": str(old / "g0/RESULT.json"), "upper_elements_not_historical_census": bound})
    extra = campaign.read(old / "g1/RESULT.json")
    if len(extra["rows"]) != 17 or not all(extra["checks"].values()):
        raise AssertionError("The original extra generic evidence changed")
    bound = 17 * 2 * 2 * 3 * max(37 * 53, 20 * (37 + 53)) * 5
    components.append({"kind": "original17_extra_rows_max_batch2_channels3_37x53_both_algorithms",
                       "source": str(old / "g1/RESULT.json"), "upper_elements_not_historical_census": bound})
    frozen = campaign.read(old / "g0/CONTRACT.json")["sources"][1]
    current = identity(campaign.CODE / "interior_ssim_reuse.py")
    if (frozen["sha256"], frozen["bytes"]) != (current["sha256"], current["bytes"]):
        raise AssertionError("The frozen historical interior mathematics changed")
    return {"components": components, "total_upper_elements_not_historical_census": sum(
                c["upper_elements_not_historical_census"] for c in components),
            "proof": "Each original Q30 map call executes at most five native convolutions. Reused target has four ten-pixel strips; their total output area is 20*(H+W). All saved supported current calls and original generic shapes are included. CPU/error cases are conservatively charged as CUDA work. No profiler or physical MAC claim."}


def work_cost():
    own = campaign.OWN
    formal = campaign.read(own / "c1/RESULT.json")
    extra = []
    for file in sorted(own.rglob("CONV_WORK.json")):
        relative = file.relative_to(own)
        if relative.parts[0] == "c1":
            continue
        value = campaign.read(file)
        extra.append({"source": str(file), "elements": trace_elements(value["trace"])})
    for path, name in ((own / "g0/RESULT.json", "trace"), (own / "x0/RESULT.json", "work")):
        extra.append({"source": str(path), "elements": trace_elements(campaign.read(path)[name])})
    historical = historical_work_upper()
    extra_total = historical["total_upper_elements_not_historical_census"] + sum(r["elements"] for r in extra)
    surcharge = extra_total / 6
    medians = {mode: statistics.median(row["CUDA_conv_output_elements"] for row in formal["rows"] if row["mode"] == mode)
               for mode in ("native", "interior")}
    return {"historical_upper": historical, "all_new_extra_native_census": extra,
            "extra_elements_measured_or_conservative_upper": extra_total, "per_side_elements_surcharge": surcharge,
            "formal_raw_medians": medians,
            "net_reduction_lower_bound_under_registered_six_side_model": 1 -
                (medians["interior"] + surcharge) / (medians["native"] + surcharge),
            "formal_six_sides_not_recharged_as_extra": True}


def time_cost(extra_seconds=0., tail_reserve=10.):
    own = campaign.OWN
    formal = campaign.read(own / "c1/RESULT.json")
    old = campaign.ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/is001"
    old_result, old_closure = campaign.read(old / "c0/RESULT.json"), campaign.read(old / "NEGATIVE_CLOSURE_001.json")
    components = [{"source": str(old / "c0/RESULT.json"), "seconds": sum(r["complete_whole_seconds"] for r in old_result["rows"])},
                  {"source": str(old / "NEGATIVE_CLOSURE_001.json"), "seconds": old_closure[
                      "all_measured_nonoverlapping_first_failed_review_and_campaign_extra_fees_seconds"]}]
    for file in sorted(own.glob("*_lease.json")):
        lease = campaign.read(file)
        seconds = lease["child_elapsed_seconds"]
        if file.name == "c1_lease.json":
            seconds -= sum(r["complete_whole_seconds"] for r in formal["rows"])
        if seconds < 0:
            raise AssertionError("A formal execution envelope was double charged or inconsistent")
        components.append({"source": str(file), "seconds": seconds})
    components.append({"source": "known_native_read_and_web_api_metadata_commands_upper_budget", "seconds": 20.})
    components.append({"source": "separate_root_review_and_publisher_measured_so_far", "seconds": extra_seconds})
    components.append({"source": "reserved_post_exit_and_native_text_execution_tail", "seconds": tail_reserve})
    total = sum(r["seconds"] for r in components)
    surcharge = total / 6
    adjusted = {mode: [r["complete_whole_seconds"] + surcharge + formal["comparisons"][r["pair"]]["comparison_seconds"] / 2
                      for r in formal["rows"] if r["mode"] == mode] for mode in ("native", "interior")}
    return {"components": components, "known_execution_extra_seconds_with20_upper_and_tail_reserve": total,
            "per_side_seconds_surcharge": surcharge, "adjusted_whole_seconds": adjusted,
            "adjusted_whole_median_ratio": statistics.median(adjusted["interior"]) / statistics.median(adjusted["native"]),
            "raw_whole_median_ratio": formal["raw_medians"]["interior"]["complete_whole_seconds"] /
                formal["raw_medians"]["native"]["complete_whole_seconds"],
            "raw_each_whole_ratios": [c["whole_ratio"] for c in formal["comparisons"]],
            "all_text_research_R_and_D_or_deployment_payback_measured": False}


def closed(row):
    try:
        process = psutil.Process(row["pid"])
        return process.create_time() != row["creation_time"] or not process.is_running()
    except psutil.NoSuchProcess:
        return True


def main():
    started = time.perf_counter()
    own = campaign.OWN
    output = own / "SELF_REVIEW_001.json"
    if output.exists():
        raise FileExistsError("The actual review needs a fresh receipt")
    formal = campaign.read(own / "c1/RESULT.json")
    checks = {"three_fresh_current_complete_pairs": len(formal["rows"]) == 6 and len(formal["comparisons"]) == 3,
              "old_failed_c0_not_erased_or_used_as_new_formal": campaign.read(own / "c0/RESULT.json")["status"] == "failed_original_complete_protection"
                  and campaign.read(own / "c0/RESULT.json")["comparisons"][1]["whole_ratio"] > 1.1}
    for trial in range(3):
        comparison = campaign.compare(own / f"c1/p{trial}/native", own / f"c1/p{trial}/interior")
        checks[f"pair{trial}_all_original_outputs_and_protections_recomputed"] = all(comparison["checks"].values())
        for mode in ("native", "interior"):
            folder = own / f"c1/p{trial}" / mode
            meter = campaign.read(folder / "CONV_WORK.json")
            row = next(r for r in formal["rows"] if r["pair"] == trial and r["mode"] == mode)
            if trace_elements(meter["trace"]) != row["CUDA_conv_output_elements"]:
                raise AssertionError("A formal native convolution row changed")
    for name in ("g0", "x0", "r1"):
        checks[name + "_all_original_component_or_recovery_checks"] = all(campaign.read(own / name / "RESULT.json")["checks"].values())
    current_ids = []
    for contract in (own / "c1/CONTRACT.json", own / "r1/CONTRACT.json", own / "x0/CONTRACT.json"):
        payload = campaign.read(contract)
        for row in payload["sources"] + payload["inputs"]:
            path = campaign.windows(row["path"])
            current = identity(path)
            if (current["sha256"], current["bytes"]) != (row["sha256"], row["bytes"]):
                raise AssertionError("An original complete source or input changed")
            current_ids.append(current)
    checks["all_current_sources_and_complete_inputs_unchanged"] = True
    copies = 0
    for folder in (own / "c0/source_copies", own / "c1/source_copies"):
        contract = campaign.read(folder.parent / "CONTRACT.json")
        for index, row in enumerate(contract["sources"]):
            target = folder / (f"{index:02d}_" + campaign.windows(row["path"]).name)
            actual = identity(target)
            if (actual["sha256"], actual["bytes"]) != (row["sha256"], row["bytes"]):
                raise AssertionError("An actual executed source snapshot changed")
            copies += 1
    checks["failed_and_new_actual_controller_source_copies_preserved"] = copies == 30
    old_fault = campaign.read(own / "r0/native_error/FAULT_RESULT.json")
    saved = own / "r0/SOURCE_16_fault_interior_conv_work.py"
    checks["first_fault_entry_import_failure_preserved"] = old_fault["error"]["message"] == "Select the rasterizer before importing a CUDA backend"
    if identity(saved)["sha256"] != old_fault["script"]["sha256"]:
        raise AssertionError("The first actual failed fault-entry source was overwritten")
    windows_rows, linux_rows = [], []
    for file in sorted(own.rglob("WINDOWS_GUARD.json")) + sorted(own.glob("*_guard.json")):
        record = campaign.read(file)
        if record["maximum_RSS_bytes"] != 8 * 1024**3 or record["peak_tree_RSS_bytes"] > record["maximum_RSS_bytes"]:
            raise AssertionError("An original actual Windows resource bound changed")
        windows_rows.extend(record.get("owned_process_identities", []))
    for file in own.rglob("WSL_RESOURCE.json"):
        record = campaign.read(file)
        if record["maximum_RSS_bytes"] != 8 * 1024**3 or record["peak_tree_RSS_bytes"] > record["maximum_RSS_bytes"]:
            raise AssertionError("An original actual Linux memory bound changed")
        linux_rows.append({"pid": record["pid"], "creation_time": record["creation_time"]})
    unique_windows = {(r["pid"], r["creation_time"]): r for r in windows_rows}
    checks["all_recorded_actual_Windows_instances_closed"] = all(closed(r) for r in unique_windows.values())
    probe_source = "import json,psutil; rows=" + json.dumps(linux_rows) + "; out=[]\nfor r in rows:\n try:\n  p=psutil.Process(r['pid']); gone=p.create_time()!=r['creation_time'] or not p.is_running()\n except psutil.NoSuchProcess: gone=True\n out.append(dict(r,gone=gone))\nprint(json.dumps(out))"
    result = subprocess.run(["wsl", "-d", "E26-C2-LIV-20260907-134802", "--",
        "/opt/e26_thin_surface_v1/venv/bin/python", "-c", probe_source], capture_output=True, text=True, check=True)
    linux_checks = json.loads(result.stdout)
    checks["all_recorded_actual_Linux_instances_closed"] = all(r["gone"] for r in linux_checks)
    leases = [campaign.read(file) for file in own.glob("*_lease.json")]
    checks["all_actual_original_leases_exited_and_released"] = all(r.get("lease_released_utc") and r.get("returncode") in (0, 1) for r in leases)
    lock = campaign.ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/coordination_001/exclusive_benchmark.lock"
    with lock.open("r+b", buffering=0) as handle:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    checks["OS_exclusive_lock_acquired_and_released"] = True
    work = work_cost()
    costs = time_cost(extra_seconds=time.perf_counter() - started)
    checks["registered_net_native_work_lower_bound_at_least10percent"] = work["net_reduction_lower_bound_under_registered_six_side_model"] >= .1
    checks["original_raw_and_adjusted_whole_guards"] = costs["raw_whole_median_ratio"] <= 1.01
    checks["original_raw_and_adjusted_whole_guards"] &= costs["adjusted_whole_median_ratio"] <= 1.01
    checks["original_raw_and_adjusted_whole_guards"] &= all(r <= 1.1 for r in costs["raw_each_whole_ratios"])
    output_bytes = sum(p.stat().st_size for p in own.rglob("*") if p.is_file())
    checks["original_complete_output_and_disk_bounds"] = output_bytes <= 12 * 1024**3 and shutil.disk_usage(own).free >= 20 * 1024**3
    owner = psutil.Process()
    campaign.atomic_json(output, {"status": "root_self_review_passed" if all(checks.values()) else "failed_verification",
        "checks": checks, "work": work, "cost": costs, "reviewer_is_independent_agent": False,
        "reviewer": {"pid": owner.pid, "creation_time": owner.create_time()},
        "current_sources_and_inputs": len(current_ids), "current_data_and_source_identities": current_ids,
        "preserved_controller_copies": copies, "Windows_instances": list(unique_windows.values()),
        "Linux_instances": linux_checks, "original_released_leases": len(leases), "output_bytes": output_bytes,
        "review_seconds": time.perf_counter() - started, "new_optimization_count": 0})
    if not all(checks.values()):
        raise AssertionError("A separate root original-output, cost, work or closure review failed")
    print(json.dumps({"status": "root_self_review_passed", "checks": len(checks), "net_native_work": work[
        "net_reduction_lower_bound_under_registered_six_side_model"], "whole_ratio": costs["adjusted_whole_median_ratio"]}), flush=True)


if __name__ == "__main__":
    main()

