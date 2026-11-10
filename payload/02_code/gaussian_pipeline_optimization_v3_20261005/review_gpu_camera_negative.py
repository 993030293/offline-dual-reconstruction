"""Close the negative CUDA remapping branch against actual saved evidence."""

import time
STARTED = time.perf_counter()

import argparse
import json
from pathlib import Path
import subprocess
import sys

import psutil

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).parent
OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/rm_gpu001"
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def windows(path):
    value = str(path)
    if value.startswith("/mnt/c/"):
        return Path("C:/" + value[len("/mnt/c/"):])
    return Path(value)


def same_identity(row):
    actual = data_identity(windows(row["path"]))
    return actual["bytes"] == row["bytes"] and actual["sha256"] == row["sha256"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.resolve().relative_to(OUT.resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    run = read(OUT / "c0/RESULT.json")
    generic = read(OUT / "g3/RESULT.json")
    pair = run["pairs"][0]
    guards = [OUT / name for name in ("PIXEL_GUARD_001.json", "ADAPTER_GUARD_001.json",
              "ADAPTER_GUARD_002.json", "ADAPTER_GUARD_003.json", "CAMPAIGN_GUARD_001.json")]
    guards += sorted((OUT / "c0").rglob("*_GUARD.json"))
    guard_rows = [read(path) for path in guards]
    win = {}
    for row in guard_rows:
        for item in row.get("owned_process_identities", []):
            win[(item["pid"], item["creation_time"])] = item
    leases = [OUT / name for name in ("PIXEL_LEASE_001.json", "ADAPTER_LEASE_001.json",
              "ADAPTER_LEASE_002.json", "ADAPTER_LEASE_003.json", "CAMPAIGN_LEASE_001.json")]
    for path in leases:
        row = read(path)
        win[(row["guard_pid"], row["guard_creation_time"])] = {"pid": row["guard_pid"], "creation_time": row["guard_creation_time"]}
        if row.get("child_creation_time"):
            win[(row["child_pid"], row["child_creation_time"])] = {"pid": row["child_pid"], "creation_time": row["child_creation_time"]}
    win_rows = []
    for row in win.values():
        try:
            process = psutil.Process(row["pid"])
            alive = process.create_time() == row["creation_time"]
        except psutil.NoSuchProcess:
            alive = False
        win_rows.append(dict(row, alive=alive))
    resource_paths = sorted(OUT.rglob("WSL_RESOURCE.json"))
    linux = {(read(path)["pid"], read(path)["creation_time"]):
             {"pid": read(path)["pid"], "creation_time": read(path)["creation_time"]} for path in resource_paths}
    payload = json.dumps(list(linux.values()))
    script = "import json,psutil\nrows=json.loads(" + repr(payload) + ")\nresult=[]\n"
    script += "for row in rows:\n try:\n  p=psutil.Process(row['pid']); alive=p.create_time()==row['creation_time'] and p.status()!=psutil.STATUS_ZOMBIE\n except psutil.NoSuchProcess: alive=False\n result.append(dict(row,alive=alive))\nprint(json.dumps(result))\n"
    linux_rows = json.loads(subprocess.check_output(["wsl", "-d", "E26-C2-LIV-20260907-134802", "--",
        "/opt/e26_thin_surface_v1/venv/bin/python", "-B", "-c", script], text=True, timeout=30))
    identities = []
    for record in (read(OUT / "g0/RESULT.json"), read(OUT / "g3/RESULT.json"), run,
                   read(OUT / "c0/t0/native/RESULT.json"), read(OUT / "c0/t0/CUDA/RESULT.json")):
        identities.extend(record.get("sources", []))
        identities.extend(record.get("inputs", []))
    identities.extend(read(OUT / "g0/MANIFEST.json")["payloads"])
    for path in (OUT / "g3", OUT / "c0"):
        for source in path.rglob("WORKER_RESULT.json"):
            record = read(source)
            identities.extend(record["sources"])
            identities.extend(record["map_identities"])
            identities.extend(record["raw_identities"])
    identities = list({(str(windows(row["path"])), row["sha256"]): row for row in identities}.values())
    identity_checks = [dict(row, current_exact=same_identity(row)) for row in identities]
    copies = []
    for directory in [OUT / "g0/source_snapshot", OUT / "g3/source_snapshot", OUT / "c0/source_snapshot",
                      OUT / "c0/t0/native/source_snapshot", OUT / "c0/t0/CUDA/source_snapshot"]:
        for path in sorted(directory.glob("*")):
            original = CODE / path.name[4:]
            if not original.is_file():
                candidates = list((ROOT / "02_code").glob("*/" + path.name[4:]))
                if len(candidates) != 1:
                    raise ValueError("A saved source copy has ambiguous original ownership")
                original = candidates[0]
            copies.append({"path": str(path), "original": str(original), "exact": path.read_bytes() == original.read_bytes()})
    costs = [{"guard": str(path), "guard_child_elapsed_seconds": read(path)["elapsed_seconds"],
              "lease": str(lease), "elapsed_seconds": read(lease)["child_elapsed_seconds"]}
             for path, lease in zip(guards[:5], leases)]
    previous = read(OUT.parent / "rm001/NEGATIVE_CLOSURE_001.json")
    prior_cost = previous["all_four_first_child_seconds_once"] + previous["whole_root_review_seconds"]
    checks = {"registered_first_complete_pair_both_costs_worse_stop": run["status"] == "first_complete_pair_outputs_preserved_both_costs_worse_stop_no_acceptance" and len(run["pairs"]) == 1 and pair["whole_seconds"]["CUDA"] > pair["whole_seconds"]["native"] and pair["complete_reference_preparation_stage_seconds"]["CUDA"] > pair["complete_reference_preparation_stage_seconds"]["native"],
              "all_actual24_48_outputs_and_original_inputs_protected": all(pair["checks"].values()) and read(OUT / "c0/t0/INVARIANTS.json")["all_protected"],
              "current43_complete_pixels_and_all7_error_scope_guards": generic["status"] == "passed" and all(generic["checks"].values()) and len(generic["rows"]) == 43 and len(generic["guard_checks"]) == 7 and all(generic["guard_checks"].values()),
              "initial_CUDA_failure_and_slower_pipe_sources_preserved": "Invalid device argument" in (OUT / "g1/pixels/worker.log").read_text(encoding="utf-8") and len(list((OUT / "g1/source_snapshot").glob("*"))) == 6 and len(list((OUT / "g2/source_snapshot").glob("*"))) == 6,
              "all_current_complete_source_input_and_map_identities_exact": all(row["current_exact"] for row in identity_checks),
              "all_current_saved_source_copies_exact": all(row["exact"] for row in copies),
              "all_recorded_Windows_and_Linux_instances_gone": not any(row["alive"] for row in win_rows + linux_rows),
              "all5_experiment_leases_released": len(leases) == 5 and all(read(path).get("lease_released_utc") and read(path)["status"] == "child_exited" for path in leases),
              "all_original_Win_and_Linux_RSS_limits": all(row["peak_tree_RSS_bytes"] <= 8 * 1024**3 for row in guard_rows) and all(read(path)["peak_tree_RSS_bytes"] <= 8 * 1024**3 for path in resource_paths),
              "later_three_pairs_external_training_and_production_not_claimed": run["new_optimization_count"] == 0 and len(run["runs"]) == 2}
    report = {"status": "negative_branch_closed" if all(checks.values()) else "failed_review", "checks": checks,
              "review_kind": "root_single_chat_self_review", "pair": pair, "current_identities": identity_checks,
              "source_copies": copies, "Windows": win_rows, "Linux": linux_rows,
              "current_nonoverlapping_guard_costs": costs,
              "prior_CPU_negative_first_and_review_seconds": prior_cost,
              "all_prior_and_current_nonoverlapping_experiment_seconds": prior_cost + sum(row["elapsed_seconds"] for row in costs),
              "review_before_publication_seconds": time.perf_counter() - STARTED,
              "complete_three_pair_final_cost_external_and_quality": "not_run_after_registered_negative_first_pair",
              "new_optimization_count": 0, "parent_additional_accepted_count": 18}
    atomic_json(args.output / "REVIEW.json", report)
    print(json.dumps({"status": report["status"], "checks": checks, "current_identities": len(identity_checks),
                      "source_copies": len(copies), "Windows": len(win_rows), "Linux": len(linux_rows)}), flush=True)
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
