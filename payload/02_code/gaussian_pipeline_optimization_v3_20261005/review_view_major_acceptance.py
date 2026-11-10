"""Review a fixed view-major evaluation against its preregistered memory policy."""

import time

STARTED = time.perf_counter()

import argparse
import ast
import json
from pathlib import Path
import statistics
import subprocess
import sys

import psutil

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parent
BASE = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005"
OUT = BASE / "vm001"
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def gone(row):
    try:
        return psutil.Process(row["pid"]).create_time() != row["creation_time"]
    except psutil.NoSuchProcess:
        return True


def mapped(path):
    text = str(path)
    return Path("C:/" + text[7:]) if text.startswith("/mnt/c/") else Path(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--linux", action="store_true")
    parser.add_argument("--output-name", choices=("a0", "a1", "a2"), default="a0")
    args = parser.parse_args()
    destination = OUT / args.output_name
    if args.linux:
        processes = []
        for path in OUT.rglob("PROCESS_IDENTITY.json"):
            row = read(path)
            if row.get("scope") == "WSL":
                processes.append(row)
        unique = {(row["pid"], row["creation_time"]): row for row in processes}
        rows = [{**row, "gone": gone(row)} for row in unique.values()]
        checks = {"all_actual_saved_Linux_instances_gone": bool(rows) and all(row["gone"] for row in rows),
                  "all_six_current_Linux_full_evaluation_guards_completed": all(read(OUT / f"c2/p{index}/{mode}/measurement/WSL_RESOURCE.json")["status"] == "completed"
                      for index in range(3) for mode in ("native", "view")),
                  "external_real_Linux_guard_completed": read(OUT / "x0/WSL_RESOURCE.json")["status"] == "completed"}
        atomic_json(destination / "LINUX_REVIEW.json", {"status": "passed" if all(checks.values()) else "failed", "checks": checks,
                    "instances": rows, "new_optimization_count": 0})
        if not all(checks.values()):
            raise AssertionError("An actual Linux owner or mandatory resource guard remained unresolved")
        return
    destination.mkdir(parents=True, exist_ok=False)
    owner = psutil.Process()
    ancestor_identities = {(item.pid, item.create_time()) for item in owner.parents()}
    atomic_json(destination / "PROCESS_IDENTITY.json", {"pid": owner.pid, "creation_time": owner.create_time()})
    contract = read(OUT / "c2/CONTRACT.json")
    inputs = contract["sources"] + contract["inputs"]
    current = []
    for expected in inputs:
        actual = data_identity(mapped(expected["path"]))
        current.append({"path": str(mapped(expected["path"])), "exact": (expected["bytes"], expected["sha256"]) == (actual["bytes"], actual["sha256"])})
    catalog = []
    for filename in ("CONTRACT.json", "RESULT.json"):
        for path in OUT.rglob(filename):
            row = read(path)
            catalog.extend(row.get("sources", []))
    copies = []
    for folder in OUT.rglob("source_copies"):
        for path in sorted(folder.iterdir()):
            identity = data_identity(path)
            matches = [row for row in catalog if Path(str(row["path"]).replace("\\", "/")).name == path.name[3:]
                       and (row["bytes"], row["sha256"]) == (identity["bytes"], identity["sha256"])]
            copies.append({"path": str(path), "recorded_revision_exact": bool(matches),
                           "matched_recorded_source": matches[0]["path"] if matches else None})
    guards = sorted(OUT.glob("*_GUARD_*.json")) + sorted(OUT.glob("c*/p*/*/WINDOWS_GUARD.json"))
    guards += sorted(OUT.glob("r*/*/WINDOWS_GUARD.json"))
    windows = []
    for path in guards:
        row = read(path)
        if any(item["pid"] == owner.pid and item["creation_time"] == owner.create_time()
               for item in row["owned_process_identities"]):
            continue
        if row["status"] not in ("completed", "failed_child"):
            raise ValueError("An actual prior bounded guard did not terminate normally")
        if row["peak_tree_RSS_bytes"] > row["maximum_RSS_bytes"] or row["elapsed_seconds"] > row["maximum_seconds"]:
            raise ValueError("A registered original Windows resource budget was exceeded")
        windows.extend(row["owned_process_identities"])
    leases = []
    for path in sorted(OUT.glob("*_LEASE_*.json")):
        row = read(path)
        if (row.get("child_pid"), row.get("child_creation_time")) in ancestor_identities:
            continue
        leases.append(path)
    for path in leases:
        row = read(path)
        if row["status"] != "child_exited" or not row.get("lease_released_utc"):
            raise ValueError("A prior complete or failed measured group did not release its lease")
        windows.extend([{"pid": row["guard_pid"], "creation_time": row["guard_creation_time"]},
                        {"pid": row["child_pid"], "creation_time": row["child_creation_time"]}])
    for path in OUT.glob("*/PROCESS_IDENTITY.json"):
        row = read(path)
        if row.get("scope") != "WSL" and path.parent != destination:
            windows.append(row)
    unique = {(row["pid"], row["creation_time"]): row for row in windows}
    win_rows = [{**row, "gone": gone(row)} for row in unique.values()]
    subprocess.run(["wsl", "-d", "E26-C2-LIV-20260907-134802", "--", "/opt/e26_thin_surface_v1/venv/bin/python", "-B",
                    "/mnt/c/" + Path(__file__).as_posix()[3:], "--linux", "--output-name", args.output_name], check=True, timeout=120)
    linux = read(destination / "LINUX_REVIEW.json")
    campaign, external, recovery = read(OUT / "c2/RESULT.json"), read(OUT / "x0/RESULT.json"), read(OUT / "r1/RESULT.json")
    maximum_ssim = 0.0
    exact_rows, exact_artifacts, exact_states = [], [], []
    for index in range(3):
        evaluations = {mode: read(OUT / f"c2/p{index}/{mode}/pair/evaluation/EVALUATION.json") for mode in ("native", "view")}
        for label in ("baseline", "combined"):
            first = evaluations["native"]["results"][label]["rows"]
            second = evaluations["view"]["results"][label]["rows"]
            exact_rows.append(first == second and len(first) == 24)
            for left, right in zip(first, second):
                maximum_ssim = max(maximum_ssim, *(abs(left[region]["SSIM"] - right[region]["SSIM"]) for region in ("full_image", "target")))
        exact_artifacts.append((OUT / f"c2/p{index}/native/pair/evaluation/SIX_VIEW_COMPARISON.jpg").read_bytes() ==
                               (OUT / f"c2/p{index}/view/pair/evaluation/SIX_VIEW_COMPARISON.jpg").read_bytes())
        for mode in ("native", "view"):
            measurement = read(OUT / f"c2/p{index}/{mode}/measurement/RESULT.json")
            exact_states.append(all(measurement["checks"].values()) and len(measurement["model_states"]) == 2)
            for label in ("baseline", "combined"):
                expected = evaluations[mode]["results"][label]["model"]
                actual = data_identity(OUT / f"c2/p{index}/{mode}/pair/{label}/model/point_cloud/iteration_12000/point_cloud.ply")
                if (expected["bytes"], expected["sha256"]) != (actual["bytes"], actual["sha256"]):
                    raise ValueError("An original current complete physical model copy changed")
            ast.parse((OUT / f"c2/p{index}/{mode}/measurement/COMPOSED_PAIR.py").read_text(encoding="utf-8"))
    for name in ("view_major_evaluation.py", "evaluate_view_major.py", "run_view_major_campaign.py", "check_view_major_alias.py",
                 "check_view_major_generic.py", "external_view_major_reference.py", "fault_view_major_evaluation.py", "run_view_major_recovery.py"):
        ast.parse((CODE / name).read_text(encoding="utf-8"), filename=name)
    prior_leases = sorted((BASE / "is001").glob("*_LEASE_*.json"))
    previous_fees = sum(read(path)["child_elapsed_seconds"] for path in prior_leases)
    current_raw_wholes = sum(row["complete_whole_seconds"] for row in campaign["rows"])
    current_fees = sum(read(path)["child_elapsed_seconds"] for path in leases) - current_raw_wholes
    if current_fees < 0:
        raise ValueError("The current exited campaign envelope excludes a side measurement")
    shared = previous_fees + current_fees
    raw = campaign["raw_medians"]
    cuda_reduction = 1 - raw["view"]["peak_CUDA_reserved_bytes"] / raw["native"]["peak_CUDA_reserved_bytes"]
    fees_medians = {mode: {key: raw[mode][key] + shared/6 for key in ("complete_phase_seconds", "complete_whole_seconds")}
                    for mode in ("native", "view")}
    ratios = [(next(row for row in campaign["rows"] if row["pair"] == index and row["mode"] == "view")["complete_whole_seconds"] + shared/6) /
              (next(row for row in campaign["rows"] if row["pair"] == index and row["mode"] == "native")["complete_whole_seconds"] + shared/6) for index in range(3)]
    checks = {"all45_current_original_source_input_identities": len(current) == 45 and all(row["exact"] for row in current),
              "all_saved_source_copies_exact_their_recorded_revisions": bool(copies) and all(row["recorded_revision_exact"] for row in copies),
              "all_actual_prior_Windows_instances_gone": bool(win_rows) and all(row["gone"] for row in win_rows),
              "all_actual_prior_Linux_instances_and_current_guards_closed": all(linux["checks"].values()),
              "all_three_current_complete_pairs_original_outputs_protected": len(exact_rows) == 6 and all(exact_rows) and all(exact_states)
                  and len(exact_artifacts) == 3 and all(exact_artifacts) and maximum_ssim <= 1e-7,
              "current_generic_and_real_full_alias_regression_passed": read(OUT / "g2/RESULT.json")["status"] == "generic_passed"
                  and all(read(OUT / "g2/RESULT.json")["checks"].values()),
              "full360_external_nonblind_component_passed": external["status"] == "external_component_passed" and all(external["checks"].values()) and len(external["rows"]) == 360,
              "owned_real_failure_SIGINT_same_dir_rejection_fresh_recovery_passed": recovery["status"] == "recovery_passed" and all(recovery["checks"].values()),
              "primary_actual_CUDA_peak_reduction_at_least20percent": cuda_reduction >= .2,
              "all_measured_fees_whole_median_not_increased_each_at_most1_1": fees_medians["view"]["complete_whole_seconds"] <= fees_medians["native"]["complete_whole_seconds"] and all(ratio <= 1.1 for ratio in ratios),
              "old_R134_original_speed_negative_not_reclassified": read(BASE / "is001/NEGATIVE_CLOSURE_001.json")["new_optimization_count"] == 0,
              "actual_preregistered_memory20percent_policy_read_as_text": "至少20%" in (ROOT / "research/2026-10-05-gaussian-optimization-round3-50/R135_VIEW_MAJOR_EVALUATION_TASK.md").read_text(encoding="utf-8")}
    result = {"status": "all_preregistered_checks_passed_final_review_cost_pending" if all(checks.values()) else "failed_review", "checks": checks,
              "current_original_sources_and_inputs": current, "recorded_revision_source_copies": copies, "Windows": win_rows,
              "Linux_review": str(destination / "LINUX_REVIEW.json"), "prior_R134_all_measured_first_and_failed_groups_seconds": previous_fees,
              "R135_prior_measured_fees_and_campaign_extra_seconds": current_fees,
              "all_prior_shared_fees_before_this_exited_review_seconds": shared, "raw_medians": raw,
              "medians_with_prior_fees": fees_medians, "pair_whole_ratios_with_prior_fees": ratios,
              "actual_CUDA_reserved_peak_reduction": cuda_reduction, "maximum_absolute_SSIM_error": maximum_ssim,
              "review_before_publication_seconds": time.perf_counter() - STARTED,
              "root_single_chat_self_review": True, "no_subagent": True, "new_optimization_count": 0,
              "all_prior_leases_released": len(leases), "own_review_exit_and_final_cost_require_caller_check": True}
    atomic_json(destination / "REVIEW.json", result)
    print(json.dumps({key: result[key] for key in ("status", "checks", "actual_CUDA_reserved_peak_reduction", "all_prior_shared_fees_before_this_exited_review_seconds")}), flush=True)
    if not all(checks.values()):
        raise AssertionError("A fixed memory, whole, source, output or actual-owner acceptance protection failed")


if __name__ == "__main__":
    main()
