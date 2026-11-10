"""Review measured negative SSIM costs and close original resource owners."""

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
V1 = ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"
OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/is001"
sys.path.insert(0, str(V1))
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
    args = parser.parse_args()
    destination = OUT / "r0"
    linux_folders = [OUT / "g0", OUT / "g1"] + [OUT / f"c0/p{index}/{mode}/measurement" for index in range(3) for mode in ("native", "interior")]
    if args.linux:
        rows = [{**read(folder / "PROCESS_IDENTITY.json"), "gone": gone(read(folder / "PROCESS_IDENTITY.json"))} for folder in linux_folders]
        copies = []
        for folder in linux_folders:
            contract = read(folder / ("CONTRACT.json" if folder.name in ("g0", "g1") else "RESULT.json"))
            sources = contract["sources"]
            saved = sorted((folder / "source_copies").iterdir())
            for copy, identity in zip(saved, sources):
                copies.append({"copy": str(copy), "exact": copy.read_bytes() == Path(identity["path"]).read_bytes()})
            if len(saved) != len(sources) or sources != [data_identity(row["path"]) for row in sources]:
                raise ValueError("An exact Linux evaluator source or saved copy changed")
        checks = {"all8_actual_Linux_instances_gone": len(rows) == 8 and all(row["gone"] for row in rows),
                  "all8_actual_Linux_guards_completed": all(read(folder / "WSL_RESOURCE.json")["status"] == "completed" for folder in linux_folders),
                  "all70_native_saved_source_copies_exact": len(copies) == 70 and all(row["exact"] for row in copies)}
        atomic_json(destination / "LINUX_REVIEW.json", {"status": "passed" if all(checks.values()) else "failed", "checks": checks,
                    "instances": rows, "copies": copies, "new_optimization_count": 0})
        if not all(checks.values()):
            raise AssertionError("The exact native source or original Linux closure failed")
        return
    destination.mkdir(parents=True, exist_ok=False)
    owner = psutil.Process()
    atomic_json(destination / "PROCESS_IDENTITY.json", {"pid": owner.pid, "creation_time": owner.create_time()})
    contract, campaign = read(OUT / "c0/CONTRACT.json"), read(OUT / "c0/RESULT.json")
    identities = contract["sources"] + contract["inputs"]
    identity_checks = []
    for expected in identities:
        actual = data_identity(mapped(expected["path"]))
        identity_checks.append({"path": str(mapped(expected["path"])), "exact": (expected["bytes"], expected["sha256"]) == (actual["bytes"], actual["sha256"])})
    source_copies = sorted((OUT / "c0/source_copies").iterdir())
    if len(source_copies) != len(contract["sources"]):
        raise ValueError("The saved complete campaign source count changed")
    copy_checks = [{"path": str(copy), "exact": copy.read_bytes() == mapped(expected["path"]).read_bytes()}
                   for copy, expected in zip(source_copies, contract["sources"])]
    lease_paths = [OUT / (name + "_LEASE_001.json") for name in ("GENERIC", "EXTRA", "CAMPAIGN")]
    guard_paths = [OUT / (name + "_GUARD_001.json") for name in ("GENERIC", "EXTRA", "CAMPAIGN")]
    guard_paths += [OUT / f"c0/p{index}/{mode}/WINDOWS_GUARD.json" for index in range(3) for mode in ("native", "interior")]
    win = []
    for path in guard_paths:
        guard = read(path)
        if guard["status"] != "completed":
            raise ValueError("An original actual Windows guard failed")
        win.extend(guard["owned_process_identities"])
    for path in lease_paths:
        lease = read(path)
        if lease["returncode"] != 0 or not lease.get("lease_released_utc"):
            raise ValueError("An original experiment lease failed or remained owned")
        win += [{"pid": lease["guard_pid"], "creation_time": lease["guard_creation_time"]},
                {"pid": lease["child_pid"], "creation_time": lease["child_creation_time"]}]
    win.append(read(OUT / "c0/PROCESS_IDENTITY.json"))
    unique = {(row["pid"], row["creation_time"]): row for row in win}
    windows = [{**row, "gone": gone(row)} for row in unique.values()]
    subprocess.run(["wsl", "-d", "E26-C2-LIV-20260907-134802", "--", "/opt/e26_thin_surface_v1/venv/bin/python", "-B",
                    "/mnt/c/" + Path(__file__).as_posix()[3:], "--linux"], check=True, timeout=120)
    linux = read(destination / "LINUX_REVIEW.json")
    parameters, metrics, artifacts = [], [], []
    maximum_ssim = 0.0
    for index in range(3):
        evaluations = {mode: read(OUT / f"c0/p{index}/{mode}/pair/evaluation/EVALUATION.json") for mode in ("native", "interior")}
        for mode in ("native", "interior"):
            folder = OUT / f"c0/p{index}/{mode}"
            measurement = read(folder / "measurement/RESULT.json")
            parameters.append({"pair": index, "mode": mode, "all_original_checks": all(measurement["checks"].values()),
                               "complete_model_fields": len(measurement["model_states"]) == 2,
                               "all96_SSIM_input_rows": len(measurement["rows"]) == 96})
            for label in ("baseline", "combined"):
                recorded = evaluations[mode]["results"][label]["model"]
                actual = data_identity(folder / f"pair/{label}/model/point_cloud/iteration_12000/point_cloud.ply")
                if (recorded["bytes"], recorded["sha256"]) != (actual["bytes"], actual["sha256"]):
                    raise ValueError("An original complete physical model output changed")
            ast.parse((folder / "measurement/COMPOSED_PAIR.py").read_text(encoding="utf-8"))
            ast.parse((folder / "measurement/COMPOSED_DAG.py").read_text(encoding="utf-8"))
        for label in ("baseline", "combined"):
            first, second = evaluations["native"]["results"][label]["rows"], evaluations["interior"]["results"][label]["rows"]
            for left, right in zip(first, second):
                maximum_ssim = max(maximum_ssim, *(abs(left[region]["SSIM"] - right[region]["SSIM"]) for region in ("full_image", "target")))
                metrics.append({"metadata_exact": {k: v for k, v in left.items() if k not in ("full_image", "target")} ==
                                {k: v for k, v in right.items() if k not in ("full_image", "target")},
                                "PSNR_MSE_exact": all(left[region][key] == right[region][key] for region in ("full_image", "target") for key in ("PSNR_dB", "MSE"))})
        artifacts.append((OUT / f"c0/p{index}/native/pair/evaluation/SIX_VIEW_COMPARISON.jpg").read_bytes() ==
                         (OUT / f"c0/p{index}/interior/pair/evaluation/SIX_VIEW_COMPARISON.jpg").read_bytes())
    medians = campaign["raw_medians"]
    raw_speed = 1 - medians["interior"]["complete_phase_seconds"] / medians["native"]["complete_phase_seconds"]
    campaign_extra = read(lease_paths[2])["child_elapsed_seconds"] - sum(row["complete_whole_seconds"] for row in campaign["rows"])
    if campaign_extra < 0:
        raise ValueError("The original campaign fee envelope does not contain its side costs")
    known_fees = sum(read(path)["child_elapsed_seconds"] for path in lease_paths[:2]) + campaign_extra
    cuda = {mode: statistics.median(read(OUT / f"c0/p{index}/{mode}/measurement/RESULT.json")["peak_CUDA_reserved_bytes"]
                                    for index in range(3)) for mode in ("native", "interior")}
    checks = {"all43_original_source_input_identities": len(identity_checks) == 43 and all(row["exact"] for row in identity_checks),
              "all12_campaign_saved_source_copies_exact": len(copy_checks) == 12 and all(row["exact"] for row in copy_checks),
              "all_native_sources_and_Linux_instances_closed": all(linux["checks"].values()),
              "all_actual_prior_Windows_instances_gone": all(row["gone"] for row in windows),
              "all_six_complete_real_evaluators_protected": len(parameters) == 6 and all(all(row[k] for k in ("all_original_checks", "complete_model_fields", "all96_SSIM_input_rows")) for row in parameters),
              "all144_model_view_rows_metadata_PSNR_MSE_exact": len(metrics) == 144 and all(all(row.values()) for row in metrics),
              "all_actual_SSIM_values_original_1e_7": maximum_ssim <= 1e-7,
              "all_three_complete_artifacts_byte_exact": len(artifacts) == 3 and all(artifacts),
              "raw_speed_before_any_first_review_fees_already_below5percent": 0 <= raw_speed < .05,
              "task_speed_gate_not_replaced_by_posthoc_CUDA_memory_observation": "速度V2至少5%" in (ROOT / "research/2026-10-05-gaussian-optimization-round3-50/R134_INTERIOR_SSIM_REUSE_TASK.md").read_text(encoding="utf-8")}
    result = {"status": "closed_negative_original_speed_gate" if all(checks.values()) else "failed_review", "checks": checks,
              "raw_complete_phase_reduction": raw_speed, "raw_medians": medians,
              "known_first_generic_and_campaign_extra_fees_seconds": known_fees,
              "campaign_extra_seconds": campaign_extra, "review_before_publication_seconds": time.perf_counter() - STARTED,
              "maximum_absolute_SSIM_error": maximum_ssim, "original_input_source_identities": identity_checks,
              "original_campaign_source_copies": copy_checks, "Windows": windows,
              "observed_actual_CUDA_reserved_bytes": cuda, "observed_CUDA_reduction_not_new_registered_acceptance": 1 - cuda["interior"] / cuda["native"],
              "root_single_chat_self_review": True, "new_optimization_count": 0,
              "external_recovery_training_and_new_preview": "not_run_after_original_speed_failure",
              "own_review_exit_requires_caller_check": True}
    atomic_json(destination / "REVIEW.json", result)
    print(json.dumps({key: result[key] for key in ("status", "checks", "raw_complete_phase_reduction", "known_first_generic_and_campaign_extra_fees_seconds", "observed_actual_CUDA_reserved_bytes")}), flush=True)
    if not all(checks.values()):
        raise AssertionError("A full SSIM negative result source, cost or real resource closure failed")


if __name__ == "__main__":
    main()
