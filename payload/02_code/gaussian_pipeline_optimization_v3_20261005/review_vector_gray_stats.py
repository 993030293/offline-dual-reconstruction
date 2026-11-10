"""Perform a separate root self-review of original full RAW vector statistics."""

import argparse
from pathlib import Path
import statistics
import sys
import time

import cv2
import numpy as np
import psutil

import profile_quality_corner_cost as reference
from raw_initial_invariants import compare_initial_outputs
from vector_gray_stats_evidence import cost_snapshot

ROOT = reference.ROOT
OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005"


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    output = parser.parse_args().output.resolve()
    base = OUT / "vector_gray_stats_001"
    output.relative_to(base)
    output.mkdir(parents=True, exist_ok=False)
    load, atomic_json, identity = reference.load, reference.atomic_json, reference.data_identity
    process = psutil.Process()
    atomic_json(output / "PROCESS_IDENTITY.json", {"pid": process.pid, "creation_time": process.create_time()})
    state = {"status": "running", "accepted_count_contribution": 0, "review_type": "separate_root_self_review_not_independent_agent"}
    checks = {}
    try:
        results = {name: load(base / name / "RESULT.json") for name in ("b0", "g0", "s0", "v0", "c0")}
        previous = load(OUT / "gm001/p0/RESULT.json")
        checks["original_R144_scalar_negative_3point535990percent_preserved_count0"] = (
            previous["status"] == "component_screen_negative_below5percent" and previous["reduction"] < .05)
        checks["generic14_interface_integer_bit_and_real_RAW_checks_passed"] = results["g0"]["status"] == (
            "exact_generic_and32RAW_passed_complete128_pending") and len(results["g0"]["checks"]) == 14 and all(results["g0"]["checks"].values())
        checks["three_original_complete128_cold_interleaved_pairs_passed"] = results["s0"]["status"] == (
            "complete128_screen_positive_complete7338_pending") and results["s0"]["reduction"] >= .05 and len(results["s0"]["pairs"]) == 3 and all(
                all(row["checks"].values()) for row in results["s0"]["pairs"])
        checks["real_entry_partial_cache_recovery_corruption_warm_and_external_passed"] = results["v0"]["status"] == (
            "real_entry_interruption_score_recovery_and_external_derived_checks_passed") and all(results["v0"]["checks"].values())
        external = load(base / "v0/EXTERNAL.json")
        checks["external32_derived_component_explicit_development_not_sensor_RAW_GT_blind"] = len(external["rows"]) == 32 and (
            external["derived_RGGB_not_sensor_RAW_or_GT"] and external["development_inputs_not_blind"] and all(
                row["six_fields_and_float_bits_exact"] for row in external["rows"]))
        complete = results["c0"]
        checks["three_fresh_AB_BA_AB_complete7338_pairs_original_schedule_no_restart"] = complete["status"] == (
            "three_complete_fresh_pairs_finished_fees_recovery_review_pending") and len(complete["pairs"]) == 3 and len(complete["runs"]) == 6 and (
                [pair["order"] for pair in complete["pairs"]] == [["current", "vector"], ["vector", "current"], ["current", "vector"]])
        protocol = load(base / "c0/PROTOCOL.json")
        comparisons = []
        for trial, pair in enumerate(complete["pairs"]):
            folder = base / "c0" / ("t" + str(trial))
            protected = {"pair": compare_initial_outputs(folder / "current/initial", folder / "vector/initial"),
                "current_reference": compare_initial_outputs(Path(protocol["reference_initial"]), folder / "current/initial"),
                "vector_reference": compare_initial_outputs(Path(protocol["reference_initial"]), folder / "vector/initial")}
            left, right = folder / "current_quality", folder / "vector_quality"
            names = sorted(path.name for path in left.glob("*.json"))
            equal = len(names) == 7338 and names == sorted(path.name for path in right.glob("*.json")) and all(
                (left / name).read_bytes() == (right / name).read_bytes() for name in names)
            checks["t" + str(trial) + "_all7338_decisions_cache1002_JPEG_K_coarse_poses11583_pairs_exact"] = (
                equal and all(value["all_protected"] for value in protected.values()) and all(pair["checks"].values()))
            comparisons.append({"trial": trial, "protected": protected, "all7338_cache_bytes_exact": equal})
        identities, snapshots = [], []
        for name, result in results.items():
            sources = protocol["sources"] if name == "c0" else result["sources"]
            for source in sources:
                if identity(Path(source["path"])) != source:
                    raise ValueError("An executed original source, DLL or compiler changed")
                identities.append(source)
            for path in (base / name / "source_snapshot").iterdir():
                if name == "b0":
                    original = next(Path(source["path"]) for source in sources if Path(source["path"]).name == path.name)
                else:
                    original = Path(sources[int(path.name.split("_", 1)[0])]["path"])
                    if path.name.split("_", 1)[1] != original.name:
                        raise ValueError("An exact source snapshot index and name differ")
                if path.read_bytes() != original.read_bytes():
                    raise ValueError("An executed code-source snapshot differs from its actual source")
                snapshots.append({"snapshot": str(path), "original": str(original)})
        checks["all_frozen_actual_source_ids_and_executed_code_snapshots_exact"] = len(identities) >= 75 and len(snapshots) >= 65
        related = [Path(__file__), Path(__file__).with_name("vector_gray_stats_evidence.py"), Path(reference.__file__), reference.V1 / "runtime.py"]
        related_ids = [identity(path) for path in related]
        for path in related:
            (output / ("CURRENT_" + path.name)).write_bytes(path.read_bytes())
        telemetry = load(base / "tc0/RESULT.json")
        prior_failed_review = load(base / "r0/FAILURE.json")
        checks["prior_review_failed_on_actual_build_telemetry_preserved_and_strictly_closed"] = (
            telemetry["status"] == "exact_owned_build_telemetry_closed" and all(telemetry["checks"].values())
            and prior_failed_review["error"] == "An exact saved completed process instance is still alive")
        owned = {}
        own_label = output.name.upper()
        for path in base.glob("LEASE_*.json"):
            if path.name == "LEASE_" + own_label + ".json":
                continue
            row = load(path)
            expected_code = 1 if path.name == "LEASE_R0.json" else 0
            if row.get("returncode") != expected_code or not row.get("lease_released_utc"):
                raise ValueError("An actual prerequisite lease did not exit and release")
            for prefix in ("guard", "child"):
                owned[(row[prefix + "_pid"], row[prefix + "_creation_time"]) ] = str(path)
        memory = {"current": [], "vector": []}
        guards = list(base.glob("GUARD_*.json")) + list((base / "c0").glob("t*/*_PROCESS.json"))
        for path in guards:
            if path.name == "GUARD_" + own_label + ".json":
                continue
            row = load(path)
            expected_code = 1 if path.name == "GUARD_R0.json" else 0
            expected_status = "failed_child" if expected_code else "completed"
            if row["status"] != expected_status or row["returncode"] != expected_code or row["elapsed_seconds"] > row["maximum_seconds"] or (
                    row["peak_tree_RSS_bytes"] > row["maximum_RSS_bytes"]):
                raise ValueError("An actual original resource guard failed")
            for child in row.get("owned_process_identities", []):
                owned[(child["pid"], child["creation_time"])] = str(path)
            if path.parent.name in ("t0", "t1", "t2"):
                memory[path.name.removesuffix("_PROCESS.json")].append(row["peak_tree_RSS_bytes"])
        for pid, born in owned:
            try:
                if psutil.Process(pid).create_time() == born:
                    raise ValueError("An exact saved completed process instance is still alive")
            except psutil.NoSuchProcess:
                pass
        checks["all_prior_exact_PID_creation_time_instances_exited_and_guards_passed"] = len(owned) > 20
        for name in ("g0", "s0", "v0"):
            close = load(base / name / "RESOURCE_CLOSE.json")
            if not all(value for key, value in close.items() if isinstance(value, bool) and key != "counted"):
                raise ValueError("An original owned component did not close")
        checks["all_original_component_factories_threads_IPP_and_DLLs_closed"] = True
        output_bytes = sum(path.stat().st_size for path in (base / "c0").rglob("*") if path.is_file())
        checks["full_independent_outputs_below12GiB"] = output_bytes < 12 * 1024**3
        parent = load(ROOT / "research/2026-10-05-gaussian-optimization-round3-50/CURRENT_STATE.json")
        checks["parent26_unique_mechanisms_receipts_unchanged_no_premature_R144_V2_count"] = parent["additional_accepted_count"] == (
            len(parent["accepted_mechanisms"])) == len(set(parent["accepted_mechanisms"])) == len(parent["acceptance_receipts"]) == 26 and (
                all((ROOT / path).is_file() for path in parent["acceptance_receipts"]))
        cost = cost_snapshot(base, complete, time.perf_counter() - started)
        checks["all_current_known_fees_complete_phase_reduction_at_least5percent"] = cost["net_reductions"]["phase_seconds"] >= .05
        checks["all_current_known_fees_whole_median_ratio_at_most1point01"] = cost["whole_median_ratio"] <= 1.01
        checks["each_whole_pair_ratio_at_most1point1"] = all(pair["whole_seconds"]["vector"] <= 1.1 * pair["whole_seconds"]["current"] for pair in complete["pairs"])
        if not all(checks.values()):
            raise ValueError("A separate root source, semantics or cost review contract failed")
        atomic_json(output / "CURRENT_NATIVE_ENVIRONMENT.json", {"python": sys.version, "numpy": np.__version__, "opencv": cv2.__version__,
            "binary": identity(next(Path(cv2.__file__).parent.glob("*.pyd"))), "IPP_enabled": cv2.ipp.useIPP(), "optimized": cv2.useOptimized(),
            "CPU_features": cv2.getCPUFeaturesLine(), "opencv_build_information": cv2.getBuildInformation(), "other_hardware_builds_not_verified": True})
        state.update(status="separate_root_self_review_passed_post_exit_final_cost_and_text_pending", checks=checks,
            scientific_comparisons=comparisons, sources_verified=identities, snapshots_verified=snapshots, related_ids=related_ids,
            current_source_identity_count=len(identities), executed_source_snapshot_count=len(snapshots), actual_prior_instances_exited=len(owned),
            output_bytes=output_bytes, actual_full_guard_peak_tree_RSS_bytes=memory,
            peak_tree_RSS_medians_bytes={name: statistics.median(values) for name, values in memory.items()},
            cost_before_reviewer_exit=cost, source=identity(Path(__file__)), node_seconds=time.perf_counter() - started,
            unverified=["other_CPU_OpenCV_builds", "power_loss_recovery", "GT_metrics", "unseen_object_whole_chain", "new_training", "mapper", "LIO", "production_deployment"])
        atomic_json(output / "RESULT.json", state)
        print({"status": state["status"], "checks": len(checks), "phase_net_reduction": cost["net_reductions"]["phase_seconds"],
            "whole_net_reduction": cost["net_reductions"]["whole_seconds"], "known_shared_seconds": cost["known_extra_fee_seconds"], "counted": False}, flush=True)
    except BaseException as error:
        state.update(status="failed", checks=checks, error_type=type(error).__name__, error=str(error), seconds=time.perf_counter() - started)
        atomic_json(output / "FAILURE.json", state)
        raise


if __name__ == "__main__":
    main()
