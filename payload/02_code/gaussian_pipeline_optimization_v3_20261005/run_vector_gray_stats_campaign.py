"""Run the frozen resumable three-pair original full RAW vector-moments comparison."""

import argparse
from pathlib import Path
import statistics
import subprocess
import sys
import time

import psutil

import profile_quality_corner_cost as reference
from raw_initial_invariants import compare_initial_outputs

ROOT = reference.ROOT
OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005"


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "generic", "pilot", "library", "gray-library", "corner-library", "reference-initial"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--through", type=int, choices=(1, 3), required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(OUT / "vector_gray_stats_001")
    load, atomic_json, identity = reference.load, reference.atomic_json, reference.data_identity
    generic, pilot = load(args.generic / "RESULT.json"), load(args.pilot / "RESULT.json")
    if generic["status"] != "exact_generic_and32RAW_passed_complete128_pending" or not all(generic["checks"].values()) or (
            pilot["status"] != "complete128_screen_positive_complete7338_pending" or pilot["reduction"] < .05):
        raise ValueError("An actually executed frozen vector prerequisite failed")
    names = ("vector_gray_stats.py", "native_vector_gray_stats.cpp", "planar_corner.py", "native_planar_corner.cpp",
        "exact_bayer_gray.py", "native_quality_workspace.py", "quality_histogram.py", "run_vector_gray_stats_input.py",
        "run_exact_raw_gray_input.py", "run_raw_histogram_quality.py", "run_raw_calibration.py", "calibration_body_cache.py",
        "run_bounded_journal_task.py")
    paths = [Path(__file__), *[Path(__file__).with_name(name) for name in names], reference.V1 / "quality_engine.py",
        reference.V1 / "run_initial_pipeline.py", reference.V1 / "selection_engine.py", reference.V1 / "pose_interpolation.py",
        *[reference.V2 / name for name in ("run_effective_initial.py", "raw_initial_invariants.py", "indexed_candidate_pairs.py",
            "parallel_raw_export.py", "single_copy_raw_reader.py", "encoded_jpeg_cache.py", "windowed_raw_reader.py")],
        args.library.resolve(strict=True), args.gray_library.resolve(strict=True), args.corner_library.resolve(strict=True)]
    sources = [identity(path) for path in paths]
    by_path = {row["path"]: row for row in sources}
    for prereq in (generic, pilot):
        for row in prereq["sources"]:
            if row["path"] in by_path and by_path[row["path"]] != row:
                raise ValueError("An executed vector scientific source changed")
    if output.exists():
        protocol, state = load(output / "PROTOCOL.json"), load(output / "STATE.json")
        if args.through != 3 or state["status"] != "first_complete_pair_positive_resume_same_frozen_schedule" or (
                len(state["pairs"]) != 1 or protocol["sources"] != sources or protocol["library"] != str(args.library.resolve())
                or protocol["gray_library"] != str(args.gray_library.resolve()) or protocol["corner_library"] != str(args.corner_library.resolve())
                or protocol["reference_initial"] != str(args.reference_initial.resolve())):
            raise ValueError("Only the original positive first pair can resume the frozen schedule")
        state["status"] = "running"
    else:
        output.mkdir(parents=True, exist_ok=False)
        protocol = {"sources": sources, "library": str(args.library.resolve()), "gray_library": str(args.gray_library.resolve()),
            "corner_library": str(args.corner_library.resolve()), "generic": str(args.generic.resolve()), "pilot": str(args.pilot.resolve()),
            "reference_initial": str(args.reference_initial.resolve()), "schedule": [["current", "vector"], ["vector", "current"], ["current", "vector"]],
            "raw_frames": 7338, "eligible": 6797, "selected": 1002, "pairs": 11583, "workers": 4, "capacity": 16,
            "minimum_after_all_known_fees_complete_phase_reduction": .05, "fee_amortization_sides": 6,
            "whole_median_ratio_limit": 1.01, "each_whole_ratio_limit": 1.1, "independent_output_budget_bytes": 12 * 1024**3,
            "each_leaf_time_seconds": 600, "each_leaf_RAM_bytes": 8 * 1024**3, "controller_per_invocation_seconds": 1200,
            "R147_R149_R141_R104_Q78_R99_Q50_Q53_P18_common_incumbent": True, "original_scoring_and_JPEG_cache_off": True,
            "all_first_failed_generic_pilot_recovery_review_fees_retained": True, "counted": False}
        atomic_json(output / "PROTOCOL.json", protocol)
        snapshot = output / "source_snapshot"
        snapshot.mkdir()
        for position, path in enumerate(paths[:-3]):
            (snapshot / (str(position) + "_" + path.name)).write_bytes(path.read_bytes())
        state = {"status": "running", "pairs": [], "runs": [], "invocations": [], "accepted_count_contribution": 0}
    process = psutil.Process()
    atomic_json(output / ("PROCESS_IDENTITY_" + str(args.through) + ".json"), {"pid": process.pid, "creation_time": process.create_time()})
    invocation = {"through": args.through, "pair_count_before": len(state["pairs"]), "status": "running"}
    state["invocations"].append(invocation)
    leaf_total = 0.0
    try:
        for trial in range(len(state["pairs"]), args.through):
            folder = output / ("t" + str(trial))
            folder.mkdir()
            results = {}
            for mode in protocol["schedule"][trial]:
                if time.perf_counter() - started > 1200 or sources != [identity(path) for path in paths]:
                    raise ValueError("The original campaign source or time guard changed")
                command = [sys.executable, str(Path(__file__).with_name("run_bounded_journal_task.py")), "--receipt",
                    str(folder / (mode + "_PROCESS.json")), "--log", str(folder / (mode + ".log")), "--maximum-seconds", "600",
                    "--maximum-rss-gib", "8", "--", sys.executable, str(Path(__file__).with_name("run_vector_gray_stats_input.py")),
                    "--mode", mode, "--library", str(args.library), "--gray-library", str(args.gray_library),
                    "--corner-library", str(args.corner_library), "--output", str(folder / mode), "--quality-cache", str(folder / (mode + "_quality"))]
                run = {"trial": trial, "mode": mode, "status": "running", "command": command}
                state["runs"].append(run)
                atomic_json(output / "STATE.json", state)
                before = time.perf_counter()
                with (folder / (mode + "_controller.log")).open("x", encoding="utf-8") as stream:
                    child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
                    run.update(pid=child.pid, creation_time=psutil.Process(child.pid).create_time())
                    atomic_json(output / "STATE.json", state)
                    code = child.wait()
                whole = time.perf_counter() - before
                leaf_total += whole
                run.update(returncode=code, whole_seconds=whole, status="completed" if code == 0 else "failed")
                atomic_json(output / "STATE.json", state)
                if code:
                    raise RuntimeError("An actual full7338 original consumer failed")
                output_bytes = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
                if output_bytes > protocol["independent_output_budget_bytes"]:
                    raise RuntimeError("The registered full output budget was exceeded")
                run["isolated_output_bytes_after_leaf"] = output_bytes
                fused = load(folder / mode / "FUSED_INPUT.json")
                histogram = load(folder / mode / "HISTOGRAM.json")
                corner = load(folder / mode / "CORNER_INPUT.json")
                vector = load(folder / mode / "VECTOR_INPUT.json")
                receipt = load(folder / mode / "initial/SELECTION_RECEIPT.json")
                results[mode] = {"whole_seconds": whole, "phase_seconds": receipt["timings"]["raw_image_scoring_wall_seconds"]
                    + fused["additional_input_lifecycle_seconds"] + histogram["additional_quality_lifecycle_seconds"]
                    + corner["additional_corner_lifecycle_seconds"] + vector["additional_statistics_lifecycle_seconds"],
                    "fused_input": fused, "histogram": histogram, "corner_input": corner, "vector_input": vector, "detail": receipt}
                print({"completed_leaf": mode, "trial": trial, "whole_seconds": whole, "phase_seconds": results[mode]["phase_seconds"]}, flush=True)
            before = time.perf_counter()
            invariants = {"pair": compare_initial_outputs(folder / "current/initial", folder / "vector/initial"),
                "current_reference": compare_initial_outputs(args.reference_initial, folder / "current/initial"),
                "vector_reference": compare_initial_outputs(args.reference_initial, folder / "vector/initial")}
            left, right = folder / "current_quality", folder / "vector_quality"
            files = sorted(path.name for path in left.iterdir())
            cache_equal = len(files) == 7338 and files == sorted(path.name for path in right.iterdir()) and all(
                (left / name).read_bytes() == (right / name).read_bytes() for name in files)
            comparison = time.perf_counter() - before
            vector = results["vector"]["vector_input"]["trace"]
            checks = {"all7338_original_decisions1002_JPEG_K_poses_pairs_and_reference_exact": all(value["all_protected"] for value in invariants.values()),
                "all7338_original_quality_cache_files_byte_exact": cache_equal,
                "both_actual_original_cold_denominators": all((value["detail"]["raw_frame_count"], value["detail"]["quality_and_coarse_pose_eligible_count"],
                    value["detail"]["selected_count"], value["detail"]["matching_candidate_pair_count"], value["detail"]["timings"]["raw_image_scoring_detail"]["cache_hits"])
                    == (7338, 6797, 1002, 11583, 0) for value in results.values()),
                "both_original_aliases_compilers_cv2_and_gray_DLL_restored": all(value["fused_input"]["original_quality_alias_restored"]
                    and value["histogram"]["trace"]["aliases_restored"] and value["fused_input"]["trace"]["builder_restored"]
                    and value["fused_input"]["trace"]["engine"]["DLL_unloaded"] and value["corner_input"]["original_factory_restored"]
                    and value["corner_input"]["original_cv2_module_unchanged"] and value["vector_input"]["original_factory_restored"] for value in results.values()),
                "both_incumbent7338_planar_corner_calls_no_fallback_closed": all(value["corner_input"]["trace"]["workspace"]["native_calls"] == 7338
                    and not value["corner_input"]["trace"]["workspace"]["fallback_calls"] and not value["corner_input"]["trace"]["workspace"]["active_calls"]
                    and not value["corner_input"]["trace"]["workspace"]["retained_plane_bytes"] and value["corner_input"]["trace"]["workspace"]["DLL_unloaded"] for value in results.values()),
                "vector_all7338_native_stats_no_fallback_outputs_and_DLL_closed": vector["engine"]["native_calls"] == 7338
                    and not vector["engine"]["fallback_calls"] and not vector["engine"]["active_calls"]
                    and not vector["engine"]["retained_output_bytes"] and vector["engine"]["DLL_unloaded"],
                "each_whole_at_most1point1": results["vector"]["whole_seconds"] <= 1.1 * results["current"]["whole_seconds"]}
            pair = {"trial": trial, "order": protocol["schedule"][trial], "results": results,
                "comparison_seconds_charged_each_side": comparison, "invariants": invariants, "checks": checks,
                "phase_seconds": {mode: results[mode]["phase_seconds"] + comparison for mode in results},
                "whole_seconds": {mode: results[mode]["whole_seconds"] + comparison for mode in results}}
            state["pairs"].append(pair)
            atomic_json(output / "STATE.json", state)
            print({"trial": trial, "phase_seconds": pair["phase_seconds"], "whole_seconds": pair["whole_seconds"], "checks": checks}, flush=True)
            if not all(checks.values()):
                raise ValueError("A full original RAW science, lifecycle or cost contract failed")
            if trial == 0 and 1 - pair["phase_seconds"]["vector"] / pair["phase_seconds"]["current"] < .05:
                state["status"] = "first_complete_pair_negative_stop_no_formal_acceptance"
                break
        if state["status"] == "running":
            state["status"] = "first_complete_pair_positive_resume_same_frozen_schedule" if args.through == 1 else "three_complete_fresh_pairs_finished_fees_recovery_review_pending"
        if len(state["pairs"]) == 3:
            state["medians"] = {name: {mode: statistics.median(pair[name][mode] for pair in state["pairs"])
                for mode in ("current", "vector")} for name in ("phase_seconds", "whole_seconds")}
            state["reductions_before_shared_fees"] = {name: 1 - value["vector"] / value["current"] for name, value in state["medians"].items()}
        invocation.update(status="completed", elapsed_seconds=time.perf_counter() - started, leaf_seconds=leaf_total,
            nonleaf_seconds=time.perf_counter() - started - leaf_total)
        atomic_json(output / "STATE.json", state)
        atomic_json(output / "RESULT.json", state)
        print({"status": state["status"], "complete_pairs": len(state["pairs"]), "counted": False}, flush=True)
    except BaseException as error:
        state.update(status="failed", error_type=type(error).__name__, error=str(error))
        invocation.update(status="failed", elapsed_seconds=time.perf_counter() - started, leaf_seconds=leaf_total)
        atomic_json(output / "STATE.json", state)
        atomic_json(output / "FAILURE.json", state)
        raise


if __name__ == "__main__":
    main()
