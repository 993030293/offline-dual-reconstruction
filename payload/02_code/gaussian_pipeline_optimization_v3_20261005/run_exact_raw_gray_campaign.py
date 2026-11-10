"""Measure a preregistered resumable three-pair complete RAW initial consumer."""

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"
V2 = ROOT / "02_code/gaussian_pipeline_optimization_v2_20261005"
sys.path[:0] = [str(V1), str(V2)]
from runtime import atomic_json, data_identity
from raw_initial_invariants import compare_initial_outputs


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "generic", "pilot", "library", "reference-initial"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--through", type=int, choices=(1,3), required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/bg001")
    generic, pilot = load(args.generic / "RESULT.json"), load(args.pilot / "RESULT.json")
    if generic["status"] != "generic_passed_component_and_complete_stage_pending" or not all(generic["checks"].values()) or pilot[
            "status"] != "component_screen_positive_complete_stage_pending" or pilot["reduction"] < .05:
        raise ValueError("A registered generic or complete original RAW128 prerequisite failed")
    filenames = ("exact_bayer_gray.py", "native_bayer_gray.cpp", "native_quality_workspace.py", "run_exact_raw_gray_input.py",
                 "run_raw_histogram_quality.py", "quality_histogram.py", "run_raw_calibration.py", "calibration_body_cache.py",
                 "run_bounded_journal_task.py")
    paths = [Path(__file__), *[Path(__file__).with_name(name) for name in filenames], V1 / "quality_engine.py", V1 / "run_initial_pipeline.py",
        V1 / "selection_engine.py", V1 / "pose_interpolation.py", V2 / "run_effective_initial.py", V2 / "raw_initial_invariants.py",
        *[V2 / name for name in ("indexed_candidate_pairs.py", "parallel_raw_export.py", "single_copy_raw_reader.py", "encoded_jpeg_cache.py", "windowed_raw_reader.py")],
        args.library.resolve(strict=True)]
    sources = [data_identity(path) for path in paths]
    if not all(sources[left] == generic["sources"][right] == pilot["sources"][pilot_index]
               for left,right,pilot_index in ((1,1,1),(2,2,2),(3,3,3),(6,4,4),(10,5,5),(21,7,11))):
        raise ValueError("An exact input scientific source differs from its executed prerequisites")
    if output.exists():
        protocol, state = load(output / "PROTOCOL.json"), load(output / "STATE.json")
        if args.through != 3 or state["status"] != "first_complete_pair_positive_resume_same_frozen_schedule" or len(state["pairs"]) != 1 or protocol[
                "sources"] != sources or protocol["library"] != str(args.library.resolve()) or protocol["reference_initial"] != str(args.reference_initial.resolve()):
            raise ValueError("Only the original positive pair can resume the same frozen three-pair schedule")
        state["status"] = "running"
    else:
        output.mkdir(parents=True, exist_ok=False)
        protocol = {"sources": sources, "library": str(args.library.resolve()), "reference_initial": str(args.reference_initial.resolve()),
            "generic": str(args.generic.resolve()), "pilot": str(args.pilot.resolve()),
            "schedule": [["native","fused"],["fused","native"],["native","fused"]],
            "raw_frames":7338, "eligible":6797, "selected":1002, "pairs":11583, "workers":4,
            "minimum_after_all_known_fees_complete_phase_reduction":.05, "fee_amortization_sides":6,
            "whole_median_ratio_limit":1.01, "each_whole_ratio_limit":1.1,
            "first_complete_pair_raw_phase_reduction_at_least5percent_and_whole_at_most1point1_to_continue":True,
            "independent_output_budget_bytes":12*1024**3, "each_leaf_time_seconds":600, "each_leaf_RAM_bytes":8*1024**3,
            "controller_per_invocation_seconds":1200, "control_current_Q78_R99_R104_R141_Q50_Q53_P18":True,
            "original_scoring_cache_and_JPEG_cache_off":True, "all_first_failed_generic_pilot_recovery_review_fees_retained":True, "counted":False}
        atomic_json(output / "PROTOCOL.json", protocol)
        snapshot = output / "source_snapshot"
        snapshot.mkdir()
        for position, path in enumerate(paths[:-1]):
            (snapshot / (str(position)+"_"+path.name)).write_bytes(path.read_bytes())
        state = {"status":"running", "pairs":[], "runs":[], "invocations":[], "accepted_count_contribution":0}
    invocation = {"through":args.through,"pair_count_before":len(state["pairs"]),"status":"running"}
    state["invocations"].append(invocation)
    process = psutil.Process()
    atomic_json(output / ("PROCESS_IDENTITY_"+str(args.through)+".json"), {"pid":process.pid,"creation_time":process.create_time(),"scope":"Windows"})
    leaf_total = 0.
    try:
        for trial in range(len(state["pairs"]),args.through):
            folder=output/("t"+str(trial))
            folder.mkdir()
            results={}
            for mode in protocol["schedule"][trial]:
                if time.perf_counter()-started > 1200 or sources != [data_identity(path) for path in paths]:
                    raise ValueError("The original complete campaign time or source guard changed")
                command=[sys.executable,str(Path(__file__).with_name("run_bounded_journal_task.py")),"--receipt",str(folder/(mode+"_PROCESS.json")),
                    "--log",str(folder/(mode+".log")),"--maximum-seconds","600","--maximum-rss-gib","8","--",sys.executable,
                    str(Path(__file__).with_name("run_exact_raw_gray_input.py")),"--mode",mode,"--library",str(args.library),
                    "--output",str(folder/mode),"--quality-cache",str(folder/(mode+"_quality"))]
                run={"trial":trial,"mode":mode,"status":"running","command":command}
                state["runs"].append(run)
                atomic_json(output/"STATE.json",state)
                before=time.perf_counter()
                with (folder/(mode+"_controller.log")).open("x",encoding="utf-8") as stream:
                    child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT)
                    run.update(pid=child.pid,creation_time=psutil.Process(child.pid).create_time())
                    atomic_json(output/"STATE.json",state)
                    code=child.wait()
                whole=time.perf_counter()-before
                leaf_total+=whole
                run.update(returncode=code,whole_seconds=whole,status="completed" if code==0 else "failed")
                atomic_json(output/"STATE.json",state)
                if code:
                    raise RuntimeError("A complete actual original RAW consumer failed")
                output_bytes = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
                if output_bytes > protocol["independent_output_budget_bytes"]:
                    raise RuntimeError("The registered isolated complete output budget was exceeded")
                run["isolated_campaign_output_bytes_after_leaf"] = output_bytes
                fused=load(folder/mode/"FUSED_INPUT.json")
                histogram=load(folder/mode/"HISTOGRAM.json")
                receipt=load(folder/mode/"initial/SELECTION_RECEIPT.json")
                results[mode]={"whole_seconds":whole,"phase_seconds":receipt["timings"]["raw_image_scoring_wall_seconds"]+
                    fused["additional_input_lifecycle_seconds"]+histogram["additional_quality_lifecycle_seconds"],
                    "fused_input":fused,"histogram":histogram,"detail":receipt}
                print({"completed_leaf":mode,"trial":trial,"whole_seconds":whole,"phase_seconds":results[mode]["phase_seconds"]},flush=True)
            before=time.perf_counter()
            invariants={"pair":compare_initial_outputs(folder/"native/initial",folder/"fused/initial"),
                "native_reference":compare_initial_outputs(args.reference_initial,folder/"native/initial"),
                "fused_reference":compare_initial_outputs(args.reference_initial,folder/"fused/initial")}
            left,right=folder/"native_quality",folder/"fused_quality"
            names=sorted(path.name for path in left.iterdir())
            cache_equal=len(names)==7338 and names==sorted(path.name for path in right.iterdir()) and all(
                (left/name).read_bytes()==(right/name).read_bytes() for name in names)
            comparison=time.perf_counter()-before
            fused_trace=results["fused"]["fused_input"]["trace"]
            native_trace=results["native"]["fused_input"]["trace"]
            checks={"all_original7338_decisions1002_JPEG_pairs_K_poses_and_current_reference_exact":all(value["all_protected"] for value in invariants.values()),
                "all7338_persistent_quality_cache_files_bytes_exact":cache_equal,
                "both_actual_cold_RAW_denominators":all((value["detail"]["raw_frame_count"],value["detail"]["quality_and_coarse_pose_eligible_count"],
                    value["detail"]["selected_count"],value["detail"]["matching_candidate_pair_count"],value["detail"]["timings"]["raw_image_scoring_detail"]["cache_hits"])
                    ==(7338,6797,1002,11583,0) for value in results.values()),
                "both_original_aliases_restored":all(value["fused_input"]["original_quality_alias_restored"] and value["histogram"]["trace"]["aliases_restored"] for value in results.values()),
                "original_workspace_and_fused_builder_restored":native_trace["original_builder_restored"] and fused_trace["builder_restored"],
                "fused_all7338_DLL_calls_original_planes_and_pool_closed":fused_trace["engine"]["calls"]==7338 and fused_trace["engine"]["DLL_unloaded"]
                    and fused_trace["engine"]["active_calls"]==0 and fused_trace["fused_workspace"]["calls"]==3*7338 and fused_trace[
                    "fused_workspace"]["owned_plane_bytes_after_close"]==0 and fused_trace["fallback_workspace"]["calls"]==0,
                "each_whole_at_most1point1":results["fused"]["whole_seconds"] <=1.1*results["native"]["whole_seconds"]}
            pair={"trial":trial,"order":protocol["schedule"][trial],"results":results,"comparison_seconds_charged_each_side":comparison,
                "invariants":invariants,"checks":checks,
                "phase_seconds":{mode:results[mode]["phase_seconds"]+comparison for mode in results},
                "whole_seconds":{mode:results[mode]["whole_seconds"]+comparison for mode in results}}
            state["pairs"].append(pair)
            atomic_json(output/"STATE.json",state)
            print({"trial":trial,"phase_seconds":pair["phase_seconds"],"whole_seconds":pair["whole_seconds"],"checks":checks},flush=True)
            if not all(checks.values()):
                raise ValueError("A complete original RAW paired scientific or performance contract failed")
            if trial==0 and 1-pair["phase_seconds"]["fused"]/pair["phase_seconds"]["native"] < .05:
                state["status"]="first_complete_pair_negative_stop_no_formal_acceptance"
                break
        if state["status"]=="running":
            state["status"]="first_complete_pair_positive_resume_same_frozen_schedule" if args.through==1 else "three_complete_fresh_pairs_finished_fees_recovery_review_pending"
        if len(state["pairs"])==3:
            state["medians"]={name:{mode:statistics.median(pair[name][mode] for pair in state["pairs"]) for mode in ("native","fused")}
                for name in ("phase_seconds","whole_seconds")}
            state["reductions_before_all_shared_fees"]={name:1-value["fused"]/value["native"] for name,value in state["medians"].items()}
        invocation.update(status="completed",elapsed_seconds=time.perf_counter()-started,leaf_seconds=leaf_total,
                          nonleaf_seconds=time.perf_counter()-started-leaf_total)
        atomic_json(output/"STATE.json",state)
        atomic_json(output/"RESULT.json",state)
        print({"status":state["status"],"complete_pairs":len(state["pairs"]),"counted":False},flush=True)
    except BaseException as error:
        state.update(status="failed",error_type=type(error).__name__,error=str(error))
        invocation.update(status="failed",elapsed_seconds=time.perf_counter()-started,leaf_seconds=leaf_total)
        atomic_json(output/"STATE.json",state)
        atomic_json(output/"FAILURE.json",state)
        raise


if __name__=="__main__":
    main()
