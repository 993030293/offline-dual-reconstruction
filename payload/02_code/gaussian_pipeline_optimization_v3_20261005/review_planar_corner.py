"""Perform a separate root self-review of the frozen whole RAW corner experiment."""

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

ROOT=reference.ROOT
OUT=ROOT/"04_outputs/gaussian_pipeline_optimization_v3_20261005"


def cost_snapshot(base,complete,extra_seconds=0.):
    rows=[]
    for path in sorted(base.glob("LEASE_*.json")):
        lease=reference.load(path)
        if "child_elapsed_seconds" in lease:
            rows.append({"path":str(path),"seconds":lease["child_elapsed_seconds"],"returncode":lease["returncode"]})
    leaf=sum(row["whole_seconds"] for row in complete["runs"])
    comparison=sum(row["comparison_seconds_charged_each_side"] for row in complete["pairs"])
    shared=sum(row["seconds"] for row in rows)-leaf-comparison+extra_seconds
    if shared<0:
        raise ValueError("The actual whole corner fee ledger became negative")
    medians={name:{mode:statistics.median(row[name][mode] for row in complete["pairs"])+shared/6
        for mode in ("current","planar")} for name in ("phase_seconds","whole_seconds")}
    return {"lease_rows":rows,"leaf_windows_already_counted_seconds":leaf,"physical_comparison_already_counted_seconds":comparison,
        "known_extra_fee_seconds":shared,"amortization_sides":6,"complete_known_medians":medians,
        "net_reductions":{name:1-values["planar"]/values["current"] for name,values in medians.items()},
        "whole_median_ratio":medians["whole_seconds"]["planar"]/medians["whole_seconds"]["current"],
        "text_research_wall_time_not_metered":True,"pending_stdout_exit_not_assumed_zero":True}


def main():
    started=time.perf_counter()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    output=parser.parse_args().output.resolve()
    output.relative_to(OUT/"qc001")
    output.mkdir(parents=True,exist_ok=False)
    load,atomic_json,identity=reference.load,reference.atomic_json,reference.data_identity
    base=OUT/"qc001"
    process=psutil.Process()
    atomic_json(output/"PROCESS_IDENTITY.json",{"pid":process.pid,"creation_time":process.create_time()})
    state={"status":"running","accepted_count_contribution":0,"review_type":"separate_root_self_review_not_independent_agent"}
    checks={}
    try:
        results={name:load(base/name/"RESULT.json") for name in ("p0","i0","b0","g0","s0","v0","c0")}
        checks["original_cost_diagnostic_all128_protected32_corners_below_cap_and_unimplemented_prefix0"]=all(results["p0"]["checks"].values()) and results["p0"]["upper800_count"]==0 and results["p0"]["accepted_count_contribution"]==0
        checks["R148_IPP_original48_generic32_real_exact_but_cost_negative_preserved0"]=results["i0"]["status"]=="closed_negative_component_speed_insufficient" and all(results["i0"]["checks"].values()) and results["i0"]["component_reduction"]<.05 and results["i0"]["accepted_count_contribution"]==0
        checks["R149_actual_generic_and_RAW128_sources_science_and_component_passed"]=all(results["g0"]["checks"].values()) and results["g0"]["status"]=="component_passed_complete_stage_pending" and results["s0"]["status"]=="complete128_screen_positive_complete7338_pending" and len(results["s0"]["pairs"])==3 and all(all(pair["checks"].values()) for pair in results["s0"]["pairs"])
        checks["all29_original_interface_real_entry_fault_cache_recovery_external_checks_passed"]=results["v0"]["status"]=="original_interface_real_entry_interruption_score_recovery_and_external_derived_operator_checks_passed" and len(results["v0"]["checks"])==29 and all(results["v0"]["checks"].values())
        external=load(base/"v0/EXTERNAL.json")
        checks["external32_component_derived_inputs_marked_development_not_RAW_GT_or_blind"]=len(external["rows"])==32 and external["derived_RGGB_not_sensor_RAW_or_GT"] and external["development_inputs_not_blind"] and all(row["response_coordinates_order_and_six_fields_exact"] for row in external["rows"])
        complete=results["c0"]
        checks["three_original_AB_BA_AB_fresh_complete7338_pairs_no_restart"]=complete["status"]=="three_complete_fresh_pairs_finished_fees_recovery_review_pending" and len(complete["pairs"])==3 and len(complete["runs"])==6 and [pair["order"] for pair in complete["pairs"]]==[["current","planar"],["planar","current"],["current","planar"]]
        protocol=load(base/"c0/PROTOCOL.json")
        frozen_reference=Path(protocol["reference_initial"])
        comparisons=[]
        for trial,pair in enumerate(complete["pairs"]):
            folder=base/"c0"/("t"+str(trial))
            protected={"pair":compare_initial_outputs(folder/"current/initial",folder/"planar/initial"),
                "current_reference":compare_initial_outputs(frozen_reference,folder/"current/initial"),
                "planar_reference":compare_initial_outputs(frozen_reference,folder/"planar/initial")}
            left,right=folder/"current_quality",folder/"planar_quality"
            names=sorted(path.name for path in left.glob("*.json"))
            equal=len(names)==7338 and names==sorted(path.name for path in right.glob("*.json")) and all((left/name).read_bytes()==(right/name).read_bytes() for name in names)
            checks["t"+str(trial)+"_all_original7338_cache_decisions1002_JPEG_K_metric_poses_and11583_pairs_exact"]=equal and all(value["all_protected"] for value in protected.values()) and all(pair["checks"].values())
            comparisons.append({"trial":trial,"protected":protected,"all7338_cache_byte_equal":equal})
        identities=[]
        snapshots=[]
        for name,result in results.items():
            sources=protocol["sources"] if name=="c0" else result["sources"]
            for source in sources:
                if identity(Path(source["path"]))!=source:
                    raise ValueError("An executed original corner source, DLL or compiler changed")
                identities.append(source)
            folder=base/name/"source_snapshot"
            for path in folder.iterdir():
                if name=="b0":
                    original=next(Path(source["path"]) for source in sources if Path(source["path"]).name==path.name)
                else:
                    original=Path(sources[int(path.name.split("_",1)[0])]["path"])
                if path.read_bytes()!=original.read_bytes():
                    raise ValueError("An executed scientific text-source snapshot differs from its frozen source")
                snapshots.append({"snapshot":str(path),"original":str(original)})
        checks["all_executed_current_sources_DLLs_compiler_and_actual_code_text_snapshots_exact"]=len(identities)>60 and len(snapshots)>50
        original={}
        for path in base.glob("LEASE_*.json"):
            if path.name=="LEASE_R0.json":
                continue
            row=load(path)
            if row.get("returncode")!=0 or not row.get("lease_released_utc"):
                raise ValueError("A real prerequisite lease did not exit and release")
            for prefix in ("guard","child"):
                original[(row[prefix+"_pid"],row[prefix+"_creation_time"])]=str(path)
        guards=list(base.glob("GUARD_*.json"))+list((base/"c0").glob("t*/*_PROCESS.json"))
        memory={"current":[],"planar":[]}
        for path in guards:
            if path.name=="GUARD_R0.json":
                continue
            row=load(path)
            if row["status"]!="completed" or row["returncode"]!=0 or row["peak_tree_RSS_bytes"]>row["maximum_RSS_bytes"] or row["elapsed_seconds"]>row["maximum_seconds"]:
                raise ValueError("An actual original process or resource guard failed")
            for child in row.get("owned_process_identities",[]):
                original[(child["pid"],child["creation_time"])]=str(path)
            if path.parent.name in ("t0","t1","t2"):
                memory[path.name.removesuffix("_PROCESS.json")].append(row["peak_tree_RSS_bytes"])
        for pid,born in original:
            try:
                if psutil.Process(pid).create_time()==born:
                    raise ValueError("A completed saved process instance is still alive")
            except psutil.NoSuchProcess:
                pass
        checks["all_prior_exact_PID_creation_time_instances_exited_and_guards_passed"]=len(original)>20
        for name in ("p0","i0","g0","s0","v0"):
            close=load(base/name/"RESOURCE_CLOSE.json")
            if not all(value for key,value in close.items() if isinstance(value,bool) and key!="counted"):
                raise ValueError("An original owned component lifecycle did not close")
        checks["all_component_original_threads_factories_IPP_and_DLL_close_receipts_passed"]=True
        output_bytes=sum(path.stat().st_size for path in (base/"c0").rglob("*") if path.is_file())
        checks["actual_fresh_full_outputs_below12GiB"]=output_bytes<12*1024**3
        parent=load(ROOT/"research/2026-10-05-gaussian-optimization-round3-50/CURRENT_STATE.json")
        checks["parent25_unique_accepted_mechanisms_receipts_preserved_no_R149_premature_count"]=parent["additional_accepted_count"]==len(parent["accepted_mechanisms"])==len(set(parent["accepted_mechanisms"]))==len(parent["acceptance_receipts"])==25 and parent["R149_new_optimization_count"]==0 and all((ROOT/path).is_file() for path in parent["acceptance_receipts"])
        cost=cost_snapshot(base,complete,time.perf_counter()-started)
        checks["all_current_known_fee_phase_reduction_at_least5percent"]=cost["net_reductions"]["phase_seconds"]>=.05
        checks["all_current_known_fee_whole_median_ratio_at_most1point01"]=cost["whole_median_ratio"]<=1.01
        checks["each_actual_whole_pair_ratio_at_most1point1"]=all(pair["whole_seconds"]["planar"]<=1.1*pair["whole_seconds"]["current"] for pair in complete["pairs"])
        if not all(checks.values()):
            raise ValueError("A separate original corner source, semantic or cost review contract failed")
        binary=next(Path(cv2.__file__).parent.glob("*.pyd"))
        atomic_json(output/"CURRENT_NATIVE_ENVIRONMENT.json",{"python":sys.version,"numpy":np.__version__,"opencv":cv2.__version__,
            "binary":identity(binary),"IPP_enabled":cv2.ipp.useIPP(),"optimized":cv2.useOptimized(),"CPU_features":cv2.getCPUFeaturesLine(),
            "opencv_build_information":cv2.getBuildInformation(),"other_hardware_and_builds_not_verified":True})
        state.update(status="separate_root_self_review_passed_post_exit_final_cost_and_text_publication_pending",checks=checks,
            scientific_comparisons=comparisons,sources_verified=identities,snapshots_verified=snapshots,
            current_source_identity_count=len(identities),executed_text_source_snapshot_count=len(snapshots),actual_prior_instances_exited=len(original),
            output_bytes=output_bytes,actual_full_guard_peak_tree_RSS_bytes=memory,
            peak_tree_RSS_medians_bytes={name:statistics.median(values) for name,values in memory.items()},
            cost_before_reviewer_exit=cost,source=identity(Path(__file__)),node_seconds=time.perf_counter()-started,
            unverified=["layout_vs_postselection_individual_speed_attribution","other_CPU_or_OpenCV_builds","power_loss_recovery","GT_metrics","unseen_object_whole_chain","new_training","mapper","LIO_replay","production_deployment"])
        atomic_json(output/"RESULT.json",state)
        print({"status":state["status"],"checks":len(checks),"phase_net_reduction":cost["net_reductions"]["phase_seconds"],
            "whole_net_reduction":cost["net_reductions"]["whole_seconds"],"known_shared_seconds":cost["known_extra_fee_seconds"],"counted":False},flush=True)
    except BaseException as error:
        state.update(status="failed",checks=checks,error_type=type(error).__name__,error=str(error),seconds=time.perf_counter()-started)
        atomic_json(output/"FAILURE.json",state)
        raise


if __name__=="__main__":
    main()
